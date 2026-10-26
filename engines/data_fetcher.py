"""
Binance USDT-M Futures Public Historical Data Backfill & Cache
Fetches:
1. Public `/fapi/v1/klines` (includes exact aggTrades-derived taker_buy_base_volume for CVD)
2. Public `/fapi/v1/fundingRate` (historical 8h funding rates)
Saves locally to `data/raw/` as CSV/Parquet for 100% deterministic replay.
"""
import os
import time
from typing import Dict, List, Optional
import requests
import pandas as pd
import numpy as np
from config import CONFIG, Phase1Config

BINANCE_FAPI_BASE = "https://fapi.binance.com"


class BinanceDataFetcher:
    def __init__(self, cfg: Phase1Config = CONFIG, data_dir: str = "data/raw"):
        self.cfg = cfg
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.session = requests.Session()

    def _klines_cache_path(self, symbol: str, interval: str, days: int) -> str:
        return os.path.join(self.data_dir, f"{symbol}_{interval}_{days}d_klines.csv")

    def _funding_cache_path(self, symbol: str, days: int) -> str:
        return os.path.join(self.data_dir, f"{symbol}_{days}d_funding.csv")

    def _rows_to_klines_df(self, all_rows: List[ list ], now_ms: int) -> pd.DataFrame:
        cols = [
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_ms",
            "quote_volume",
            "trade_count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        df = pd.DataFrame(all_rows, columns=cols)
        df["open_time_ms"] = df["open_time_ms"].astype(np.int64)
        df["close_time_ms"] = df["close_time_ms"].astype(np.int64)
        df = df.drop_duplicates(subset=["open_time_ms"]).sort_values("open_time_ms")

        # Drop the currently incomplete live candle (last row if close_time_ms > now_ms)
        df = df[df["close_time_ms"] < now_ms].copy()

        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        ]
        for c in numeric_cols:
            df[c] = df[c].astype(float)
        df["trade_count"] = df["trade_count"].astype(int)

        df["taker_sell_volume"] = np.maximum(0.0, df["volume"] - df["taker_buy_volume"])
        df["volume_delta"] = df["taker_buy_volume"] - df["taker_sell_volume"]
        df["delta_ratio"] = np.where(
            df["volume"] > 0, df["volume_delta"] / df["volume"], 0.0
        )
        df["cvd"] = df["volume_delta"].cumsum()
        df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
        return df.drop(columns=["ignore"]).reset_index(drop=True)

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "15m",
        days: int = 180,
        use_cache: bool = True,
        refresh_tail: bool = False,
    ) -> pd.DataFrame:
        cache_path = self._klines_cache_path(symbol, interval, days)
        now_ms = int(time.time() * 1000)
        interval_ms = 3600_000 if interval == "1h" else (900_000 if interval == "15m" else 3600_000)

        if use_cache and os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df["open_time"] = pd.to_datetime(df["open_time"], format="ISO8601", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], format="ISO8601", utc=True)
            df = df.sort_values("open_time").reset_index(drop=True)

            if refresh_tail and len(df) > 0:
                last_close_ms = int(df["close_time_ms"].iloc[-1])
                if now_ms - last_close_ms >= interval_ms:
                    try:
                        tail_rows = []
                        cursor_ms = last_close_ms + 1
                        while cursor_ms < now_ms:
                            resp = self.session.get(
                                f"{BINANCE_FAPI_BASE}/fapi/v1/klines",
                                params={
                                    "symbol": symbol,
                                    "interval": interval,
                                    "startTime": cursor_ms,
                                    "endTime": now_ms,
                                    "limit": 1500,
                                },
                                timeout=10,
                            )
                            resp.raise_for_status()
                            batch = resp.json()
                            if not batch:
                                break
                            tail_rows.extend(batch)
                            if int(batch[-1][0]) <= cursor_ms or len(batch) < 1500:
                                break
                            cursor_ms = int(batch[-1][6]) + 1
                        if tail_rows:
                            tail_df = self._rows_to_klines_df(tail_rows, now_ms)
                            if len(tail_df) > 0:
                                merged = (
                                    pd.concat([df, tail_df], ignore_index=True)
                                    .drop_duplicates(subset=["open_time_ms"])
                                    .sort_values("open_time_ms")
                                    .reset_index(drop=True)
                                )
                                min_keep_ms = now_ms - int(days * 86400 * 1000)
                                merged = merged[merged["open_time_ms"] >= min_keep_ms].reset_index(drop=True)
                                merged["cvd"] = merged["volume_delta"].cumsum()
                                merged.to_csv(cache_path, index=False)
                                return merged
                    except Exception:
                        pass
            return df

        start_ms = now_ms - int(days * 86400 * 1000)
        all_rows = []
        cursor_ms = start_ms
        limit = 1500

        while cursor_ms < now_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor_ms,
                "endTime": now_ms,
                "limit": limit,
            }
            resp = self.session.get(
                f"{BINANCE_FAPI_BASE}/fapi/v1/klines", params=params, timeout=15
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_rows.extend(batch)
            last_open_ms = int(batch[-1][0])
            if last_open_ms <= cursor_ms:
                break
            # Advance by 1 bar
            cursor_ms = int(batch[-1][6]) + 1
            if len(batch) < limit:
                break
            time.sleep(0.08)

        if not all_rows:
            raise RuntimeError(f"No klines returned from Binance for {symbol}")

        df = self._rows_to_klines_df(all_rows, now_ms)
        df.to_csv(cache_path, index=False)
        return df

    def fetch_funding_rates(
        self, symbol: str, days: int = 180, use_cache: bool = True
    ) -> pd.DataFrame:
        cache_path = self._funding_cache_path(symbol, days)
        if use_cache and os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df["funding_time"] = pd.to_datetime(df["funding_time"], format="ISO8601", utc=True)
            return df.sort_values("funding_time").reset_index(drop=True)

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - int(days * 86400 * 1000)
        all_rows = []
        cursor_ms = start_ms

        while cursor_ms < now_ms:
            params = {
                "symbol": symbol,
                "startTime": cursor_ms,
                "endTime": now_ms,
                "limit": 1000,
            }
            resp = self.session.get(
                f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate", params=params, timeout=15
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_rows.extend(batch)
            last_ms = int(batch[-1]["fundingTime"])
            if last_ms <= cursor_ms or len(batch) < 1000:
                break
            cursor_ms = last_ms + 1
            time.sleep(0.08)

        if not all_rows:
            return pd.DataFrame(columns=["symbol", "funding_time", "funding_rate"])

        df = pd.DataFrame(all_rows)
        df["funding_time"] = pd.to_datetime(df["fundingTime"].astype(np.int64), unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = (
            df[["symbol", "funding_time", "funding_rate"]]
            .drop_duplicates(subset=["funding_time"])
            .sort_values("funding_time")
            .reset_index(drop=True)
        )
        df.to_csv(cache_path, index=False)
        return df
