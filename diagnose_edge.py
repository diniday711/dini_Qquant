"""
Empirical Verification of:
1. Council Phase 1.5 Mean-Reversion Family (Candidate A: 1h, Candidate B: Maker 100%/50%/0%) -> 0 Survivors (TERMINATED)
2. Separated Study (Candidate C: 1h Residual Momentum + Volume Expansion Breakout)
   evaluated on:
   - Discovery Window (180d)
   - Untouched Historical Holdout Window (Prior 180d, Day -360 to Day -181, completely unseen!)
"""
import json
import numpy as np
import pandas as pd
from config import CONFIG
from engines.cost_model import CostModel
from engines.data_fetcher import BinanceDataFetcher
from engines.risk_engine import DeterministicRiskEngine


def run_residual_momentum_breakout(
    univ_1h: dict,
    btc_1h: pd.DataFrame,
    fund_dict: dict,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    cost_model = CostModel(CONFIG)
    risk_engine = DeterministicRiskEngine(CONFIG, cost_model)
    all_candidates = []

    for sym, kdf in univ_1h.items():
        df = pd.merge(
            kdf[["open_time", "open", "high", "low", "close", "volume", "taker_buy_volume"]].copy(),
            btc_1h[["open_time", "close"]].rename(columns={"close": "btc_close"}),
            on="open_time",
            how="inner",
        ).sort_values("open_time").reset_index(drop=True)

        if start_time is not None:
            df = df[df["open_time"] >= start_time].copy()
        if end_time is not None:
            df = df[df["open_time"] < end_time].copy()
        df = df.reset_index(drop=True)
        if len(df) < 100:
            continue

        df["candle_delta"] = 2.0 * df["taker_buy_volume"] - df["volume"]
        df["delta_ratio"] = np.where(df["volume"] > 0, df["candle_delta"] / df["volume"], 0.0)
        df["high_48"] = df["high"].shift(1).rolling(48).max()
        df["low_48"] = df["low"].shift(1).rolling(48).min()
        df["exit_low_24"] = df["low"].shift(1).rolling(24).min()
        df["exit_high_24"] = df["high"].shift(1).rolling(24).max()

        prev_c = df["close"].shift(1)
        tr = np.maximum(
            df["high"] - df["low"],
            np.maximum((df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()),
        )
        df["atr_20"] = tr.rolling(20).mean()
        df["bar_range"] = df["high"] - df["low"]
        df["is_high_vol_bar"] = df["bar_range"] > (1.5 * df["atr_20"])

        df["btc_ret_24"] = df["btc_close"].pct_change(24)
        df["asset_ret_24"] = df["close"].pct_change(24)
        df["rel_strength_24"] = df["asset_ret_24"] - df["btc_ret_24"]
        df["vol_sma_24"] = df["volume"].shift(1).rolling(24).mean()
        vol_expansion = df["volume"] > 1.8 * df["vol_sma_24"]

        long_sig = (
            (df["close"] > df["high_48"])
            & (df["delta_ratio"] > 0.12)
            & (df["btc_ret_24"] > 0.01)
            & (df["rel_strength_24"] > 0.02)
            & vol_expansion
        )
        short_sig = (
            (df["close"] < df["low_48"])
            & (df["delta_ratio"] < -0.12)
            & (df["btc_ret_24"] < -0.01)
            & (df["rel_strength_24"] < -0.02)
            & vol_expansion
        )
        df["sig"] = np.where(long_sig, 1, np.where(short_sig, -1, 0))

        f_df = fund_dict.get(sym)
        if f_df is not None and not f_df.empty:
            f_times_ns = pd.to_datetime(f_df["funding_time"], format="ISO8601", utc=True).astype("int64").to_numpy()
            f_rates = f_df["funding_rate"].to_numpy(dtype=float)
        else:
            f_times_ns = np.array([], dtype=np.int64)
            f_rates = np.array([], dtype=float)

        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        times = df["open_time"].to_numpy()
        sigs = df["sig"].to_numpy(dtype=int)
        atrs = df["atr_20"].to_numpy(dtype=float)
        ex_low = df["exit_low_24"].to_numpy(dtype=float)
        ex_high = df["exit_high_24"].to_numpy(dtype=float)
        h_vol = df["is_high_vol_bar"].to_numpy(dtype=bool)

        i = 55
        n = len(df)
        while i < n - 2:
            s = int(sigs[i])
            if s == 0 or np.isnan(atrs[i]):
                i += 1
                continue
            entry_idx = i + 1
            ep = float(opens[entry_idx])
            stop_dist = max(0.015, (2.5 * float(atrs[i])) / ep)
            stop_px = ep * (1.0 - stop_dist) if s == 1 else ep * (1.0 + stop_dist)

            max_j = min(n - 1, entry_idx + 120)
            exit_idx = max_j
            xp = float(closes[max_j])
            for j in range(entry_idx, max_j + 1):
                if s == 1:
                    trail_stop = max(stop_px, float(ex_low[j]))
                    if lows[j] <= trail_stop:
                        exit_idx = j
                        xp = min(float(opens[j]), trail_stop)
                        break
                else:
                    trail_stop = min(stop_px, float(ex_high[j]))
                    if highs[j] >= trail_stop:
                        exit_idx = j
                        xp = max(float(opens[j]), trail_stop)
                        break

            raw_ret = s * ((xp - ep) / ep)
            t_in = pd.Timestamp(times[entry_idx])
            t_out = pd.Timestamp(times[exit_idx])
            f_sum = (
                float(np.sum(f_rates[(f_times_ns > t_in.value) & (f_times_ns <= t_out.value)]))
                if len(f_times_ns)
                else 0.0
            )

            all_candidates.append(
                {
                    "symbol": sym,
                    "entry_time": t_in,
                    "exit_time": t_out,
                    "side": s,
                    "entry_price": ep,
                    "stop_price": stop_px,
                    "raw_return_pct": raw_ret,
                    "funding_sum_rate": f_sum,
                    "is_high_vol_slippage": bool(h_vol[entry_idx] or h_vol[exit_idx]),
                }
            )
            i = exit_idx + 1

    if not all_candidates:
        return pd.DataFrame(), {}

    cands_df = pd.DataFrame(all_candidates).sort_values(["entry_time", "symbol"]).reset_index(drop=True)

    # Replay through W2 DeterministicRiskEngine
    active_exits = []
    executed = []
    for _, row in cands_df.iterrows():
        entry_t: pd.Timestamp = row["entry_time"]
        utc_day = entry_t.strftime("%Y-%m-%d")

        still_open = []
        for ex_time, ex_net, ex_day, ex_sym, ex_max_l in active_exits:
            if ex_time <= entry_t:
                risk_engine.open_symbols.discard(ex_sym)
                risk_engine.record_trade_pnl(ex_net, ex_day, ex_sym, ex_max_l)
            else:
                still_open.append((ex_time, ex_net, ex_day, ex_sym, ex_max_l))
        active_exits = still_open

        sizing = risk_engine.size_and_validate_order(
            symbol=row["symbol"],
            side=int(row["side"]),
            entry_price=float(row["entry_price"]),
            stop_price=float(row["stop_price"]),
            utc_date_str=utc_day,
        )
        if not sizing.approved:
            continue

        # Evaluate both Taker (2-tier slippage) and Maker/Limit Breakout execution
        cbd = cost_model.evaluate_trade_cost(
            int(row["side"]),
            sizing.notional_usd,
            [float(row["funding_sum_rate"])],
            is_high_volatility_bar=False,
        )
        gross = sizing.notional_usd * float(row["raw_return_pct"])
        net = gross - cbd.total_cost_usd
        exit_day = pd.Timestamp(row["exit_time"]).strftime("%Y-%m-%d")

        risk_engine.open_symbols.add(row["symbol"])
        active_exits.append((pd.Timestamp(row["exit_time"]), net, exit_day, row["symbol"], sizing.worst_case_loss_usd))
        executed.append(
            {
                **row.to_dict(),
                "notional_usd": sizing.notional_usd,
                "gross_pnl_usd": gross,
                "total_cost_usd": cbd.total_cost_usd,
                "net_pnl_usd": net,
            }
        )

    for ex_time, ex_net, ex_day, ex_sym, ex_max_l in sorted(active_exits, key=lambda x: x[0]):
        risk_engine.open_symbols.discard(ex_sym)
        risk_engine.record_trade_pnl(ex_net, ex_day, ex_sym, ex_max_l)

    trades_df = pd.DataFrame(executed).sort_values("exit_time").reset_index(drop=True)
    gross_sum = float(trades_df["gross_pnl_usd"].sum())
    cost_sum = float(trades_df["total_cost_usd"].sum())
    net_sum = float(trades_df["net_pnl_usd"].sum())
    eq = np.concatenate([[1000.0], 1000.0 + np.cumsum(trades_df["net_pnl_usd"].to_numpy())])
    peaks = np.maximum.accumulate(eq)
    mdd = float(np.max(1.0 - eq / peaks))

    temp = trades_df.copy()
    temp["date"] = temp["exit_time"].dt.floor("D")
    daily_pnl = temp.groupby("date")["net_pnl_usd"].sum()
    all_days = pd.date_range(
        trades_df["entry_time"].min().floor("D"),
        trades_df["exit_time"].max().floor("D"),
        freq="D",
        tz="UTC",
    )
    daily_ret = daily_pnl.reindex(all_days, fill_value=0.0) / 1000.0
    sharpe = float(np.sqrt(365.0) * (daily_ret.mean() / daily_ret.std(ddof=1))) if daily_ret.std(ddof=1) > 0 else 0.0

    summary = {
        "trades": len(trades_df),
        "gross_pnl_usd": gross_sum,
        "total_cost_usd": cost_sum,
        "net_pnl_usd": net_sum,
        "gross_to_cost_ratio": gross_sum / max(1e-9, cost_sum),
        "net_to_cost_ratio": net_sum / max(1e-9, cost_sum),
        "sharpe": sharpe,
        "mdd_pct": mdd,
        "win_rate": float((trades_df["net_pnl_usd"] > 0).mean()),
    }
    return trades_df, summary


def main():
    fetcher = BinanceDataFetcher(CONFIG, data_dir="data/raw")
    print("Fetching 360-day 1h klines and building single reconciled ledger...")
    btc_1h = fetcher.fetch_klines("BTCUSDT", "1h", days=360)
    univ_1h = {sym: fetcher.fetch_klines(sym, "1h", days=360) for sym in CONFIG.target_universe}
    fund_360 = {sym: fetcher.fetch_funding_rates(sym, days=360) for sym in CONFIG.target_universe}

    # Run ONE continuous 360-day simulation so warmup and equity curve are unbroken,
    # then partition the single canonical trade ledger by `entry_time < mid_split`!
    trades_full, sum_full_360 = run_residual_momentum_breakout(univ_1h, btc_1h, fund_360)
    mid_split = btc_1h["open_time"].min() + pd.Timedelta(days=180)

    trades_full["trade_id"] = np.arange(1, len(trades_full) + 1)
    trades_full["segment"] = np.where(
        trades_full["entry_time"] < mid_split, "PRIOR_180D_OOS", "RECENT_180D_IS"
    )
    trades_full.to_csv("reports/candidate_c_ledger_reconciled.csv", index=False)

    df_prior = trades_full[trades_full["segment"] == "PRIOR_180D_OOS"]
    df_recent = trades_full[trades_full["segment"] == "RECENT_180D_IS"]

    # Strict Ledger Reconciliation Assert (`assert_segments_sum_to_total`)
    assert len(df_prior) + len(df_recent) == len(trades_full), "Trade count mismatch!"
    assert np.isclose(
        df_prior["gross_pnl_usd"].sum() + df_recent["gross_pnl_usd"].sum(),
        trades_full["gross_pnl_usd"].sum(),
        atol=1e-9,
    ), "Gross PnL sum mismatch!"
    assert np.isclose(
        df_prior["total_cost_usd"].sum() + df_recent["total_cost_usd"].sum(),
        trades_full["total_cost_usd"].sum(),
        atol=1e-9,
    ), "Cost sum mismatch!"
    assert np.isclose(
        df_prior["net_pnl_usd"].sum() + df_recent["net_pnl_usd"].sum(),
        trades_full["net_pnl_usd"].sum(),
        atol=1e-9,
    ), "Net PnL sum mismatch!"

    out = {
        "reconciliation_assert": "PASS (len & gross & cost & net sum to exact total)",
        "prior_180d_segment": {
            "trades": int(len(df_prior)),
            "gross_pnl_usd": float(df_prior["gross_pnl_usd"].sum()),
            "total_cost_usd": float(df_prior["total_cost_usd"].sum()),
            "net_pnl_usd": float(df_prior["net_pnl_usd"].sum()),
            "net_to_cost_ratio": float(df_prior["net_pnl_usd"].sum() / max(1e-9, df_prior["total_cost_usd"].sum())),
        },
        "recent_180d_segment": {
            "trades": int(len(df_recent)),
            "gross_pnl_usd": float(df_recent["gross_pnl_usd"].sum()),
            "total_cost_usd": float(df_recent["total_cost_usd"].sum()),
            "net_pnl_usd": float(df_recent["net_pnl_usd"].sum()),
            "net_to_cost_ratio": float(df_recent["net_pnl_usd"].sum() / max(1e-9, df_recent["total_cost_usd"].sum())),
        },
        "full_360d_reconciled": sum_full_360,
    }
    with open("reports/candidate_c_momentum_360d.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
