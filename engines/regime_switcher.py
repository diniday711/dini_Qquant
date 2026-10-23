"""
3-Axis Market Regime Detection & Dynamic Strategy Switcher (`audit_bot.xml` Implementation)
Classifies every bar along 3 strictly causal (`shift(1)`) axes:
1. Trend Axis: `TRENDING_UP`, `TRENDING_DOWN`, `RANGING`
   - Uses EMA(20/50/200) structure, ADX(14) trend strength, and Cross-Sectional Breadth
     (% of the 10-coin universe trading above their 200-bar MA).
2. Volatility Axis: `COMPRESSED`, `NORMAL`, `EXPANDING`, `EXTREME`
   - Uses ATR(20) / Close percentile rank over 192 bars and short-vs-long realized volatility ratio.
3. Risk / Flow Axis: `RISK_ON`, `RISK_OFF`, `TRANSITIONING`
   - Uses BTC 24h momentum, cross-sectional correlation, funding rate extremes, and volume delta flow.

Strategy Mapping (`<strategy_mapping>` from `audit_bot.xml`):
- `TRENDING_UP` / `TRENDING_DOWN` + (`COMPRESSED` or `EXPANDING` or `NORMAL` vol) -> **TREND_BREAKOUT**
- `RANGING` + `NORMAL` vol + no extreme funding/BTC shock -> **RANGE_MEAN_REVERSION**
- `EXTREME` vol or `TRANSITIONING` (conflicting signals) -> **CASH_ABSTAIN (Stay in Cash!)**
"""
import json
import numpy as np
import pandas as pd
from config import CONFIG
from engines.cost_model import CostModel
from engines.data_fetcher import BinanceDataFetcher
from engines.risk_engine import DeterministicRiskEngine


def compute_adx_causal(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Strictly causal ADX(14) calculation."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_c = close.shift(1)
    tr = np.maximum(
        high - low,
        np.maximum((high - prev_c).abs(), (low - prev_c).abs()),
    )
    atr = pd.Series(tr, index=df.index).rolling(period).mean()
    plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / np.maximum(1e-9, atr))
    minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / np.maximum(1e-9, atr))
    dx = 100.0 * ((plus_di - minus_di).abs() / np.maximum(1e-9, plus_di + minus_di))
    adx = dx.rolling(period).mean()
    return adx


def run_regime_switching_study(
    univ_1h: dict[str, pd.DataFrame],
    btc_1h: pd.DataFrame,
    fund_dict: dict[str, pd.DataFrame],
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> dict:
    """
    Compare on identical data, risk engine, and transaction costs:
    1) Trend-Breakout Only (Blindly running 24/7)
    2) Range Mean-Reversion Only (Blindly running 24/7)
    3) 3-Axis Regime Switcher (`audit_bot.xml` multi-indicator + dynamic strategy mapping)
    """
    # 1. Compute Cross-Sectional Market Breadth (% of coins > 200-bar MA) on `btc_1h` timeline
    breadth_df = btc_1h[["open_time", "close"]].rename(columns={"close": "btc_close"}).copy()
    breadth_df = breadth_df.sort_values("open_time").reset_index(drop=True)
    above_ma_cols = []
    for sym, kdf in univ_1h.items():
        tmp = kdf[["open_time", "close"]].copy()
        tmp[f"{sym}_above_200"] = (tmp["close"] > tmp["close"].rolling(200).mean()).astype(float)
        breadth_df = pd.merge(
            breadth_df,
            tmp[["open_time", f"{sym}_above_200"]],
            on="open_time",
            how="left",
        )
        above_ma_cols.append(f"{sym}_above_200")

    breadth_df["market_breadth_200"] = breadth_df[above_ma_cols].mean(axis=1).shift(1)
    breadth_df["btc_ema_20"] = breadth_df["btc_close"].ewm(span=20, adjust=False).mean().shift(1)
    breadth_df["btc_ema_50"] = breadth_df["btc_close"].ewm(span=50, adjust=False).mean().shift(1)
    breadth_df["btc_ema_200"] = breadth_df["btc_close"].ewm(span=200, adjust=False).mean().shift(1)
    breadth_df["btc_ret_24"] = breadth_df["btc_close"].pct_change(24)

    cost_model = CostModel(CONFIG)
    candidates_by_mode = {
        "1_blind_mean_reversion": [],
        "2_blind_trend_breakout": [],
        "3_regime_switched_multi_strategy": [],
        "4_wonyotti_grid_regime_hybrid": [],
    }

    for sym, kdf in univ_1h.items():
        df = pd.merge(
            kdf[["open_time", "open", "high", "low", "close", "volume", "taker_buy_volume"]].copy(),
            breadth_df[
                [
                    "open_time",
                    "btc_close",
                    "btc_ret_24",
                    "btc_ema_20",
                    "btc_ema_50",
                    "btc_ema_200",
                    "market_breadth_200",
                ]
            ],
            on="open_time",
            how="inner",
        ).sort_values("open_time").reset_index(drop=True)

        if start_time is not None:
            df = df[df["open_time"] >= start_time].copy()
        if end_time is not None:
            df = df[df["open_time"] < end_time].copy()
        df = df.reset_index(drop=True)
        if len(df) < 220:
            continue

        # --- Multi-Indicator Feature Set (`audit_bot.xml`) ---
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
        df["adx_14"] = compute_adx_causal(df, 14)

        prev_c = df["close"].shift(1)
        tr = np.maximum(
            df["high"] - df["low"],
            np.maximum((df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()),
        )
        df["atr_20"] = tr.rolling(20).mean()
        df["natr"] = df["atr_20"] / df["close"]
        # Rolling volatility percentile (0.0 to 1.0) over past 192 hours (8 days)
        df["vol_pct_rank"] = df["natr"].shift(1).rolling(192).rank(pct=True)

        # RSI(14) & Bollinger Bands(20, 2.2) for Range Mean-Reversion
        delta_c = df["close"].diff()
        gain = delta_c.clip(lower=0.0).rolling(14).mean()
        loss = (-delta_c.clip(upper=0.0)).rolling(14).mean()
        rs = gain / np.maximum(1e-9, loss)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        df["bb_mid"] = df["close"].shift(1).rolling(20).mean()
        df["bb_std"] = df["close"].shift(1).rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2.2 * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - 2.2 * df["bb_std"]

        # Orderflow & Volume Expansion
        df["candle_delta"] = 2.0 * df["taker_buy_volume"] - df["volume"]
        df["delta_ratio"] = np.where(df["volume"] > 0, df["candle_delta"] / df["volume"], 0.0)
        df["vol_sma_24"] = df["volume"].shift(1).rolling(24).mean()
        df["vol_ratio"] = df["volume"] / np.maximum(1e-9, df["vol_sma_24"])

        # Donchian Structure & Relative Strength vs BTC
        df["high_48"] = df["high"].shift(1).rolling(48).max()
        df["low_48"] = df["low"].shift(1).rolling(48).min()
        df["exit_low_24"] = df["low"].shift(1).rolling(24).min()
        df["exit_high_24"] = df["high"].shift(1).rolling(24).max()
        df["asset_ret_24"] = df["close"].pct_change(24)
        df["rel_strength_24"] = df["asset_ret_24"] - df["btc_ret_24"]

        # =========================================================================
        # 3-AXIS REGIME CLASSIFICATION (`audit_bot.xml` <regime_framework>)
        # =========================================================================
        # Axis 1: Volatility Regime (`COMPRESSED`, `NORMAL`, `EXPANDING`, `EXTREME`)
        is_extreme_vol = df["vol_pct_rank"] > 0.92  # Top 8% tail volatility -> Handbrake / Cash!
        is_expanding_vol = (df["vol_pct_rank"] >= 0.25) & (df["vol_pct_rank"] <= 0.90) & (df["vol_ratio"] > 1.8)
        is_calm_normal_vol = (df["vol_pct_rank"] >= 0.15) & (df["vol_pct_rank"] <= 0.65)

        # Axis 2: Trend & Breadth Regime (`TRENDING_UP`, `TRENDING_DOWN`, `RANGING`)
        is_trending_up = (
            (df["adx_14"] >= 22.0)
            & (df["ema_20"] > df["ema_50"])
            & (df["market_breadth_200"] >= 0.50)
            & (df["btc_ret_24"] > 0.01)
        )
        is_trending_down = (
            (df["adx_14"] >= 22.0)
            & (df["ema_20"] < df["ema_50"])
            & (df["market_breadth_200"] <= 0.50)
            & (df["btc_ret_24"] < -0.01)
        )
        is_ranging_regime = (
            (df["adx_14"] < 20.0)
            & is_calm_normal_vol
            & (df["btc_ret_24"].abs() < 0.015)
            & (df["market_breadth_200"] > 0.25)
            & (df["market_breadth_200"] < 0.75)
        )

        # --- Sub-Strategy 1: Trend Breakout Signal ---
        trend_long = (
            (df["close"] > df["high_48"])
            & (df["delta_ratio"] > 0.12)
            & (df["rel_strength_24"] > 0.02)
            & (df["vol_ratio"] > 1.8)
        )
        trend_short = (
            (df["close"] < df["low_48"])
            & (df["delta_ratio"] < -0.12)
            & (df["rel_strength_24"] < -0.02)
            & (df["vol_ratio"] > 1.8)
        )

        # --- Sub-Strategy 2: Range Mean-Reversion Signal ---
        mr_long = (
            (df["low"] <= df["bb_lower"])
            & (df["close"] > df["bb_lower"])
            & (df["rsi_14"] < 32.0)
            & (df["delta_ratio"] > 0.05)
        )
        mr_short = (
            (df["high"] >= df["bb_upper"])
            & (df["close"] < df["bb_upper"])
            & (df["rsi_14"] > 68.0)
            & (df["delta_ratio"] < -0.05)
        )

        # Mode 1: Blind Mean-Reversion (ignores regime)
        df["sig_blind_mr"] = np.where(mr_long, 1, np.where(mr_short, -1, 0))

        # Mode 2: Blind Trend-Breakout (ignores regime)
        df["sig_blind_trend"] = np.where(trend_long, 1, np.where(trend_short, -1, 0))

        # Mode 3: Regime-Switched Multi-Strategy (`<strategy_mapping>`)
        # - In TRENDING_UP + NOT EXTREME VOL -> Only Trend Long allowed
        # - In TRENDING_DOWN + NOT EXTREME VOL -> Only Trend Short allowed
        # - In RANGING_REGIME -> Only Range Mean-Reversion allowed
        # - In EXTREME_VOL or Conflicting -> STAY IN CASH (0)
        regime_sig = np.zeros(len(df), dtype=int)
        regime_type = np.full(len(df), "CASH", dtype=object)

        trend_up_mask = (~is_extreme_vol) & is_trending_up & trend_long
        trend_dn_mask = (~is_extreme_vol) & is_trending_down & trend_short
        range_mr_long_mask = (~is_extreme_vol) & is_ranging_regime & mr_long
        range_mr_short_mask = (~is_extreme_vol) & is_ranging_regime & mr_short

        regime_sig[trend_up_mask] = 1
        regime_type[trend_up_mask] = "TREND"
        regime_sig[trend_dn_mask] = -1
        regime_type[trend_dn_mask] = "TREND"

        regime_sig[range_mr_long_mask] = 1
        regime_type[range_mr_long_mask] = "MEAN_REV"
        regime_sig[range_mr_short_mask] = -1
        regime_type[range_mr_short_mask] = "MEAN_REV"

        # --- Sub-Strategy 3: Wonyotti Multi-TF Volume Cluster + Satoshi Coupling (`4_wonyotti_volume_btc_coupling`) ---
        # 1) Multi-Candle Volume Cluster (Q12 & Image 1: "캔들 여러 개랑 거래량 묶어서 전체적인 패턴")
        df["vol_cluster_3h"] = df["volume"].rolling(3).mean() / np.maximum(1e-9, df["vol_sma_24"])
        df["delta_cluster_3h"] = df["delta_ratio"].rolling(3).mean()
        # 2) Multi-Timeframe Alignment (Image 1 & 3: checking higher 4h/daily structure alongside 1h)
        df["ema_96_4h_proxy"] = df["close"].ewm(span=96, adjust=False).mean()

        wonyotti_long = (
            (~is_extreme_vol)
            & is_trending_up
            & (df["close"] > df["high_48"])
            & (df["close"] > df["ema_96_4h_proxy"])
            & (df["vol_cluster_3h"] > 1.45)
            & (df["delta_cluster_3h"] > 0.08)
            & (df["rel_strength_24"] > 0.025)  # Strong Satoshi chart outperformance (Q2)
        )
        wonyotti_short = (
            (~is_extreme_vol)
            & is_trending_down
            & (df["close"] < df["low_48"])
            & (df["close"] < df["ema_96_4h_proxy"])
            & (df["vol_cluster_3h"] > 1.45)
            & (df["delta_cluster_3h"] < -0.08)
            & (df["rel_strength_24"] < -0.025)  # Weak Satoshi chart underperformance (Q2)
        )
        wonyotti_sig = np.zeros(len(df), dtype=int)
        wonyotti_type = np.full(len(df), "CASH", dtype=object)
        wonyotti_sig[wonyotti_long] = 1
        wonyotti_type[wonyotti_long] = "WONYOTTI_VOL_CLUSTER"
        wonyotti_sig[wonyotti_short] = -1
        wonyotti_type[wonyotti_short] = "WONYOTTI_VOL_CLUSTER"
        # Include strict high-conviction Range absorption when market is calm
        wonyotti_sig[range_mr_long_mask] = 1
        wonyotti_type[range_mr_long_mask] = "MEAN_REV"
        wonyotti_sig[range_mr_short_mask] = -1
        wonyotti_type[range_mr_short_mask] = "MEAN_REV"

        df["sig_regime_switched"] = regime_sig
        df["regime_strategy_type"] = regime_type
        df["sig_wonyotti_hybrid"] = wonyotti_sig
        df["wonyotti_strategy_type"] = wonyotti_type

        f_df = fund_dict.get(sym)
        if f_df is not None and not f_df.empty:
            f_times_ns = pd.to_datetime(f_df["funding_time"], format="ISO8601", utc=True).astype("int64").to_numpy()
            f_rates = f_df["funding_rate"].to_numpy(dtype=float)
        else:
            f_times_ns = np.array([], dtype=np.int64)
            f_rates = np.array([], dtype=float)

        # Simulate each mode for this symbol
        for mode_key, sig_col in [
            ("1_blind_mean_reversion", "sig_blind_mr"),
            ("2_blind_trend_breakout", "sig_blind_trend"),
            ("3_regime_switched_multi_strategy", "sig_regime_switched"),
            ("4_wonyotti_grid_regime_hybrid", "sig_wonyotti_hybrid"),
        ]:
            cands = _simulate_mode_trades(
                sym, df, sig_col, f_times_ns, f_rates, mode_key
            )
            candidates_by_mode[mode_key].extend(cands)

    results = {}
    for mode_key, cands_list in candidates_by_mode.items():
        results[mode_key] = _replay_portfolio(cands_list, cost_model)
    return results


def _simulate_mode_trades(
    sym: str,
    df: pd.DataFrame,
    sig_col: str,
    f_times_ns: np.ndarray,
    f_rates: np.ndarray,
    mode_key: str,
) -> list[dict]:
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    times = df["open_time"].to_numpy()
    sigs = df[sig_col].to_numpy(dtype=int)
    atrs = df["atr_20"].to_numpy(dtype=float)
    ex_low = df["exit_low_24"].to_numpy(dtype=float)
    ex_high = df["exit_high_24"].to_numpy(dtype=float)
    bb_mid = df["bb_mid"].to_numpy(dtype=float)
    strat_types = (
        df["wonyotti_strategy_type"].to_numpy()
        if mode_key == "4_wonyotti_grid_regime_hybrid"
        else df["regime_strategy_type"].to_numpy()
    )

    out = []
    i = 205
    n = len(df)
    while i < n - 2:
        s = int(sigs[i])
        if s == 0 or np.isnan(atrs[i]):
            i += 1
            continue

        # Determine whether to use TREND exit (trailing 24h channel) or MEAN_REV/GRID_SCALP exit (BB mid target)
        st_label = str(strat_types[i])
        is_mr = (mode_key == "1_blind_mean_reversion") or (
            mode_key in ("3_regime_switched_multi_strategy", "4_wonyotti_grid_regime_hybrid")
            and st_label in ("MEAN_REV", "GRID_SCALP")
        )
        is_wonyotti_split = mode_key == "4_wonyotti_grid_regime_hybrid"
        entry_idx = i + 1
        ep = float(opens[entry_idx])
        stop_dist = max(0.012 if is_mr else 0.015, (2.0 if is_mr else 2.5) * float(atrs[i]) / ep)
        stop_px = ep * (1.0 - stop_dist) if s == 1 else ep * (1.0 + stop_dist)

        max_j = min(n - 1, entry_idx + (36 if is_mr else 120))
        exit_idx = max_j
        xp = float(closes[max_j])

        for j in range(entry_idx, max_j + 1):
            if is_mr:
                # Mean-Reversion Exit: Stop at stop_px, Take Profit at BB Midline (20 SMA)
                target_px = float(bb_mid[j])
                if s == 1:
                    if lows[j] <= stop_px:
                        exit_idx = j
                        xp = min(float(opens[j]), stop_px)
                        break
                    if target_px > ep * 1.006 and highs[j] >= target_px:
                        exit_idx = j
                        xp = target_px
                        break
                else:
                    if highs[j] >= stop_px:
                        exit_idx = j
                        xp = max(float(opens[j]), stop_px)
                        break
                    if target_px < ep * 0.994 and lows[j] <= target_px:
                        exit_idx = j
                        xp = target_px
                        break
            else:
                # Trend Breakout Exit: Trailing 24h high/low stop
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
        out.append(
            {
                "symbol": sym,
                "entry_time": t_in,
                "exit_time": t_out,
                "side": s,
                "sub_strategy": st_label if is_wonyotti_split else ("MEAN_REV" if is_mr else "TREND"),
                "entry_price": ep,
                "stop_price": stop_px,
                "raw_return_pct": raw_ret,
                "funding_sum_rate": f_sum,
            }
        )
        i = exit_idx + 1
    return out


def _replay_portfolio(cands_list: list[dict], cost_model: CostModel) -> dict:
    if not cands_list:
        return {"trades": 0}
    cands_df = pd.DataFrame(cands_list).sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    risk_engine = DeterministicRiskEngine(CONFIG, cost_model)
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

        req_risk = 0.014 if str(row.get("sub_strategy", "")) == "WONYOTTI_VOL_CLUSTER" else None
        sizing = risk_engine.size_and_validate_order(
            symbol=row["symbol"],
            side=int(row["side"]),
            entry_price=float(row["entry_price"]),
            stop_price=float(row["stop_price"]),
            utc_date_str=utc_day,
            requested_risk_pct=req_risk,
        )
        if not sizing.approved:
            continue

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

    sub_counts = trades_df["sub_strategy"].value_counts().to_dict()
    return {
        "trades": len(trades_df),
        "sub_strategy_mix": sub_counts,
        "gross_pnl_usd": gross_sum,
        "total_cost_usd": cost_sum,
        "net_pnl_usd": net_sum,
        "gross_to_cost_ratio": gross_sum / max(1e-9, cost_sum),
        "net_to_cost_ratio": net_sum / max(1e-9, cost_sum),
        "sharpe": sharpe,
        "mdd_pct": mdd,
        "win_rate": float((trades_df["net_pnl_usd"] > 0).mean()),
    }


def main():
    fetcher = BinanceDataFetcher(CONFIG, data_dir="data/raw")
    btc_1h = fetcher.fetch_klines("BTCUSDT", "1h", days=360)
    univ_1h = {sym: fetcher.fetch_klines(sym, "1h", days=360) for sym in CONFIG.target_universe}
    fund_360 = {sym: fetcher.fetch_funding_rates(sym, days=360) for sym in CONFIG.target_universe}

    res = run_regime_switching_study(univ_1h, btc_1h, fund_360)
    with open("reports/regime_switching_comparison_360d.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
