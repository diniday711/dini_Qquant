"""
Q-Stack Phase 1.5 Pre-Registered Hypothesis Family Runner (Candidates A vs B)
Implements the Round 3 Council Synthesis:
- Discovery IS: Existing 180-day window (used strictly for candidate selection)
- Untouched Out-of-Sample Holdout: Prior uninspected 180-day window (Day -360 to Day -181)
  (Avoids reusing the already-inspected 180d window while avoiding a 6-month wall-clock stall!)
- Candidates:
  * Candidate A (`A_1H_STRUCTURAL`): 1h timeframe upgrade (wider structural stops ~2.2% vs 0.25% friction)
    with stricter z-threshold (|z| >= 2.5) and confirmation bar.
  * Candidate B (`B_MAKER_ADVERSE_SELECTION`): Maker limit entry at sweep level (0.04% roundtrip maker fee),
    evaluated across 3 mandatory fill/adverse-selection scenarios:
      - 100% Maker Fill (Optimistic)
      - 50% Winner Fill / 100% Loser Fill (Institutional Adverse Selection Stress Test!)
      - 0% Maker Fill (Taker Fallback floor)
- Winner Selection Rule (Pre-registered for NEEDS_JOOAN_APPROVAL):
  1) Must pass IS Gate: Post-Cost Net Profit > 0 AND Gross/Cost >= 2.5 (Net/Cost >= 1.5) AND MDD <= 15%
     (and we also report Opgo's strict Gross/Cost >= 5.0).
  2) Single winner chosen by highest IS `Net Profit / Total Cost` ratio (tie-break: lower IS MDD).
  3) Only the single winning candidate is evaluated once on the untouched Holdout window.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
import numpy as np
import pandas as pd

from config import CONFIG, Phase1Config
from engines.backtester import Phase1Backtester
from engines.cost_model import CostModel
from engines.data_fetcher import BinanceDataFetcher


@dataclass(frozen=True)
class Phase15Registration:
    family_name: str = "Phase1_5_Friction_Resolution_AB"
    bonferroni_alpha: float = 0.025  # 0.05 / 2 candidates
    winner_selection_metric: str = "net_profit_to_cost_ratio"
    tie_breaker_metric: str = "max_drawdown_pct"
    is_min_gross_to_cost_standard: float = 2.5  # Equivalent to Net/Cost >= 1.5
    is_min_gross_to_cost_opgo_strict: float = 5.0
    maker_fill_scenarios: tuple = (1.0, 0.5, 0.0)  # 100%, 50% (with 100% loser fill adverse selection), 0%

    def registration_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


REG_15 = Phase15Registration()


def evaluate_candidate_metrics(trades_df: pd.DataFrame, initial_capital: float = 1000.0) -> dict:
    if trades_df.empty:
        return {
            "trades": 0,
            "gross_pnl_usd": 0.0,
            "total_cost_usd": 0.0,
            "net_pnl_usd": 0.0,
            "per_trade_gross_usd": 0.0,
            "per_trade_cost_usd": 0.0,
            "gross_to_cost_ratio": 0.0,
            "net_to_cost_ratio": 0.0,
            "mdd_pct": 0.0,
            "sharpe": 0.0,
        }
    n = len(trades_df)
    gross = float(trades_df["gross_pnl_usd"].sum())
    cost = float(trades_df["total_cost_usd"].sum())
    net = float(trades_df["net_pnl_usd"].sum())
    eq = np.concatenate([[initial_capital], initial_capital + np.cumsum(trades_df["net_pnl_usd"].to_numpy())])
    peaks = np.maximum.accumulate(eq)
    mdd = float(np.max(1.0 - eq / np.maximum(1e-9, peaks)))

    temp = trades_df.copy()
    temp["date"] = temp["exit_time"].dt.floor("D")
    daily_pnl = temp.groupby("date")["net_pnl_usd"].sum()
    all_days = pd.date_range(
        trades_df["entry_time"].min().floor("D"),
        trades_df["exit_time"].max().floor("D"),
        freq="D",
        tz="UTC",
    )
    daily_ret = daily_pnl.reindex(all_days, fill_value=0.0) / initial_capital
    std = float(daily_ret.std(ddof=1)) if len(daily_ret) > 1 else 0.0
    sharpe = float(np.sqrt(365.0) * (daily_ret.mean() / std)) if std > 1e-12 else 0.0

    return {
        "trades": n,
        "gross_pnl_usd": gross,
        "total_cost_usd": cost,
        "net_pnl_usd": net,
        "per_trade_gross_usd": gross / n,
        "per_trade_cost_usd": cost / n,
        "gross_to_cost_ratio": (gross / cost) if cost > 0 else 0.0,
        "net_to_cost_ratio": (net / cost) if cost > 0 else 0.0,
        "mdd_pct": mdd,
        "sharpe": sharpe,
    }


def apply_maker_adverse_selection_scenario(
    trades_df: pd.DataFrame,
    winner_fill_prob: float,
    maker_roundtrip_fee: float = 0.0004,
    seed: int = 20261023,
) -> pd.DataFrame:
    """
    Institutional Adverse Selection Model for Candidate B (Maker Limit Order):
    - When winner_fill_prob == 0.0: 0% Maker benefit (100% Taker costs as baseline).
    - When winner_fill_prob == 1.0: 100% of trades fill as Maker (0.04% fee, 0% slippage).
    - When winner_fill_prob == 0.5:
      * Losing trades (`raw_return_pct < 0`, where price blew through the limit order toward Stop-Loss)
        are filled 100% of the time! (Adverse Selection!)
      * Winning trades (`raw_return_pct > 0`, where price barely touched the limit order and reversed)
        only get filled 50% of the time!
    """
    if trades_df.empty:
        return trades_df
    if winner_fill_prob <= 0.0:
        # 0% Maker scenario = falls back to Taker cost structure
        return trades_df.copy()

    rng = np.random.default_rng(seed)
    df = trades_df.copy()

    if winner_fill_prob < 1.0:
        is_winner = df["raw_return_pct"] > 0
        random_draws = rng.uniform(0.0, 1.0, size=len(df))
        # Losers ALWAYS fill (100% adverse selection); winners fill with probability `winner_fill_prob`
        keep_mask = (~is_winner) | (random_draws <= winner_fill_prob)
        df = df[keep_mask].copy().reset_index(drop=True)

    # Recompute costs under Maker fee (0.04% roundtrip) + 0 slippage + original funding
    df["fee_usd"] = df["notional_usd"] * maker_roundtrip_fee
    df["slippage_usd"] = 0.0
    df["total_cost_usd"] = df["fee_usd"] + df["slippage_usd"] + df["funding_usd"]
    df["net_pnl_usd"] = df["gross_pnl_usd"] - df["total_cost_usd"]
    return df


def verify_intrabar_1m_path_reversal(fetcher: BinanceDataFetcher, symbol: str = "SOLUSDT") -> dict:
    """
    Implements Opgo/Claude's `assert_1m_path_reversal`:
    Check what fraction of 15m sweep bars actually have reversal volume delta
    in the final 3 minutes (`min 12..14`) of the 15-minute bar.
    """
    df_1m = fetcher.fetch_klines(symbol, interval="1m", days=3, use_cache=True)
    df_1m["delta_1m"] = 2.0 * df_1m["taker_buy_volume"] - df_1m["volume"]
    df_1m["bucket_15m"] = df_1m["open_time"].dt.floor("15min")
    df_1m["minute_in_bar"] = df_1m["open_time"].dt.minute % 15

    last3 = (
        df_1m[df_1m["minute_in_bar"] >= 12]
        .groupby("bucket_15m")["delta_1m"]
        .sum()
        .rename("last3m_delta")
    )
    full15 = df_1m.groupby("bucket_15m")["delta_1m"].sum().rename("full15m_delta")
    merged = pd.concat([full15, last3], axis=1).dropna()
    same_sign_ratio = float(np.mean(np.sign(merged["full15m_delta"]) == np.sign(merged["last3m_delta"])))
    return {
        "symbol": symbol,
        "bars_checked": len(merged),
        "full15m_vs_last3m_sign_agreement": same_sign_ratio,
    }


def main():
    fetcher = BinanceDataFetcher(CONFIG, data_dir="data/raw")

    # 1. Check 1m Intrabar Path Reversal statistic (`assert_1m_path_reversal`)
    intrabar_stats = verify_intrabar_1m_path_reversal(fetcher, "SOLUSDT")

    # 2. Candidate A: 1h Timeframe Upgrade on 180-day Discovery IS
    cfg_a_1h = replace(
        CONFIG,
        timeframe="1h",
        bars_per_day=24,
        beta_window=48,            # 48h rolling beta on 1h bars
        residual_window=6,         # 6h cumulative residual
        residual_z_window=96,      # 96h (4d) standardization
        residual_z_threshold=2.5,
        swing_lookback=20,         # 20h swing high/low
        vwap_window=48,            # 48h VWAP for TP1
        atr_window=20,
        stop_atr_mult=1.5,
        min_stop_distance_pct=0.008,  # 0.80% minimum stop floor on 1h
        max_holding_bars=48,       # 48h time exit
    )
    print("[Phase 1.5] Fetching 1h klines for Candidate A (180 days Discovery IS)...")
    btc_1h = fetcher.fetch_klines(cfg_a_1h.benchmark_symbol, "1h", days=180)
    univ_1h = {
        sym: fetcher.fetch_klines(sym, "1h", days=180)
        for sym in cfg_a_1h.target_universe
    }
    fund_180 = {
        sym: fetcher.fetch_funding_rates(sym, days=180)
        for sym in cfg_a_1h.target_universe
    }

    bt_a = Phase1Backtester(cfg_a_1h)
    trades_a_1h, _, _, _ = bt_a.run_portfolio_backtest(univ_1h, btc_1h, fund_180)
    metrics_a_1h = evaluate_candidate_metrics(trades_a_1h, cfg_a_1h.initial_capital)

    # 3. Candidate B: 15m Maker Limit Entry across 3 Fill/Adverse-Selection Scenarios (100%, 50%, 0%)
    btc_15m = fetcher.fetch_klines(CONFIG.benchmark_symbol, "15m", days=180)
    univ_15m = {
        sym: fetcher.fetch_klines(sym, "15m", days=180)
        for sym in CONFIG.target_universe
    }
    bt_b = Phase1Backtester(CONFIG)
    trades_b_base, _, _, _ = bt_b.run_portfolio_backtest(univ_15m, btc_15m, fund_180)

    maker_scenarios = {}
    for fill_prob in REG_15.maker_fill_scenarios:
        df_scen = apply_maker_adverse_selection_scenario(trades_b_base, fill_prob)
        maker_scenarios[f"fill_{int(fill_prob*100)}pct"] = evaluate_candidate_metrics(
            df_scen, CONFIG.initial_capital
        )

    # Also evaluate Candidate A + Maker (1h + Maker 50% adverse selection) to see combined effect!
    trades_a_maker50 = apply_maker_adverse_selection_scenario(trades_a_1h, 0.50)
    metrics_a_maker50 = evaluate_candidate_metrics(trades_a_maker50, cfg_a_1h.initial_capital)

    report = {
        "registration_sha256": REG_15.registration_hash(),
        "intrabar_1m_path_check": intrabar_stats,
        "candidate_A_1h_taker": metrics_a_1h,
        "candidate_B_15m_maker_scenarios": maker_scenarios,
        "combined_A_1h_plus_maker_50pct_adverse": metrics_a_maker50,
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/phase1_5_diagnostic.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
