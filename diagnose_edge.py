"""
Diagnostic script (does NOT modify frozen config.py) to test why Opgo's 15m spec
had positive Gross PnL (+$19.64) and significant entry timing (p=0.0139)
but lost -$507.06 to fees/slippage, and what structural adjustment solves the friction trap.
"""
from dataclasses import replace
import numpy as np
import pandas as pd
from config import CONFIG
from engines.backtester import Phase1Backtester
from engines.data_fetcher import BinanceDataFetcher


def main():
    fetcher = BinanceDataFetcher(CONFIG, data_dir="data/raw")
    btc_klines = fetcher.fetch_klines(CONFIG.benchmark_symbol, CONFIG.timeframe, days=180)
    universe_klines = {
        sym: fetcher.fetch_klines(sym, CONFIG.timeframe, days=180)
        for sym in CONFIG.target_universe
    }
    universe_funding = {
        sym: fetcher.fetch_funding_rates(sym, days=180)
        for sym in CONFIG.target_universe
    }

    variants = [
        ("0_Opgo_Baseline_15m", CONFIG),
        (
            "1_Wider_Stop_And_Target_Filter",
            replace(
                CONFIG,
                stop_atr_mult=3.0,
                min_stop_distance_pct=0.012,  # 1.2% minimum structural stop so friction is <15% of stop
                min_reward_to_cost_mult=8.0,  # Target must be >= 8x roundtrip cost
                tp2_residual_z_target=0.0,    # Hold TP2 until full mean reversion (z=0)
            ),
        ),
        (
            "2_Maker_Limit_Execution_At_Sweep",
            replace(
                CONFIG,
                roundtrip_fee_rate=0.0004,           # Binance USDT-M Maker fee (0.02% x 2 = 0.04%)
                normal_roundtrip_slippage_rate=0.0,  # Limit order fills at limit price or better
                stressed_roundtrip_slippage_rate=0.0002,
                stop_atr_mult=2.5,
                min_stop_distance_pct=0.010,
                min_reward_to_cost_mult=6.0,
            ),
        ),
    ]

    # Also test Momentum Continuation (trading WITH the residual breakout instead of fading it!)
    bt_base = Phase1Backtester(CONFIG)
    trades_df, feat_store, sym_arrays, _ = bt_base.run_portfolio_backtest(
        universe_klines, btc_klines, universe_funding
    )
    print(
        f"[0_Opgo_MeanReversion_15m] Trades={len(trades_df)} | Gross=${trades_df['gross_pnl_usd'].sum():,.2f} | Cost=${trades_df['total_cost_usd'].sum():,.2f} | Net=${trades_df['net_pnl_usd'].sum():,.2f}"
    )
    for reason, grp in trades_df.groupby("exit_reason"):
        print(
            f"   -> {reason}: n={len(grp)}, Gross=${grp['gross_pnl_usd'].sum():,.2f}, Net=${grp['net_pnl_usd'].sum():,.2f}, AvgRawRet={grp['raw_return_pct'].mean():.4%}"
        )


if __name__ == "__main__":
    main()
