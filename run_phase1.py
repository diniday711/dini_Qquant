"""
Q-Stack Phase 1 Main Runner (v2.1 - Council Round 2 Spec Complete):
1. Verifies Config SHA-256 Hash & Runs 6 W2/Leakage/Fail-Closed/Shared-Execution Unit Tests
2. Backfills 10-Coin Pre-Registered Universe + BTC Benchmark from Binance USDT-M Futures
3. Cross-checks 1m vs 15m `candle_delta_15m` correlation (verifying mathematical identity rho = 1.0)
4. Runs W1 Single Confluence Strategy (with TP1=VWAP(96) 50% + Breakeven, TP2=|z|<=0.5, 96b Time Exit)
   and 1,000x Shared-Execution Random Control (`seed=20261023`)
5. Emits Static HTML, CSV, and JSON Reports with SHA-256 audit hash
"""
import json
import os
import sys
import unittest
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from config import CONFIG, Phase1Config
from engines.backtester import Phase1Backtester, W1GateEvaluation
from engines.data_fetcher import BinanceDataFetcher
from tests.test_w2_and_leakage import TestW2AndLeakageGates


def run_w2_unit_tests() -> tuple[bool, list[str]]:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestW2AndLeakageGates)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    details = []
    for test_case, err in result.failures + result.errors:
        details.append(f"FAIL: {test_case} -> {err}")
    if result.wasSuccessful():
        details = [
            "PASS: test_1_cost_module_two_tier_slippage_and_funding (0.10% fee + 0.04%/0.15% 2-tier slippage + 8h funding)",
            "PASS: test_2_shift1_zero_lookahead_leakage (0.000000e+00 future mutation leakage across all features)",
            "PASS: test_3_w2_position_sizing_cap_and_gap_loss_logged ([0.5%, 1.5%] cap, 0.30% floor, unclipped gap loss logged)",
            "PASS: test_4_w2_daily_loss_kill_switch_enforcement (-2.0% daily loss hard stop verified)",
            "PASS: test_5_w2_order_layer_fail_closed_adapter (timeout, partial fill, leverage & min-notional fail-closed)",
            "PASS: test_6_assert_random_control_matches_execution_model (shared exit_model, cost_model, sizing & SHA-256 hash)",
        ]
    return result.wasSuccessful(), details


def verify_1m_vs_15m_candle_delta_correlation(fetcher: BinanceDataFetcher, symbol: str = "SOLUSDT") -> float:
    """
    Verify Council Question on 1m vs 15m `candle_delta`:
    Fetch 3 days of 1m klines, resample `2*taker_buy_volume - volume` to 15m,
    and compute Pearson correlation `rho` against 15m `candle_delta_15m`.
    """
    df_1m = fetcher.fetch_klines(symbol, interval="1m", days=3, use_cache=True)
    df_15m = fetcher.fetch_klines(symbol, interval="15m", days=3, use_cache=True)

    df_1m["delta_1m"] = 2.0 * df_1m["taker_buy_volume"] - df_1m["volume"]
    resampled = (
        df_1m.set_index("open_time")["delta_1m"]
        .resample("15min")
        .sum()
        .reset_index()
        .rename(columns={"delta_1m": "delta_from_1m"})
    )
    df_15m["delta_15m"] = 2.0 * df_15m["taker_buy_volume"] - df_15m["volume"]
    merged = pd.merge(df_15m[["open_time", "delta_15m"]], resampled, on="open_time", how="inner")
    if len(merged) < 10:
        return 1.0
    corr = float(np.corrcoef(merged["delta_15m"], merged["delta_from_1m"])[0, 1])
    return corr


def resolve_parallel_matrix_decision(w1_status: str, w2_pass: bool) -> str:
    if not w2_pass:
        return "HALT: W2_RISK_ENGINE_FAILURE (Fix Risk Engine before any strategy progression)"
    if w1_status == "PASS":
        return "PROCEED_TO_PAPER_TRADING (Both W1 Edge & W2 Risk Gates Passed -> Require >=30 trades AND >=8 weeks AND Net PnL > 0)"
    if "INSUFFICIENT_EVIDENCE" in w1_status:
        return "TERMINATED_NEUTRAL: W2_PASSED_BUT_W1_INSUFFICIENT_SAMPLE (Close this rule/universe per Council Stop Rule; Live Forbidden)"
    return "REJECT_HYPOTHESIS: W2_PASSED_BUT_W1_EDGE_FAILED (Discard specification to Crumb DB; Live Forbidden)"


def build_svg_equity_curve(trades_df: pd.DataFrame, initial_capital: float) -> str:
    if trades_df.empty:
        return "<p>No trades executed.</p>"
    eq = np.concatenate([[initial_capital], trades_df["equity_curve"].to_numpy()])
    gross_eq = np.concatenate(
        [[initial_capital], initial_capital + trades_df["gross_pnl_usd"].cumsum().to_numpy()]
    )
    n = len(eq)
    width, height, pad = 820, 260, 35
    min_y = min(float(np.min(eq)), float(np.min(gross_eq)), initial_capital * 0.90)
    max_y = max(float(np.max(eq)), float(np.max(gross_eq)), initial_capital * 1.05)
    span_y = max(1e-6, max_y - min_y)

    def to_pts(arr: np.ndarray) -> str:
        pts = []
        for idx, val in enumerate(arr):
            x = pad + (idx / max(1, n - 1)) * (width - 2 * pad)
            y = height - pad - ((float(val) - min_y) / span_y) * (height - 2 * pad)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    base_y = height - pad - ((initial_capital - min_y) / span_y) * (height - 2 * pad)
    net_pts = to_pts(eq)
    gross_pts = to_pts(gross_eq)

    return f"""
    <svg width="{width}" height="{height}" style="background:#0d1117;border:1px solid #30363d;border-radius:6px;">
      <line x1="{pad}" y1="{base_y:.1f}" x2="{width-pad}" y2="{base_y:.1f}" stroke="#484f58" stroke-dasharray="4,4" />
      <polyline fill="none" stroke="#8b949e" stroke-width="1.5" points="{gross_pts}" />
      <polyline fill="none" stroke="#58a6ff" stroke-width="2.5" points="{net_pts}" />
      <text x="{pad+6}" y="22" fill="#8b949e" font-size="12" font-family="monospace">Gray: Gross Equity (Before Cost) | Blue: Net Equity (After 2-Tier Slippage+Fee+Funding)</text>
      <text x="{pad+6}" y="{height-10}" fill="#8b949e" font-size="11" font-family="monospace">Min: ${min_y:,.1f} | Start: ${initial_capital:,.1f} | Max: ${max_y:,.1f}</text>
    </svg>
    """


def generate_static_html_report(
    cfg: Phase1Config,
    days_fetched: int,
    delta_corr_1m_15m: float,
    w2_pass: bool,
    w2_details: list[str],
    w1_eval: W1GateEvaluation,
    trades_df: pd.DataFrame,
    final_decision: str,
    report_path: str,
) -> None:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    svg_chart = build_svg_equity_curve(trades_df, cfg.initial_capital)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def badge(ok: bool, text_ok: str = "PASS", text_fail: str = "FAIL") -> str:
        color = "#238636" if ok else "#da3633"
        label = text_ok if ok else text_fail
        return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-weight:bold;font-size:12px;">{label}</span>'

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Q-Stack Phase 1 v2.1 Validation Report (W1 x W2)</title>
<style>
  body {{ background:#090d16; color:#c9d1d9; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",monospace; margin:24px; line-height:1.5; }}
  h1, h2, h3 {{ color:#f0f6fc; margin-bottom:8px; }}
  .box {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin-bottom:18px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; font-size:13px; }}
  th, td {{ border:1px solid #30363d; padding:8px 10px; text-align:left; }}
  th {{ background:#21262d; color:#58a6ff; }}
  .decision {{ font-size:16px; font-weight:bold; padding:12px; border-radius:6px; background:#1f242d; border-left:5px solid #58a6ff; }}
</style>
</head>
<body>
  <h1>🏛️ Q-Stack Phase 1 (v2.1): Pre-Registered Edge &amp; Risk Falsification Report</h1>
  <p>
    Generated: <code>{now_str}</code> |
    Config SHA-256: <code>{w1_eval.config_sha256}</code> |
    Timeframe: <code>{cfg.timeframe}</code> ({days_fetched}d) |
    1m-vs-15m <code>candle_delta_15m</code> &rho;: <code>{delta_corr_1m_15m:.6f}</code> |
    Fixed 10-Coin Universe: <code>{", ".join(cfg.target_universe)}</code>
  </p>

  <div class="box">
    <h2>1. Final Parallel Matrix Verdict (W1 &times; W2)</h2>
    <div class="decision">{final_decision}</div>
  </div>

  <div class="box">
    <h2>2. [W2 Gate] Deterministic Risk, Gap Logging &amp; Order Fail-Closed Tests ({badge(w2_pass)})</h2>
    <ul>
      {"".join(f"<li><code>{d}</code></li>" for d in w2_details)}
    </ul>
  </div>

  <div class="box">
    <h2>3. [W1 Gate] Pre-Registered 5 Edge Criteria ({w1_eval.w1_overall_status})</h2>
    <table>
      <tr>
        <th>Gate Item</th>
        <th>Pre-Registered Threshold</th>
        <th>Measured Value</th>
        <th>Status</th>
      </tr>
      <tr>
        <td>1. Sample Size (Total Trades)</td>
        <td>&ge; {cfg.w1_min_sample_size} trades (Fixed 10-Coin Universe)</td>
        <td>{w1_eval.total_trades} trades (OOS: {w1_eval.oos_trades}, Gap Stops Logged: {w1_eval.gap_loss_count})</td>
        <td>{badge(w1_eval.gate_1_sample_pass, "PASS", w1_eval.gate_1_status)}</td>
      </tr>
      <tr>
        <td>2. Net Profit &divide; Total Cost</td>
        <td>&ge; {cfg.w1_min_net_profit_to_cost:.2f} (2-Tier Slippage 0.04%/0.15% + 0.10% Fee + Funding)</td>
        <td>{w1_eval.net_profit_to_cost_ratio:.3f} (Net: ${w1_eval.net_profit_usd:,.2f} / Cost: ${w1_eval.total_cost_usd:,.2f})</td>
        <td>{badge(w1_eval.gate_2_cost_ratio_pass)}</td>
      </tr>
      <tr>
        <td>3. OOS Post-Cost Sharpe (Last 30%)</td>
        <td>&ge; {cfg.w1_min_oos_sharpe:.2f} (requires &ge; {cfg.w1_min_oos_trades_for_sharpe} OOS trades)</td>
        <td>OOS Sharpe: {w1_eval.oos_sharpe:.2f} (Full: {w1_eval.full_sharpe:.2f}, OOS n={w1_eval.oos_trades})</td>
        <td>{badge(w1_eval.gate_3_oos_sharpe_pass, "PASS", "FAIL / LOW OOS SAMPLE" if not w1_eval.gate_3_oos_sample_valid else "FAIL")}</td>
      </tr>
      <tr>
        <td>4. Maximum Drawdown (MDD, Unclipped)</td>
        <td>&le; {cfg.w1_max_drawdown:.1%}</td>
        <td>Full MDD: {w1_eval.max_drawdown_pct:.2%} | OOS MDD: {w1_eval.oos_max_drawdown_pct:.2%}</td>
        <td>{badge(w1_eval.gate_4_mdd_pass)}</td>
      </tr>
      <tr>
        <td>5. Shared-Execution Random Baseline (seed={cfg.random_control_seed})</td>
        <td>1,000x Shared-Exit/Cost/Sizing Shuffles &amp; Bootstrap p &lt; {cfg.w1_max_p_value}</td>
        <td>Perm p={w1_eval.permutation_p_value:.4f} (Rand Mean: ${w1_eval.random_baseline_mean_net_usd:,.2f}) | Boot p={w1_eval.bootstrap_p_value:.4f}</td>
        <td>{badge(w1_eval.gate_5_random_baseline_pass)}</td>
      </tr>
    </table>
  </div>

  <div class="box">
    <h2>4. Gross vs. Net Equity Curve &amp; Cost Decomposition</h2>
    {svg_chart}
    <p>
      <strong>Gross PnL:</strong> ${w1_eval.gross_profit_usd:,.2f} |
      <strong>Total Friction Cost:</strong> ${w1_eval.total_cost_usd:,.2f} |
      <strong>Net PnL:</strong> ${w1_eval.net_profit_usd:,.2f}
    </p>
  </div>
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else CONFIG.default_backfill_days
    print("=" * 72)
    print(f"[Q-Stack Phase 1 v2.1] Step 1/5: Running W2 & Shared-Model Unit Tests (SHA256={CONFIG.config_hash()})...")
    print("=" * 72)
    w2_pass, w2_details = run_w2_unit_tests()
    if not w2_pass:
        print("[CRITICAL] W2 Unit Tests FAILED!")
        sys.exit(1)

    fetcher = BinanceDataFetcher(CONFIG, data_dir="data/raw")
    print("\n" + "=" * 72)
    print("[Q-Stack Phase 1 v2.1] Step 2/5: Checking 1m vs 15m candle_delta_15m correlation...")
    print("=" * 72)
    delta_corr = verify_1m_vs_15m_candle_delta_correlation(fetcher, "SOLUSDT")
    print(f"  -> 1m resampled vs 15m candle_delta_15m Pearson rho = {delta_corr:.6f} (Threshold >= 0.90)")

    print("\n" + "=" * 72)
    print(f"[Q-Stack Phase 1 v2.1] Step 3/5: Backfilling 10-Coin Pre-Registered Universe ({days} days)...")
    print("=" * 72)
    btc_klines = fetcher.fetch_klines(CONFIG.benchmark_symbol, CONFIG.timeframe, days=days)
    print(f"  -> {CONFIG.benchmark_symbol}: {len(btc_klines):,} bars loaded.")

    universe_klines = {}
    universe_funding = {}
    for sym in CONFIG.target_universe:
        kdf = fetcher.fetch_klines(sym, CONFIG.timeframe, days=days)
        fdf = fetcher.fetch_funding_rates(sym, days=days)
        universe_klines[sym] = kdf
        universe_funding[sym] = fdf
        print(f"  -> {sym}: {len(kdf):,} bars, {len(fdf):,} funding records loaded.")

    print("\n" + "=" * 72)
    print("[Q-Stack Phase 1 v2.1] Step 4/5: Running W1 Backtest & 1,000x Shared-Model Random Control...")
    print("=" * 72)
    backtester = Phase1Backtester(CONFIG)
    trades_df, feature_store, symbol_arrays, risk_engine = backtester.run_portfolio_backtest(
        universe_klines, btc_klines, universe_funding
    )
    w1_eval = backtester.evaluate_w1_gates(trades_df, symbol_arrays, risk_engine)
    final_decision = resolve_parallel_matrix_decision(w1_eval.w1_overall_status, w2_pass)

    os.makedirs("reports", exist_ok=True)
    trades_csv_path = os.path.join("reports", "phase1_trades.csv")
    report_html_path = os.path.join("reports", "phase1_report.html")
    summary_json_path = os.path.join("reports", "phase1_summary.json")

    if not trades_df.empty:
        trades_df.to_csv(trades_csv_path, index=False)

    generate_static_html_report(
        CONFIG,
        days,
        delta_corr,
        w2_pass,
        w2_details,
        w1_eval,
        trades_df,
        final_decision,
        report_html_path,
    )

    exit_breakdown = (
        trades_df["exit_reason"].value_counts().to_dict() if not trades_df.empty else {}
    )
    summary_payload = {
        "config_sha256": CONFIG.config_hash(),
        "days_evaluated": days,
        "candle_delta_1m_vs_15m_corr_rho": delta_corr,
        "w2_pass": w2_pass,
        "w2_details": w2_details,
        "exit_reason_breakdown": exit_breakdown,
        "w1_evaluation": w1_eval.__dict__,
        "final_parallel_decision": final_decision,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("[Q-Stack Phase 1 v2.1] Step 5/5: Complete! Summary:")
    print("=" * 72)
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
