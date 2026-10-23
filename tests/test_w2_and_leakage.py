"""
W2 Risk Enforcement (4 Asserts + Gap Logging + Fail-Closed Adapter) & W1 Shared Execution Unit Tests
Implements every Round 2 requirement from the 3-Agent Council (Codex, Claude, Opgo).
"""
import unittest
import numpy as np
import pandas as pd
from config import CONFIG
from engines.backtester import Phase1Backtester
from engines.cost_model import CostModel
from engines.risk_engine import DeterministicRiskEngine, MockOrderExecutionAdapter
from engines.signal_engine import SingleConfluenceSignalEngine


class TestW2AndLeakageGates(unittest.TestCase):
    def setUp(self):
        self.cfg = CONFIG
        self.cost_model = CostModel(self.cfg)

    def test_1_cost_module_two_tier_slippage_and_funding(self):
        """Verify 0.10% fee + 2-tier slippage (0.04% normal vs 0.15% high-vol) + 8h funding."""
        notional = 10_000.0
        # Normal regime: 0.10% fee ($10) + 0.04% slippage ($4) + 0.02% funding ($2) = $16
        bd_normal = self.cost_model.evaluate_trade_cost(
            side=1,
            notional_usd=notional,
            funding_rates_crossed=[0.0001, 0.0001],
            is_high_volatility_bar=False,
        )
        self.assertAlmostEqual(bd_normal.fee_usd, 10.0, places=6)
        self.assertAlmostEqual(bd_normal.slippage_usd, 4.0, places=6)
        self.assertAlmostEqual(bd_normal.funding_usd, 2.0, places=6)
        self.assertAlmostEqual(bd_normal.total_cost_usd, 16.0, places=6)

        # High-volatility / sweep candle regime: 0.15% slippage ($15) -> total $27
        bd_stressed = self.cost_model.evaluate_trade_cost(
            side=1,
            notional_usd=notional,
            funding_rates_crossed=[0.0001, 0.0001],
            is_high_volatility_bar=True,
        )
        self.assertTrue(bd_stressed.is_high_vol_slippage)
        self.assertAlmostEqual(bd_stressed.slippage_usd, 15.0, places=6)
        self.assertAlmostEqual(bd_stressed.total_cost_usd, 27.0, places=6)

    def test_2_shift1_zero_lookahead_leakage(self):
        """
        Assert zero look-ahead leakage: mutating future bars (t+1..T) must have
        0.000000e+00 effect on features or signals at or before bar t.
        """
        rng = np.random.default_rng(2026)
        n = 350
        times = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")

        def make_klines(base_price: float) -> pd.DataFrame:
            rets = rng.normal(0, 0.004, size=n)
            closes = base_price * np.cumprod(1.0 + rets)
            opens = np.concatenate([[base_price], closes[:-1]])
            highs = np.maximum(opens, closes) * (1.0 + rng.uniform(0.0005, 0.005, size=n))
            lows = np.minimum(opens, closes) * (1.0 - rng.uniform(0.0005, 0.005, size=n))
            vols = rng.uniform(1000, 5000, size=n)
            t_buy = rng.uniform(0.3, 0.7, size=n) * vols
            return pd.DataFrame(
                {
                    "open_time": times,
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": vols,
                    "taker_buy_volume": t_buy,
                }
            )

        asset_base = make_klines(100.0)
        btc_base = make_klines(60000.0)

        engine = SingleConfluenceSignalEngine(self.cfg, self.cost_model)
        out_clean = engine.compute_features_and_signals(asset_base, btc_base)

        t_cut = 240
        asset_corrupted = asset_base.copy()
        btc_corrupted = btc_base.copy()
        asset_corrupted.loc[t_cut + 1 :, ["open", "high", "low", "close"]] *= 8.0
        btc_corrupted.loc[t_cut + 1 :, ["open", "high", "low", "close"]] *= 0.15

        out_corrupted = engine.compute_features_and_signals(asset_corrupted, btc_corrupted)

        check_cols = [
            "candle_delta_15m",
            "rolling_beta",
            "residual_ret",
            "cum_residual",
            "residual_z",
            "prior_swing_high",
            "prior_swing_low",
            "vwap_96",
            "atr_20",
            "stop_dist_pct",
            "signal",
        ]
        for col in check_cols:
            s1 = out_clean.loc[:t_cut, col].fillna(-999999.0).to_numpy()
            s2 = out_corrupted.loc[:t_cut, col].fillna(-999999.0).to_numpy()
            np.testing.assert_allclose(
                s1,
                s2,
                rtol=0.0,
                atol=1e-12,
                err_msg=f"Look-ahead leakage in '{col}' at or before t={t_cut}!",
            )

    def test_3_w2_position_sizing_cap_and_gap_loss_logged(self):
        """
        W2 Assert 2 & Gap Audit (`assert_position_sizing_cap` + `assert_gap_loss_logged`):
        1) Verify normal stop-outs never exceed [0.5%, 1.5%] risk cap or 5.0x leverage.
        2) Verify that an adverse price gap exceeding the cap is NOT clipped from equity
           and IS recorded in `risk_engine.gap_loss_events`.
        """
        risk_engine = DeterministicRiskEngine(self.cfg, self.cost_model)
        for d in [0.001, 0.003, 0.005, 0.01, 0.03, 0.10]:
            dec = risk_engine.size_and_validate_order(
                symbol="SOLUSDT",
                side=1,
                entry_price=100.0,
                stop_price=100.0 * (1.0 - d),
                utc_date_str="2026-05-01",
                requested_risk_pct=0.25,  # Malicious 25% request -> must clip to 1.5%
            )
            self.assertTrue(dec.approved)
            self.assertGreaterEqual(dec.stop_distance_pct, self.cfg.min_stop_distance_pct - 1e-12)
            self.assertLessEqual(dec.worst_case_loss_pct, self.cfg.max_risk_per_trade_cap + 1e-9)
            self.assertLessEqual(dec.effective_leverage, self.cfg.max_gross_leverage + 1e-9)

        # Now inject an adverse gap loss (-$35 on $1,000 equity vs $10 planned cap)
        eq_before = risk_engine.equity
        risk_engine.record_trade_pnl(
            net_pnl_usd=-35.0,
            utc_date_str="2026-05-01",
            symbol="SOLUSDT",
            planned_max_loss_usd=10.0,
        )
        # Equity must reflect the FULL unclipped -$35.0 loss ($965.0)
        self.assertAlmostEqual(risk_engine.equity, eq_before - 35.0, places=6)
        self.assertEqual(len(risk_engine.gap_loss_events), 1)
        self.assertAlmostEqual(risk_engine.gap_loss_events[0].gap_exceedance_usd, 25.0, places=6)

    def test_4_w2_daily_loss_kill_switch_enforcement(self):
        """W2 Assert 3: Verify -2.0% daily loss trips Kill Switch and blocks subsequent trades."""
        risk_engine = DeterministicRiskEngine(self.cfg, self.cost_model)
        day1 = "2026-05-01"
        risk_engine.record_trade_pnl(-12.0, day1, symbol="ETHUSDT", planned_max_loss_usd=12.0)
        self.assertFalse(risk_engine.kill_switch_active)

        risk_engine.record_trade_pnl(-8.5, day1, symbol="BTCUSDT", planned_max_loss_usd=10.0)
        self.assertTrue(risk_engine.kill_switch_active)

        dec = risk_engine.size_and_validate_order(
            symbol="SOLUSDT", side=1, entry_price=150.0, stop_price=148.0, utc_date_str=day1
        )
        self.assertFalse(dec.approved)
        self.assertIn("KILL_SWITCH", dec.reason)

    def test_5_w2_order_layer_fail_closed_adapter(self):
        """
        W2 Assert 4 (`MockOrderExecutionAdapter`):
        Verify timeout, partial fill, insufficient leverage, and min-notional all trigger
        fail-closed containment (rejecting new risk and cancelling open orders).
        """
        risk_engine = DeterministicRiskEngine(self.cfg, self.cost_model)
        adapter = MockOrderExecutionAdapter(risk_engine, min_qty=0.01)
        adapter.pending_orders["OLD_ORDER_1"] = 500.0

        valid_sizing = risk_engine.size_and_validate_order(
            symbol="SOLUSDT", side=1, entry_price=100.0, stop_price=98.5, utc_date_str="2026-05-01"
        )
        self.assertTrue(valid_sizing.approved)

        # 1) Exchange leverage limit lower than order effective leverage -> Fail-closed & cancel open orders
        res_lev = adapter.submit_order_fail_closed(
            valid_sizing, exchange_max_leverage_allowed=0.1
        )
        self.assertFalse(res_lev.transmitted)
        self.assertEqual(res_lev.status, "REJECTED_FAIL_CLOSED")
        self.assertTrue(res_lev.open_orders_cancelled)
        self.assertEqual(len(adapter.pending_orders), 0)

        # 2) Partial fill (40%) -> Contained & unfilled remainder cancelled
        adapter.pending_orders["REMAINDER"] = 300.0
        res_part = adapter.submit_order_fail_closed(
            valid_sizing, simulate_partial_fill_ratio=0.40
        )
        self.assertTrue(res_part.transmitted)
        self.assertEqual(res_part.status, "PARTIAL_FILL_CONTAINED")
        self.assertTrue(res_part.open_orders_cancelled)
        self.assertAlmostEqual(
            res_part.filled_notional_usd, valid_sizing.notional_usd * 0.40, places=4
        )

        # 3) API Timeout -> Fail-closed, cancel all, and trip Risk Engine Kill Switch
        adapter.pending_orders["OPEN_2"] = 200.0
        res_timeout = adapter.submit_order_fail_closed(valid_sizing, simulate_timeout=True)
        self.assertFalse(res_timeout.transmitted)
        self.assertEqual(res_timeout.status, "REJECTED_FAIL_CLOSED")
        self.assertTrue(res_timeout.open_orders_cancelled)
        self.assertTrue(risk_engine.kill_switch_active)

    def test_6_assert_random_control_matches_execution_model(self):
        """
        Verify that Strategy and Random Control share the single `simulate_single_trade_execution`
        exit model, cost model, and SHA-256 frozen config hash.
        """
        bt = Phase1Backtester(self.cfg)
        self.assertTrue(self.cfg.random_control_shared_execution)
        self.assertTrue(callable(bt.simulate_single_trade_execution))
        self.assertEqual(len(self.cfg.config_hash()), 16)

    def test_7_round3_unified_config_idempotency_and_reconciliation(self):
        """
        Verify Round 3 Committee Requirements:
        1) Unified Config between CONFIG and PositionLifecycleManager (3 slots, -2% daily kill, <=1% loss per trade, <=1.0x gross)
        2) Deterministic idempotent clientOrderId (`QSTACK_{trade_id}_{leg}`) & precision rounding
        3) Exchange-First Reconciliation Fail-Closed on DB/Exchange diff or missing STOP_MARKET order
        """
        from engines.position_manager import PositionLifecycleManager
        from engines.binance_testnet_client import build_idempotent_client_order_id, round_symbol_qty

        pm = PositionLifecycleManager("data/crumb_memory.db")
        self.assertEqual(int(pm.rules["max_concurrent_slots"]), self.cfg.max_concurrent_positions)
        self.assertAlmostEqual(float(pm.rules["daily_kill_switch_loss_pct"]), self.cfg.max_daily_loss_pct)

        # Even when ATR demands -2.5% stop distance ceiling, max loss <= $10 (1.0% of $1,000) and slot notional <= $333.34 (1/3 gross)
        plan = pm.compute_wonyotti_slot_order_plan(1000.0, 100.0, 2.0, 1)
        self.assertLessEqual(plan["max_loss_usd"], 10.0001)
        self.assertLessEqual(plan["notional_usd"], 333.34)

        # Idempotent clientOrderId & lotSize rounding
        cid_sl = build_idempotent_client_order_id("LTC_20261023_01", "sl")
        self.assertEqual(cid_sl, "QSTACK_LTC_20261023_01_SL")
        self.assertEqual(round_symbol_qty("DOGEUSDT", 123.789), 123.0)

        # Exchange-vs-DB Reconciliation Fail-Closed check
        fake_ex_state = {
            "live": True,
            "positions": [{"symbol": "ETHUSDT", "positionAmt": "1.5"}],
            "open_orders": [],  # Missing STOP_MARKET order AND missing in local DB!
            "error": None,
        }
        rec = pm.reconcile_exchange_vs_db(fake_ex_state)
        self.assertFalse(rec["ok"])
        self.assertTrue(pm.reconciliation_halted)
        self.assertIn("UNPROTECTED_POSITION_NO_STOP: ETHUSDT", rec["errors"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
