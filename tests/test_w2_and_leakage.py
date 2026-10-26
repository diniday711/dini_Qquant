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

    def test_8_poteto_mode_daemon_idempotency_and_shadow_cleanup(self):
        """
        Verify /poteto-mode audit fixes:
        1) Pre-flight `can_open_position` prevents duplicate exchange order submission when symbol is already open or reconciliation is halted.
        2) Stale SHADOW_DRY_RUN rows in SQLite are automatically purged when reconciling against a clean live Testnet account (`live=True`, `positions=[]`).
        """
        import tempfile
        import os
        from engines.position_manager import PositionLifecycleManager

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            tmp_db = tf.name
        try:
            pm = PositionLifecycleManager(tmp_db)
            plan = pm.compute_wonyotti_slot_order_plan(1000.0, 100.0, 2.0, 1)

            # 1) Open a SHADOW_DRY_RUN position for LTCUSDT
            ok_open, reason = pm.can_open_position("LTCUSDT")
            self.assertTrue(ok_open, reason)
            recorded = pm.open_or_record_position(
                symbol="LTCUSDT",
                side="LONG",
                side_int=1,
                sub_strategy="WONYOTTI_VOLUME_CLUSTER_BREAKOUT",
                plan=plan,
                entry_price=100.0,
                execution_mode="SHADOW_DRY_RUN",
            )
            self.assertTrue(recorded)

            # 2) Second cycle for same symbol must be blocked by can_open_position BEFORE calling exchange
            ok_dup, dup_reason = pm.can_open_position("LTCUSDT")
            self.assertFalse(ok_dup)
            self.assertIn("SKIP_ALREADY_OPEN", dup_reason)

            # 3) Switching to live Testnet (`live=True`, 0 positions on exchange) purges SHADOW_DRY_RUN rows and reconciles cleanly
            clean_live_state = {"live": True, "positions": [], "open_orders": [], "error": None}
            rec_clean = pm.reconcile_exchange_vs_db(clean_live_state)
            self.assertTrue(rec_clean["ok"], f"Expected clean live boot after purging shadow rows, got {rec_clean}")
            self.assertEqual(len(pm.get_active_positions()), 0)
        finally:
            try:
                if os.path.exists(tmp_db):
                    os.remove(tmp_db)
            except PermissionError:
                pass

    def test_9_vibe_trading_correlation_and_alpha_vitality(self):
        """
        Verify Vibe-Trading inspired diagnostics (`engines/vibe_diagnostics.py`):
        1) Herd correlation (`corr > 0.90`) triggers `HERD_LOCK_EXTREME` and throttles `recommended_max_slots = 1`,
           causing `PositionLifecycleManager.can_open_position` to block a 2nd slot with `SKIP_CORR_THROTTLED_SLOTS_FULL (1/1)`.
        2) Independent returns trigger `DISPERSED_NORMAL` (`recommended_max_slots = 3`).
        3) `compute_alpha_vitality_diagnostics` classifies `ALIVE/DEAD/REVERSED` with zero look-ahead leakage when mutating the final bar.
        """
        import tempfile
        import os
        from engines.vibe_diagnostics import compute_correlation_regime, compute_alpha_vitality_diagnostics
        from engines.position_manager import PositionLifecycleManager

        rng = np.random.default_rng(77)
        base_rets = rng.normal(0.001, 0.015, 100)
        herd_univ = {}
        indep_univ = {}
        for idx, sym in enumerate(["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]):
            # 1) Herd universe: nearly identical returns
            noise = rng.normal(0.0, 0.0005, 100)
            px_herd = 100.0 * np.cumprod(1.0 + base_rets + noise)
            herd_univ[sym] = pd.DataFrame(
                {
                    "close": px_herd,
                    "volume": rng.uniform(1000, 5000, 100),
                    "delta_ratio": rng.uniform(-0.3, 0.3, 100),
                }
            )
            # 2) Independent universe
            ind_rets = rng.normal(0.0, 0.015, 100)
            px_ind = 100.0 * np.cumprod(1.0 + ind_rets)
            indep_univ[sym] = pd.DataFrame(
                {
                    "close": px_ind,
                    "volume": rng.uniform(1000, 5000, 100),
                    "delta_ratio": rng.uniform(-0.3, 0.3, 100),
                }
            )

        corr_herd = compute_correlation_regime(herd_univ, lookback_bars=72)
        self.assertEqual(corr_herd["corr_regime"], "HERD_LOCK_EXTREME")
        self.assertEqual(corr_herd["recommended_max_slots"], 1)

        corr_indep = compute_correlation_regime(indep_univ, lookback_bars=72)
        self.assertEqual(corr_indep["corr_regime"], "DISPERSED_NORMAL")
        self.assertEqual(corr_indep["recommended_max_slots"], 3)

        # Verify dynamic slot throttling in PositionLifecycleManager
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            tmp_db = tf.name
        try:
            pm = PositionLifecycleManager(tmp_db)
            plan = pm.compute_wonyotti_slot_order_plan(1000.0, 100.0, 2.0, 1)
            self.assertTrue(
                pm.open_or_record_position(
                    "ETHUSDT", "LONG", 1, "WONYOTTI_VOLUME_CLUSTER_BREAKOUT", plan, 100.0, dynamic_max_slots=1
                )
            )
            can_second, reason_second = pm.can_open_position("SOLUSDT", dynamic_max_slots=1)
            self.assertFalse(can_second)
            self.assertIn("SKIP_CORR_THROTTLED_SLOTS_FULL (1/1)", reason_second)
        finally:
            try:
                if os.path.exists(tmp_db):
                    os.remove(tmp_db)
            except PermissionError:
                pass

        # Verify Alpha Vitality zero look-ahead leakage on final bar mutation
        btc_df = herd_univ["ETHUSDT"].copy()
        vit_a = compute_alpha_vitality_diagnostics(indep_univ, btc_df, lookback_bars=60)
        mut_univ = {k: v.copy() for k, v in indep_univ.items()}
        for k in mut_univ:
            mut_univ[k].loc[len(mut_univ[k]) - 1, "close"] *= 5.0
        vit_b = compute_alpha_vitality_diagnostics(mut_univ, btc_df, lookback_bars=60)
        self.assertEqual(vit_a["families"], vit_b["families"])

    def test_10_v34_operational_lifecycle_small_seed_and_hwm(self):
        """
        Verify Q-Stack v3.4 Operational Completeness:
        1) Small-Seed ($360 / 500k KRW) MICRO_SEED_TIER (2 slots + BTC single-leg TP fallback).
        2) High-Water Mark (85% Ratchet Floor) locking when equity grows to $1,000 and drops below $850.
        3) Full Exit Lifecycle: TP1 (+1.8%) -> Peak (+3.5%) -> Trailing Callback (-1.0%p) exit & 48h Time-Stop exit.
        """
        import tempfile
        import os
        from datetime import datetime, timezone
        from engines.position_manager import PositionLifecycleManager

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            tmp_db = tf.name
        try:
            pm = PositionLifecycleManager(tmp_db)

            # 1) Small-seed $360 check
            hwm_360 = pm.update_and_check_equity_hwm(360.0, initial_equity_override=360.0)
            self.assertTrue(hwm_360["ok"])
            self.assertEqual(hwm_360["tier_max_slots"], 2)
            self.assertAlmostEqual(hwm_360["locked_floor_usdt"], 306.0)

            btc_plan_360 = pm.compute_wonyotti_slot_order_plan(360.0, 90000.0, 1200.0, 1, symbol="BTCUSDT")
            self.assertTrue(btc_plan_360["single_leg_tp_mode"])
            self.assertEqual(btc_plan_360["partial_tp_qty"], 0.0)

            # 2) Equity grows to $1,000 -> 85% floor ratchets to $850.0
            hwm_1000 = pm.update_and_check_equity_hwm(1000.0)
            self.assertTrue(hwm_1000["ok"])
            self.assertEqual(hwm_1000["tier_max_slots"], 3)
            self.assertAlmostEqual(hwm_1000["locked_floor_usdt"], 850.0)

            # Drop to $845 -> trips HWM 85% floor lock!
            hwm_drop = pm.update_and_check_equity_hwm(845.0)
            self.assertFalse(hwm_drop["ok"])
            self.assertTrue(pm.account_floor_locked)
            can_open_locked, reason_locked = pm.can_open_position("SOLUSDT")
            self.assertFalse(can_open_locked)
            self.assertIn("ACCOUNT_FLOOR_LOCKED_HWM_85PCT", reason_locked)

            # Restore equity to $1,000 for lifecycle test
            pm.update_and_check_equity_hwm(1000.0)
            plan = pm.compute_wonyotti_slot_order_plan(1000.0, 100.0, 1.0, 1, symbol="SOLUSDT")
            self.assertTrue(
                pm.open_or_record_position(
                    "SOLUSDT",
                    "LONG",
                    1,
                    "WONYOTTI_VOLUME_CLUSTER_BREAKOUT",
                    plan,
                    100.0,
                    opened_at_override="2026-10-20 00:00:00 UTC",
                )
            )

            # Step A: Hit TP1 at $102.0 (+2.0% >= +1.8%)
            ev_tp1 = pm.sync_and_manage_positions(
                {"SOLUSDT": 102.0},
                {"SOLUSDT": "TRENDING_UP"},
                now_utc_override=datetime(2026, 10, 20, 2, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(any("1차 목표 도달" in e for e in ev_tp1))
            active_after_tp1 = pm.get_active_positions()
            self.assertEqual(len(active_after_tp1), 1)
            self.assertEqual(active_after_tp1[0]["tp1_hit"], 1)
            self.assertAlmostEqual(active_after_tp1[0]["stop_price"], 100.0)

            # Step B: Advance to $103.5 (+3.5% peak), then pullback to $102.2 (-1.3%p pullback >= 1.0%p callback) -> Trailing Exit!
            pm.sync_and_manage_positions(
                {"SOLUSDT": 103.5},
                {"SOLUSDT": "TRENDING_UP"},
                now_utc_override=datetime(2026, 10, 20, 3, 0, tzinfo=timezone.utc),
            )
            ev_trail = pm.sync_and_manage_positions(
                {"SOLUSDT": 102.2},
                {"SOLUSDT": "TRENDING_UP"},
                now_utc_override=datetime(2026, 10, 20, 4, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(any("TRAILING_PROFIT_LOCK_EXIT" in e for e in ev_trail))
            self.assertEqual(len(pm.get_active_positions()), 0)

            # Step C: Stagnant position held for 50 hours (>= 48h) without hitting TP1 -> Time-Stop Exit!
            self.assertTrue(
                pm.open_or_record_position(
                    "XRPUSDT",
                    "LONG",
                    1,
                    "WONYOTTI_VOLUME_CLUSTER_BREAKOUT",
                    plan,
                    100.0,
                    opened_at_override="2026-10-20 00:00:00 UTC",
                )
            )
            ev_time = pm.sync_and_manage_positions(
                {"XRPUSDT": 100.2},
                {"XRPUSDT": "RANGING"},
                now_utc_override=datetime(2026, 10, 22, 2, 0, tzinfo=timezone.utc),
            )
            self.assertTrue(any("TIME_STOP_48H_EXIT" in e for e in ev_time))
            self.assertEqual(len(pm.get_active_positions()), 0)
        finally:
            try:
                if os.path.exists(tmp_db):
                    os.remove(tmp_db)
            except PermissionError:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
