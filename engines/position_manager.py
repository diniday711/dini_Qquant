"""
Unified W2 Isolated Position & Full Lifecycle Manager (`engines/position_manager.py`)
Implements Round 3 Committee Synthesis + v3.4 Operational Completeness:
1. Unified Config with `CONFIG` (`config.py`):
   - `max_concurrent_slots = 3` (`2` in `MICRO_SEED_TIER` `< $600`)
   - `daily_kill_switch_loss_pct = 0.020` (`-2.0%`)
   - `risk_per_trade_pct = 0.010` (`1.0%` max loss per trade even at `-2.5%` stop ceiling)
   - `max_gross_leverage = 1.0` (`<= 1.0x` total gross equity)
2. High-Water Mark (`85%` Ratchet Floor) & Small-Seed (`$360` / 500k KRW) Scaling (`MIN_NOTIONAL` safe).
3. Complete Exit Lifecycle:
   - `HARD_STOP_LOSS` (`-2.5%` ceiling) / `BREAKEVEN_STOP_EXIT`
   - `TP1` (`+1.8%` 50% partial take-profit + move stop to breakeven)
   - `TP2` (`+2.8%` reversion / `+4.5%` trend breakout full exit) & `TRAILING_PROFIT_LOCK_EXIT` (`-1.0%p` pullback from peak after TP1)
   - `TIME_STOP_48H_EXIT` (stagnant position cleanup after `max_hold_hours = 48h`)
   - `EMERGENCY_REGIME_FLIP_EXIT` & Operator `panic_close_all_positions`
"""
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from config import CONFIG
from engines.binance_testnet_client import SYMBOL_PRECISION_MAP, round_symbol_qty

RULES_PATH = Path(__file__).parent.parent / "user_trading_rules.json"


def load_user_trading_rules() -> Dict[str, Any]:
    default_rules = {
        "profile_name": "W2_Unified_Isolated_Hybrid_v3_4",
        "initial_capital_usdt": CONFIG.initial_capital,
        "micro_seed_threshold_usdt": 600.0,
        "micro_seed_max_slots": 2,
        "margin_mode": "ISOLATED",
        "max_concurrent_slots": CONFIG.max_concurrent_positions,  # 3 slots (Unified with W2)
        "slot_allocation_pct": 0.20,
        "risk_per_trade_pct": CONFIG.risk_per_trade_pct,          # 1.0% risk cap
        "max_gross_leverage": 1.0,                                # <= 1.0x total gross leverage
        "max_slot_leverage": int(CONFIG.max_gross_leverage),      # max 5x isolated slot leverage
        "stop_loss_mode": "HYBRID_ATR_CAPPED",
        "min_stop_distance_pct": 0.008,
        "max_stop_distance_pct": 0.025,
        "atr_stop_multiplier": 2.0,
        "take_profit_mode": "SPLIT_50_BREAKEVEN_TRAIL",
        "partial_tp_target_pct": 0.018,
        "partial_tp_ratio": 0.50,
        "tp2_trend_target_pct": 0.045,
        "tp2_reversion_target_pct": 0.028,
        "trailing_Callback_pct": 0.010,
        "max_hold_hours": 48.0,
        "daily_kill_switch_loss_pct": CONFIG.max_daily_loss_pct,  # -2.0% daily kill switch (Unified with W2)
        "hwm_floor_lock_ratio": 0.85,
        "trading_paused": False,
        "excluded_symbols": [],
    }
    if RULES_PATH.exists():
        try:
            loaded = json.loads(RULES_PATH.read_text(encoding="utf-8"))
            default_rules.update(loaded)
        except Exception:
            pass
    # Force hard W2 invariants so user_trading_rules.json can never silently relax W2
    default_rules["max_concurrent_slots"] = min(
        int(default_rules["max_concurrent_slots"]), CONFIG.max_concurrent_positions
    )
    default_rules["daily_kill_switch_loss_pct"] = min(
        float(default_rules["daily_kill_switch_loss_pct"]), CONFIG.max_daily_loss_pct
    )
    default_rules["risk_per_trade_pct"] = min(
        float(default_rules["risk_per_trade_pct"]), CONFIG.max_risk_per_trade_cap
    )
    return default_rules


def set_operator_Pause_state(paused: bool) -> Dict[str, Any]:
    rules = load_user_trading_rules()
    rules["trading_paused"] = bool(paused)
    RULES_PATH.write_text(json.dumps(rules, indent=2, ensure_ascii=False), encoding="utf-8")
    return rules


class PositionLifecycleManager:
    def __init__(self, db_path: str = "data/crumb_memory.db"):
        self.db_path = db_path
        self.rules = load_user_trading_rules()
        self.reconciliation_halted: bool = False
        self.reconciliation_errors: List[str] = []
        self.account_floor_locked: bool = False
        self._init_table()

    def _init_table(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS active_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    opened_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    symbol TEXT NOT NULL UNIQUE,
                    side TEXT NOT NULL,
                    side_int INTEGER NOT NULL,
                    sub_strategy TEXT NOT NULL,
                    margin_mode TEXT NOT NULL,
                    slot_margin_usd REAL NOT NULL,
                    notional_usd REAL NOT NULL,
                    leverage REAL NOT NULL,
                    initial_qty REAL NOT NULL,
                    remaining_qty REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    tp1_price REAL NOT NULL,
                    tp1_hit INTEGER NOT NULL DEFAULT 0,
                    realized_pnl_usd REAL NOT NULL DEFAULT 0.0,
                    unrealized_pnl_usd REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    execution_mode TEXT NOT NULL DEFAULT 'SHADOW_DRY_RUN',
                    peak_price REAL NOT NULL DEFAULT 0.0
                )
                """
            )
            cols = [r[1] for r in conn.execute("PRAGMA table_info(active_positions)").fetchall()]
            if "execution_mode" not in cols:
                conn.execute(
                    "ALTER TABLE active_positions ADD COLUMN execution_mode TEXT NOT NULL DEFAULT 'SHADOW_DRY_RUN'"
                )
            if "peak_price" not in cols:
                conn.execute(
                    "ALTER TABLE active_positions ADD COLUMN peak_price REAL NOT NULL DEFAULT 0.0"
                )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS closed_positions_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    sub_strategy TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    notional_usd REAL NOT NULL,
                    tp1_hit INTEGER NOT NULL,
                    net_pnl_usd REAL NOT NULL,
                    close_reason TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS account_watermarks (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    initial_equity_usdt REAL NOT NULL,
                    peak_equity_usdt REAL NOT NULL,
                    locked_floor_usdt REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def update_and_check_equity_hwm(
        self,
        current_equity_usdt: float,
        initial_equity_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        High-Water Mark (`85%` Ratchet Floor) & Small-Seed (`$360` / `< $600`) Tier Classifier:
        - Automatically ratchets `locked_floor_usdt = max(initial_equity * 0.85, peak_equity * 0.85)`.
        - If `current_equity_usdt <= locked_floor_usdt`, trips `self.account_floor_locked = True`.
        - If `current_equity_usdt < 600.0`, selects `MICRO_SEED_TIER` (`tier_max_slots = 2`).
        """
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        hwm_ratio = float(self.rules.get("hwm_floor_lock_ratio", 0.85))
        micro_thresh = float(self.rules.get("micro_seed_threshold_usdt", 600.0))

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM account_watermarks WHERE id = 1").fetchone()
            if row is None:
                init_eq = float(initial_equity_override or current_equity_usdt)
                peak_eq = max(init_eq, float(current_equity_usdt))
                floor_eq = peak_eq * hwm_ratio
                conn.execute(
                    "INSERT INTO account_watermarks (id, initial_equity_usdt, peak_equity_usdt, locked_floor_usdt, updated_at) VALUES (1, ?, ?, ?, ?)",
                    (init_eq, peak_eq, floor_eq, now_iso),
                )
            else:
                init_eq = float(initial_equity_override or row[1])
                peak_eq = max(float(row[2]), float(current_equity_usdt))
                floor_eq = max(init_eq * hwm_ratio, peak_eq * hwm_ratio)
                conn.execute(
                    "UPDATE account_watermarks SET initial_equity_usdt = ?, peak_equity_usdt = ?, locked_floor_usdt = ?, updated_at = ? WHERE id = 1",
                    (init_eq, peak_eq, floor_eq, now_iso),
                )
            conn.commit()

        self.account_floor_locked = bool(current_equity_usdt <= floor_eq + 1e-6)
        is_micro = bool(current_equity_usdt < micro_thresh)
        tier_name = "MICRO_SEED_TIER (50만원 소액 보호 모드)" if is_micro else "STANDARD_W2_TIER (표준 복리 모드)"
        tier_max_slots = (
            int(self.rules.get("micro_seed_max_slots", 2))
            if is_micro
            else int(self.rules["max_concurrent_slots"])
        )

        return {
            "ok": not self.account_floor_locked,
            "account_floor_locked": self.account_floor_locked,
            "current_equity_usdt": round(float(current_equity_usdt), 2),
            "initial_equity_usdt": round(init_eq, 2),
            "peak_equity_usdt": round(peak_eq, 2),
            "locked_floor_usdt": round(floor_eq, 2),
            "tier_name": tier_name,
            "tier_max_slots": tier_max_slots,
        }

    def reconcile_exchange_vs_db(self, exchange_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exchange-First Reconciliation (`positionRisk` + `openOrders` is Source of Truth):
        - If exchange reports an error, or if live exchange positions do not match SQLite `active_positions`,
          or if any live exchange position lacks an active `STOP_MARKET` order, immediately trip
          `self.reconciliation_halted = True` (Fail-Closed).
        """
        self.reconciliation_halted = False
        self.reconciliation_errors = []

        if exchange_state.get("error"):
            self.reconciliation_halted = True
            self.reconciliation_errors.append(f"EXCHANGE_API_ERROR: {exchange_state['error']}")
            return {"ok": False, "errors": self.reconciliation_errors}

        if not exchange_state.get("live", False):
            return {"ok": True, "mode": "SHADOW_CACHE_OK", "errors": []}

        # Purge any local SHADOW_DRY_RUN rows before reconciling against a live exchange
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM active_positions WHERE execution_mode = 'SHADOW_DRY_RUN'")
            conn.commit()

        db_positions = {p["symbol"]: p for p in self.get_active_positions()}
        ex_positions = {
            p["symbol"]: float(p.get("positionAmt", 0.0))
            for p in exchange_state.get("positions", [])
            if abs(float(p.get("positionAmt", 0.0))) > 0.0
        }
        open_orders = exchange_state.get("open_orders", [])

        # 1. Check Symbol Set Diff between Exchange and SQLite Cache
        if set(db_positions.keys()) != set(ex_positions.keys()):
            self.reconciliation_halted = True
            self.reconciliation_errors.append(
                f"DB_EXCHANGE_POSITION_MISMATCH: Exchange={list(ex_positions.keys())} vs SQLite={list(db_positions.keys())}"
            )

        # 2. Verify every live Exchange position has an active STOP_MARKET order
        for sym in ex_positions:
            has_stop = any(
                o.get("symbol") == sym and o.get("type") == "STOP_MARKET"
                for o in open_orders
            )
            if not has_stop:
                self.reconciliation_halted = True
                self.reconciliation_errors.append(f"UNPROTECTED_POSITION_NO_STOP: {sym}")

        return {"ok": not self.reconciliation_halted, "errors": self.reconciliation_errors}

    def compute_wonyotti_slot_order_plan(
        self,
        equity_usdt: float,
        entry_price: float,
        atr_val: float,
        side_int: int,
        symbol: str = "ETHUSDT",
    ) -> Dict[str, Any]:
        """
        Compute exact entry quantity, stop-loss price (ATR-based capped at -2.5%),
        1st partial take-profit price (+1.8%), and 2nd full take-profit price (+4.5%)
        with Small-Seed (`$360` / `< $600`) `MIN_NOTIONAL` / `min_qty` safety.
        """
        slot_pct = float(self.rules["slot_allocation_pct"])          # 20%
        micro_thresh = float(self.rules.get("micro_seed_threshold_usdt", 600.0))
        if equity_usdt < micro_thresh:
            max_slots = int(self.rules.get("micro_seed_max_slots", 2))
        else:
            max_slots = int(self.rules["max_concurrent_slots"])      # 3 slots
        max_gross_lev = float(self.rules.get("max_gross_leverage", 1.0))
        min_stop = float(self.rules["min_stop_distance_pct"])        # 0.8%
        max_stop = float(self.rules["max_stop_distance_pct"])        # 2.5% hard ceiling!
        atr_mult = float(self.rules["atr_stop_multiplier"])          # 2.0x ATR
        tp1_pct = float(self.rules["partial_tp_target_pct"])         # +1.8% 1st TP
        tp2_pct = float(self.rules.get("tp2_trend_target_pct", 0.045))
        risk_pct = float(self.rules["risk_per_trade_pct"])           # 1.0% risk per trade

        raw_atr_stop_pct = (atr_mult * atr_val) / max(1e-9, entry_price)
        stop_dist_pct = max(min_stop, min(max_stop, raw_atr_stop_pct))

        slot_margin_usd = equity_usdt * (1.0 / max(1, max_slots) if equity_usdt < micro_thresh else slot_pct)
        target_risk_usd = equity_usdt * risk_pct

        desired_notional = target_risk_usd / stop_dist_pct
        max_slot_notional = (equity_usdt * max_gross_lev) / max(1, max_slots)
        notional_usd = min(desired_notional, max_slot_notional)

        slot_leverage = max(1.0, notional_usd / max(1e-9, slot_margin_usd))
        qty = notional_usd / max(1e-9, entry_price)

        # Small-seed MIN_NOTIONAL & min_qty guard for 50% split TP
        spec = SYMBOL_PRECISION_MAP.get(symbol, {"min_qty": 0.001, "min_notional": 5.0})
        raw_half_qty = round_symbol_qty(symbol, qty * float(self.rules["partial_tp_ratio"]))
        half_notional = raw_half_qty * entry_price
        single_leg_tp_mode = bool(
            equity_usdt < micro_thresh
            and (
                notional_usd < 100.0
                or raw_half_qty < float(spec.get("min_qty", 0.001))
                or half_notional < float(spec.get("min_notional", 5.0))
            )
        )
        partial_tp_qty = 0.0 if single_leg_tp_mode else qty * float(self.rules["partial_tp_ratio"])

        if side_int == 1:
            stop_px = entry_price * (1.0 - stop_dist_pct)
            tp1_px = entry_price * (1.0 + tp1_pct)
            tp2_px = entry_price * (1.0 + tp2_pct)
        else:
            stop_px = entry_price * (1.0 + stop_dist_pct)
            tp1_px = entry_price * (1.0 - tp1_pct)
            tp2_px = entry_price * (1.0 - tp2_pct)

        max_loss_usd = notional_usd * stop_dist_pct

        return {
            "slot_margin_usd": slot_margin_usd,
            "notional_usd": notional_usd,
            "slot_leverage": slot_leverage,
            "quantity": qty,
            "partial_tp_qty": partial_tp_qty,
            "single_leg_tp_mode": single_leg_tp_mode,
            "tier_max_slots": max_slots,
            "stop_dist_pct": stop_dist_pct,
            "stop_price": stop_px,
            "tp1_pct": tp1_pct,
            "tp1_price": tp1_px,
            "tp2_pct": tp2_pct,
            "tp2_price": tp2_px,
            "max_loss_usd": max_loss_usd,
        }

    def _close_and_archive_position(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        exit_price: float,
        total_pnl: float,
        reason: str,
        now_iso: str,
        testnet_client: Any = None,
        execute_on_testnet: bool = False,
    ) -> None:
        sym = row["symbol"]
        if testnet_client is not None:
            testnet_client.close_position_market(
                symbol=sym,
                side_int=int(row["side_int"]),
                quantity=float(row["remaining_qty"]),
                execute_on_testnet=execute_on_testnet,
            )
        conn.execute("DELETE FROM active_positions WHERE symbol = ?", (sym,))
        conn.execute(
            """
            INSERT INTO closed_positions_history (
                opened_at, closed_at, symbol, side, sub_strategy,
                entry_price, exit_price, notional_usd, tp1_hit, net_pnl_usd, close_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["opened_at"],
                now_iso,
                sym,
                row["side"],
                row["sub_strategy"],
                float(row["entry_price"]),
                exit_price,
                float(row["notional_usd"]),
                int(row["tp1_hit"]),
                total_pnl,
                reason,
            ),
        )

    def sync_and_manage_positions(
        self,
        latest_prices: Dict[str, float],
        coin_regimes_map: Dict[str, str],
        testnet_client: Any = None,
        execute_on_testnet: bool = False,
        global_regime_code: str = "SELECTIVE_ROTATION",
        now_utc_override: Optional[datetime] = None,
    ) -> List[str]:
        events: List[str] = []
        now_dt = now_utc_override or datetime.now(timezone.utc)
        now_iso = now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

        if testnet_client is not None:
            ex_state = testnet_client.fetch_exchange_positions_and_orders()
            rec = self.reconcile_exchange_vs_db(ex_state)
            if not rec["ok"]:
                events.append(f"🚨 [FAIL-CLOSED 거래소 대사 불일치] {'; '.join(rec['errors'])} -> 신규 주문 차단!")
                return events

        max_hold_hours = float(self.rules.get("max_hold_hours", 48.0))
        tp2_trend_pct = float(self.rules.get("tp2_trend_target_pct", 0.045))
        tp2_rev_pct = float(self.rules.get("tp2_reversion_target_pct", 0.028))
        trail_cb_pct = float(self.rules.get("trailing_Callback_pct", 0.010))

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM active_positions WHERE status = 'OPEN'").fetchall()
            max_slots = int(self.rules["max_concurrent_slots"])
            if len(rows) > max_slots:
                for extra_r in rows[max_slots:]:
                    conn.execute("DELETE FROM active_positions WHERE id = ?", (extra_r["id"],))
                rows = rows[:max_slots]

            for r in rows:
                sym = r["symbol"]
                cur_px = latest_prices.get(sym, float(r["entry_price"]))
                side_int = int(r["side_int"])
                ep = float(r["entry_price"])
                sp = float(r["stop_price"])
                tp1 = float(r["tp1_price"])
                tp1_hit = int(r["tp1_hit"])
                rem_qty = float(r["remaining_qty"])
                init_qty = float(r["initial_qty"])
                realized = float(r["realized_pnl_usd"])
                sub_strat = str(r["sub_strategy"])
                prev_peak = float(r["peak_price"]) if float(r["peak_price"]) > 0 else ep
                peak_px = max(prev_peak, cur_px) if side_int == 1 else min(prev_peak, cur_px)

                # 1. Hard Stop-Loss or Breakeven Stop Exit
                stop_triggered = (side_int == 1 and cur_px <= sp) or (side_int == -1 and cur_px >= sp)
                if stop_triggered:
                    pnl_leg = rem_qty * side_int * (cur_px - ep)
                    total_pnl = realized + pnl_leg
                    reason = "BREAKEVEN_STOP_EXIT (본절 탈출)" if tp1_hit else "HARD_STOP_LOSS (-2.5% 이내 칼손절)"
                    self._close_and_archive_position(
                        conn, r, cur_px, total_pnl, reason, now_iso, testnet_client, execute_on_testnet
                    )
                    events.append(f"🛑 [{sym}] {reason} 완료 (청산가 ${cur_px:,.4f} | 확정 손익: ${total_pnl:+,.2f})")
                    continue

                # 2. Emergency Geopolitical Shock / Regime Flip Exit
                coin_reg = coin_regimes_map.get(sym, "")
                regime_flipped = (
                    global_regime_code == "GEOPOLITICAL_SHOCK"
                    or (side_int == 1 and coin_reg == "TRENDING_DOWN" and not tp1_hit)
                    or (side_int == -1 and coin_reg == "TRENDING_UP" and not tp1_hit)
                )
                if regime_flipped:
                    pnl_leg = rem_qty * side_int * (cur_px - ep)
                    total_pnl = realized + pnl_leg
                    reason = f"EMERGENCY_REGIME_FLIP_EXIT ({global_regime_code}/{coin_reg})"
                    self._close_and_archive_position(
                        conn, r, cur_px, total_pnl, reason, now_iso, testnet_client, execute_on_testnet
                    )
                    events.append(f"⚡ [{sym}] 장세 급반전 비상 청산 완료 ({reason} | 확정 손익: ${total_pnl:+,.2f})")
                    continue

                # 3. 2nd Target (TP2) Full Exit or Post-TP1 Trailing Callback Exit
                ret_pct = side_int * ((cur_px - ep) / max(1e-9, ep))
                peak_ret_pct = side_int * ((peak_px - ep) / max(1e-9, ep))
                is_reversion = ("REVERSION" in sub_strat) or ("ABSORPTION" in sub_strat)
                tp2_target_pct = tp2_rev_pct if is_reversion else tp2_trend_pct

                if ret_pct >= tp2_target_pct:
                    pnl_leg = rem_qty * side_int * (cur_px - ep)
                    total_pnl = realized + pnl_leg
                    reason = f"TP2_FULL_TARGET_EXIT (+{tp2_target_pct:.1%} 최종 목표 도달 전량 익절)"
                    self._close_and_archive_position(
                        conn, r, cur_px, total_pnl, reason, now_iso, testnet_client, execute_on_testnet
                    )
                    events.append(f"🏆 [{sym}] {reason} 완료! (청산가 ${cur_px:,.4f} | 총 확정 수익: ${total_pnl:+,.2f})")
                    continue

                if tp1_hit and (peak_ret_pct - ret_pct) >= trail_cb_pct and ret_pct > 0.002:
                    pnl_leg = rem_qty * side_int * (cur_px - ep)
                    total_pnl = realized + pnl_leg
                    reason = f"TRAILING_PROFIT_LOCK_EXIT (최고 수익 +{peak_ret_pct:.2%} 찍고 -{trail_cb_pct:.1%}p 꺾여 수익 확정)"
                    self._close_and_archive_position(
                        conn, r, cur_px, total_pnl, reason, now_iso, testnet_client, execute_on_testnet
                    )
                    events.append(f"💰 [{sym}] {reason} 완료! (청산가 ${cur_px:,.4f} | 총 확정 수익: ${total_pnl:+,.2f})")
                    continue

                # 4. 48-Hour Time-Stop for stagnant positions that haven't reached TP1
                try:
                    opened_dt = datetime.strptime(str(r["opened_at"]).replace(" UTC", ""), "%Y-%m-%d %H:%M:%S").replace(
                        tzinfo=timezone.utc
                    )
                    hold_hours = (now_dt - opened_dt).total_seconds() / 3600.0
                except Exception:
                    hold_hours = 0.0

                if (not tp1_hit) and hold_hours >= max_hold_hours:
                    pnl_leg = rem_qty * side_int * (cur_px - ep)
                    total_pnl = realized + pnl_leg
                    reason = f"TIME_STOP_48H_EXIT ({hold_hours:.1f}h 경과 시간손절 슬롯 회전)"
                    self._close_and_archive_position(
                        conn, r, cur_px, total_pnl, reason, now_iso, testnet_client, execute_on_testnet
                    )
                    events.append(f"⏱️ [{sym}] {reason} 완료 (청산가 ${cur_px:,.4f} | 확정 손익: ${total_pnl:+,.2f})")
                    continue

                # 5. 1st Partial Take-Profit (+1.8% TP1) + Move Stop to Breakeven
                tp1_triggered = (not tp1_hit) and (
                    (side_int == 1 and cur_px >= tp1) or (side_int == -1 and cur_px <= tp1)
                )
                if tp1_triggered:
                    closed_qty = init_qty * 0.50
                    new_rem_qty = init_qty - closed_qty
                    locked_profit = closed_qty * side_int * (tp1 - ep)
                    new_realized = realized + locked_profit
                    new_stop = ep
                    if testnet_client is not None:
                        testnet_client.move_stop_to_breakeven(sym, side_int, new_stop, execute_on_testnet)
                    conn.execute(
                        """
                        UPDATE active_positions
                        SET tp1_hit = 1, remaining_qty = ?, stop_price = ?, realized_pnl_usd = ?, peak_price = ?, updated_at = ?
                        WHERE symbol = ?
                        """,
                        (new_rem_qty, new_stop, new_realized, peak_px, now_iso, sym),
                    )
                    events.append(
                        f"🎯 [{sym}] +1.8% 1차 목표 도달! 50% 분할익절(+${locked_profit:,.2f} 확정) 및 잔여 50% 손절가 본절(${new_stop:,.4f}) 잠금 완료!"
                    )
                    rem_qty = new_rem_qty

                unreal = rem_qty * side_int * (cur_px - ep)
                conn.execute(
                    "UPDATE active_positions SET unrealized_pnl_usd = ?, peak_price = ?, updated_at = ? WHERE symbol = ?",
                    (unreal, peak_px, now_iso, sym),
                )

            conn.commit()
        return events

    def panic_close_all_positions(
        self,
        testnet_client: Any = None,
        latest_prices: Optional[Dict[str, float]] = None,
        execute_on_testnet: bool = False,
    ) -> List[str]:
        """
        Operator Emergency Remote Control (`panic-close`):
        Immediately closes all open positions at market (`reduceOnly=true`), cancels all open orders,
        and sets `trading_paused = True`.
        """
        set_operator_Pause_state(True)
        self.rules["trading_paused"] = True
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        events: List[str] = []
        prices = latest_prices or {}

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM active_positions WHERE status = 'OPEN'").fetchall()
            for r in rows:
                sym = r["symbol"]
                ep = float(r["entry_price"])
                cur_px = float(prices.get(sym, ep))
                side_int = int(r["side_int"])
                rem_qty = float(r["remaining_qty"])
                pnl_leg = rem_qty * side_int * (cur_px - ep)
                total_pnl = float(r["realized_pnl_usd"]) + pnl_leg
                self._close_and_archive_position(
                    conn,
                    r,
                    cur_px,
                    total_pnl,
                    "OPERATOR_PANIC_CLOSE_ALL (비상 전량 시장가 청산)",
                    now_iso,
                    testnet_client,
                    execute_on_testnet,
                )
                events.append(f"🚨 [PANIC CLOSE] {sym} 비상 시장가 전량 청산 완료 (확정 손익: ${total_pnl:+,.2f})")
            conn.commit()
        return events

    def can_open_position(self, symbol: str, dynamic_max_slots: Optional[int] = None) -> Tuple[bool, str]:
        """
        Pre-flight idempotency, operator pause, HWM floor lock, blacklist, & slot gate checked BEFORE submitting any order.
        """
        self.rules = load_user_trading_rules()
        if self.rules.get("trading_paused", False):
            return False, "OPERATOR_PAUSED_TRADING"
        if symbol in self.rules.get("excluded_symbols", []):
            return False, f"EXCLUDED_SYMBOL_BLACKLIST ({symbol})"
        if self.account_floor_locked:
            return False, "ACCOUNT_FLOOR_LOCKED_HWM_85PCT"
        if self.reconciliation_halted:
            return False, "FAIL_CLOSED_RECONCILIATION_HALTED"
        base_max_slots = int(self.rules["max_concurrent_slots"])
        eff_max_slots = (
            min(base_max_slots, max(1, int(dynamic_max_slots)))
            if dynamic_max_slots is not None
            else base_max_slots
        )
        with sqlite3.connect(self.db_path) as conn:
            already = conn.execute(
                "SELECT COUNT(*) FROM active_positions WHERE symbol = ? AND status = 'OPEN'", (symbol,)
            ).fetchone()[0]
            if already > 0:
                return False, f"SKIP_ALREADY_OPEN ({symbol})"
            cur_open = conn.execute("SELECT COUNT(*) FROM active_positions WHERE status = 'OPEN'").fetchone()[0]
            if cur_open >= eff_max_slots:
                if eff_max_slots < base_max_slots:
                    return False, f"SKIP_CORR_THROTTLED_SLOTS_FULL ({cur_open}/{eff_max_slots})"
                return False, f"SKIP_MAX_SLOTS_FULL ({cur_open}/{eff_max_slots})"
        return True, "OK"

    def open_or_record_position(
        self,
        symbol: str,
        side: str,
        side_int: int,
        sub_strategy: str,
        plan: Dict[str, Any],
        entry_price: float,
        execution_mode: str = "SHADOW_DRY_RUN",
        dynamic_max_slots: Optional[int] = None,
        opened_at_override: Optional[str] = None,
    ) -> bool:
        ok, _ = self.can_open_position(symbol, dynamic_max_slots=dynamic_max_slots)
        if not ok:
            return False
        now_iso = opened_at_override or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO active_positions (
                    opened_at, updated_at, symbol, side, side_int, sub_strategy,
                    margin_mode, slot_margin_usd, notional_usd, leverage,
                    initial_qty, remaining_qty, entry_price, stop_price, tp1_price,
                    tp1_hit, realized_pnl_usd, unrealized_pnl_usd, status, execution_mode, peak_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0.0, 0.0, 'OPEN', ?, ?)
                """,
                (
                    now_iso,
                    now_iso,
                    symbol,
                    side,
                    side_int,
                    sub_strategy,
                    self.rules["margin_mode"],
                    plan["slot_margin_usd"],
                    plan["notional_usd"],
                    plan["slot_leverage"],
                    plan["quantity"],
                    plan["quantity"],
                    entry_price,
                    plan["stop_price"],
                    plan["tp1_price"],
                    execution_mode,
                    entry_price,
                ),
            )
            conn.commit()
        return True

    def get_active_positions(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM active_positions WHERE status = 'OPEN' ORDER BY opened_at DESC").fetchall()
            return [dict(r) for r in rows]
