"""
Unified W2 Isolated Position & Lifecycle Manager (`engines/position_manager.py`)
Implements Round 3 Committee Synthesis:
1. Unified Config with `CONFIG` (`config.py`):
   - `max_concurrent_slots = 3` (matches `CONFIG.max_concurrent_positions = 3`)
   - `daily_kill_switch_loss_pct = 0.020` (matches `CONFIG.max_daily_loss_pct = -2.0%`)
   - `risk_per_trade_pct = 0.010` (strictly `1.0%` max loss per trade even at `-2.5%` stop ceiling)
   - `max_gross_leverage = 1.0` (`<= 0.333x` notional per slot so 3 slots never exceed `1.0x` gross equity)
2. Exchange-First Reconciliation (`reconcile_exchange_vs_db`):
   - Treats Binance `positionRisk` + `openOrders` as the single Source of Truth, and SQLite as a local cache.
   - Any mismatch between Exchange and SQLite or any position missing a `STOP_MARKET` order immediately trips `reconciliation_halted = True` (`Fail-Closed`).
"""
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List
from config import CONFIG

RULES_PATH = Path(__file__).parent.parent / "user_trading_rules.json"


def load_user_trading_rules() -> Dict[str, Any]:
    default_rules = {
        "profile_name": "W2_Unified_Isolated_Hybrid_v3_1",
        "initial_capital_usdt": CONFIG.initial_capital,
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
        "max_hold_hours": 48,
        "daily_kill_switch_loss_pct": CONFIG.max_daily_loss_pct,  # -2.0% daily kill switch (Unified with W2)
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


class PositionLifecycleManager:
    def __init__(self, db_path: str = "data/crumb_memory.db"):
        self.db_path = db_path
        self.rules = load_user_trading_rules()
        self.reconciliation_halted: bool = False
        self.reconciliation_errors: List[str] = []
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
                    status TEXT NOT NULL DEFAULT 'OPEN'
                )
                """
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
            conn.commit()

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
    ) -> Dict[str, float]:
        """
        Compute exact entry quantity, stop-loss price (ATR-based capped at -2.5%),
        and 1st partial take-profit price (+1.8%) with zero geometry conflict:
        - Per-trade max loss is strictly `<= 1.0%` of equity (`$10` on `$1,000`)
        - Per-slot notional is capped at `equity / max_concurrent_slots` (`33.3%` of equity)
          so 3 concurrent slots never exceed `1.0x` gross leverage!
        """
        slot_pct = float(self.rules["slot_allocation_pct"])          # 20% ($200 isolated margin)
        max_slots = int(self.rules["max_concurrent_slots"])          # 3 slots
        max_gross_lev = float(self.rules.get("max_gross_leverage", 1.0))
        min_stop = float(self.rules["min_stop_distance_pct"])        # 0.8%
        max_stop = float(self.rules["max_stop_distance_pct"])        # 2.5% hard ceiling!
        atr_mult = float(self.rules["atr_stop_multiplier"])          # 2.0x ATR
        tp1_pct = float(self.rules["partial_tp_target_pct"])         # +1.8% 1st TP
        risk_pct = float(self.rules["risk_per_trade_pct"])           # 1.0% risk per trade

        raw_atr_stop_pct = (atr_mult * atr_val) / max(1e-9, entry_price)
        stop_dist_pct = max(min_stop, min(max_stop, raw_atr_stop_pct))

        slot_margin_usd = equity_usdt * slot_pct
        target_risk_usd = equity_usdt * risk_pct

        # Notional from 1% risk vs per-slot gross leverage ceiling (equity * 1.0 / 3 = $333.33)
        desired_notional = target_risk_usd / stop_dist_pct
        max_slot_notional = (equity_usdt * max_gross_lev) / max(1, max_slots)
        notional_usd = min(desired_notional, max_slot_notional)

        slot_leverage = max(1.0, notional_usd / max(1e-9, slot_margin_usd))
        qty = notional_usd / max(1e-9, entry_price)

        if side_int == 1:
            stop_px = entry_price * (1.0 - stop_dist_pct)
            tp1_px = entry_price * (1.0 + tp1_pct)
        else:
            stop_px = entry_price * (1.0 + stop_dist_pct)
            tp1_px = entry_price * (1.0 - tp1_pct)

        max_loss_usd = notional_usd * stop_dist_pct

        return {
            "slot_margin_usd": slot_margin_usd,
            "notional_usd": notional_usd,
            "slot_leverage": slot_leverage,
            "quantity": qty,
            "partial_tp_qty": qty * float(self.rules["partial_tp_ratio"]),
            "stop_dist_pct": stop_dist_pct,
            "stop_price": stop_px,
            "tp1_pct": tp1_pct,
            "tp1_price": tp1_px,
            "max_loss_usd": max_loss_usd,
        }

    def sync_and_manage_positions(
        self,
        latest_prices: Dict[str, float],
        coin_regimes_map: Dict[str, str],
        testnet_client: Any = None,
        execute_on_testnet: bool = False,
    ) -> List[str]:
        events: List[str] = []
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        if testnet_client is not None:
            ex_state = testnet_client.fetch_exchange_positions_and_orders()
            rec = self.reconcile_exchange_vs_db(ex_state)
            if not rec["ok"]:
                events.append(f"🚨 [FAIL-CLOSED 거래소 대사 불일치] {'; '.join(rec['errors'])} -> 신규 주문 차단!")
                return events

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM active_positions WHERE status = 'OPEN'").fetchall()
            # Enforce W2 max_concurrent_slots invariant if legacy rows > max_slots
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

                stop_triggered = (side_int == 1 and cur_px <= sp) or (side_int == -1 and cur_px >= sp)
                if stop_triggered:
                    pnl_leg = rem_qty * side_int * (cur_px - ep)
                    total_pnl = realized + pnl_leg
                    reason = "BREAKEVEN_STOP_EXIT (본절 탈출)" if tp1_hit else "HARD_STOP_LOSS (-2.5% 이내 칼손절)"
                    conn.execute("DELETE FROM active_positions WHERE symbol = ?", (sym,))
                    conn.execute(
                        """
                        INSERT INTO closed_positions_history (
                            opened_at, closed_at, symbol, side, sub_strategy,
                            entry_price, exit_price, notional_usd, tp1_hit, net_pnl_usd, close_reason
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            r["opened_at"],
                            now_iso,
                            sym,
                            r["side"],
                            r["sub_strategy"],
                            ep,
                            cur_px,
                            r["notional_usd"],
                            tp1_hit,
                            total_pnl,
                            reason,
                        ),
                    )
                    events.append(f"🛑 [{sym}] {reason} 완료 (청산가 ${cur_px:,.4f} | 확정 손익: ${total_pnl:+,.2f})")
                    continue

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
                        SET tp1_hit = 1, remaining_qty = ?, stop_price = ?, realized_pnl_usd = ?, updated_at = ?
                        WHERE symbol = ?
                        """,
                        (new_rem_qty, new_stop, new_realized, now_iso, sym),
                    )
                    events.append(
                        f"🎯 [{sym}] +1.8% 1차 목표 도달! 50% 분할익절(+${locked_profit:,.2f} 확정) 및 잔여 50% 손절가 본절(${new_stop:,.4f}) 잠금 완료!"
                    )
                    rem_qty = new_rem_qty

                unreal = rem_qty * side_int * (cur_px - ep)
                conn.execute(
                    "UPDATE active_positions SET unrealized_pnl_usd = ?, updated_at = ? WHERE symbol = ?",
                    (unreal, now_iso, sym),
                )

            conn.commit()
        return events

    def open_or_record_position(
        self,
        symbol: str,
        side: str,
        side_int: int,
        sub_strategy: str,
        plan: Dict[str, float],
        entry_price: float,
    ) -> bool:
        if self.reconciliation_halted:
            return False
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        max_slots = int(self.rules["max_concurrent_slots"])
        with sqlite3.connect(self.db_path) as conn:
            cur_open = conn.execute("SELECT COUNT(*) FROM active_positions WHERE status = 'OPEN'").fetchone()[0]
            already = conn.execute(
                "SELECT COUNT(*) FROM active_positions WHERE symbol = ? AND status = 'OPEN'", (symbol,)
            ).fetchone()[0]
            if already > 0 or cur_open >= max_slots:
                return False

            conn.execute(
                """
                INSERT INTO active_positions (
                    opened_at, updated_at, symbol, side, side_int, sub_strategy,
                    margin_mode, slot_margin_usd, notional_usd, leverage,
                    initial_qty, remaining_qty, entry_price, stop_price, tp1_price,
                    tp1_hit, realized_pnl_usd, unrealized_pnl_usd, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0.0, 0.0, 'OPEN')
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
                ),
            )
            conn.commit()
        return True

    def get_active_positions(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM active_positions WHERE status = 'OPEN' ORDER BY opened_at DESC").fetchall()
            return [dict(r) for r in rows]
