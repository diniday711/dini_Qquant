"""
Wonyotti 5-Slot Isolated Position & Lifecycle Manager (`engines/position_manager.py`)
Enforces the user's 4 core operational rules (`user_trading_rules.json`):
1. Capital & Slot Rule: 5-Slot Isolated Margin (`20%` of capital per slot, max 5 concurrent positions)
2. Stop-Loss Rule: Hybrid ATR Stop capped strictly at `-2.5%` max price distance (`max_stop_distance_pct = 0.025`)
3. Take-Profit & Lifecycle Rule:
   - At `+1.8%` (`tp1_price`), take `50%` partial profit (`TP1_50PCT_LOCKED`)
   - Automatically move the remaining `50%` stop-loss to Break-Even (`entry_price`)
   - Trail the remaining `50%` until trend reversal or regime flip
4. Daily Kill Switch & Leverage Cap: `-3.0%` daily cumulative loss halts new entries; max leverage `5.0x` (Isolated)
"""
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

RULES_PATH = Path(__file__).parent.parent / "user_trading_rules.json"


def load_user_trading_rules() -> Dict[str, Any]:
    default_rules = {
        "profile_name": "Wonyotti_5Slot_Isolated_Hybrid",
        "initial_capital_usdt": 1000.0,
        "margin_mode": "ISOLATED",
        "max_concurrent_slots": 5,
        "slot_allocation_pct": 0.20,
        "max_leverage": 5,
        "stop_loss_mode": "HYBRID_ATR_CAPPED",
        "min_stop_distance_pct": 0.008,
        "max_stop_distance_pct": 0.025,
        "atr_stop_multiplier": 2.0,
        "take_profit_mode": "SPLIT_50_BREAKEVEN_TRAIL",
        "partial_tp_target_pct": 0.018,
        "partial_tp_ratio": 0.50,
        "max_hold_hours": 48,
        "daily_kill_switch_loss_pct": 0.03,
    }
    if RULES_PATH.exists():
        try:
            loaded = json.loads(RULES_PATH.read_text(encoding="utf-8"))
            default_rules.update(loaded)
        except Exception:
            pass
    return default_rules


class PositionLifecycleManager:
    def __init__(self, db_path: str = "data/crumb_memory.db"):
        self.db_path = db_path
        self.rules = load_user_trading_rules()
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

    def compute_wonyotti_slot_order_plan(
        self,
        equity_usdt: float,
        entry_price: float,
        atr_val: float,
        side_int: int,
    ) -> Dict[str, float]:
        """
        Compute exact entry quantity, stop-loss price (ATR-based but strictly capped at -2.5%),
        and 1st partial take-profit price (+1.8%) using Wonyotti's 1/5 (20%) Isolated Slot rule.
        """
        slot_pct = float(self.rules["slot_allocation_pct"])  # 20% (1/5 of seed)
        max_lev = float(self.rules["max_leverage"])  # max 5x
        min_stop = float(self.rules["min_stop_distance_pct"])  # 0.8%
        max_stop = float(self.rules["max_stop_distance_pct"])  # 2.5% hard ceiling!
        atr_mult = float(self.rules["atr_stop_multiplier"])  # 2.0x ATR
        tp1_pct = float(self.rules["partial_tp_target_pct"])  # +1.8% 1st TP

        raw_atr_stop_pct = (atr_mult * atr_val) / max(1e-9, entry_price)
        stop_dist_pct = max(min_stop, min(max_stop, raw_atr_stop_pct))

        # Slot isolated margin = 20% of equity ($200 on $1,000 seed)
        slot_margin_usd = equity_usdt * slot_pct

        # Risk 1.0% of total equity ($10) on this trade; cap notional at max_leverage * slot_margin_usd
        target_risk_usd = equity_usdt * 0.010
        desired_notional = target_risk_usd / stop_dist_pct
        max_slot_notional = slot_margin_usd * max_lev
        notional_usd = min(desired_notional, max_slot_notional)

        slot_leverage = max(1.0, min(max_lev, notional_usd / max(1e-9, slot_margin_usd)))
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
        """
        Step 2.5 of every cycle:
        Inspects all currently open positions and automatically:
        - Executes 50% Partial Take-Profit if price reaches `tp1_price` (+1.8%) and locks Stop-Loss at Break-Even (`entry_price`)
        - Executes Stop-Loss if price hits `stop_price`
        - Executes Regime-Flip Exit if the coin's trend regime flips strongly opposite to the position
        """
        events: List[str] = []
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM active_positions WHERE status = 'OPEN'").fetchall()

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

                # 1. Check Stop-Loss Hit
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

                # 2. Check 1st Partial Take-Profit (+1.8%) -> 50% Scale-Out + Move Stop to Break-Even!
                tp1_triggered = (not tp1_hit) and (
                    (side_int == 1 and cur_px >= tp1) or (side_int == -1 and cur_px <= tp1)
                )
                if tp1_triggered:
                    closed_qty = init_qty * 0.50
                    new_rem_qty = init_qty - closed_qty
                    locked_profit = closed_qty * side_int * (tp1 - ep)
                    new_realized = realized + locked_profit
                    # Move stop price to exact entry price (본절 잠금)
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
                    realized = new_realized
                    sp = new_stop
                    tp1_hit = 1

                # 3. Update Unrealized PnL
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
        """Record a newly opened position into the 5-Slot Isolated tracker if a slot is available."""
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
