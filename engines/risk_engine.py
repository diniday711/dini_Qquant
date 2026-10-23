"""
W2 Deterministic Risk & Order Execution Control Engine (v2.1)
Enforces:
1. Per-trade risk cap [0.5%, 1.5%] with minimum stop distance floor (0.30%).
2. Unclipped Gap Loss Accounting & Audit Logging (`assert_gap_loss_logged`):
   When a price gap causes realized loss to exceed the planned risk cap, the full
   realized loss is applied to equity (never artificially clipped) and recorded in `gap_loss_events`.
3. Daily loss Kill-Switch (-2.0% of start-of-day equity).
4. Order-Layer Fail-Closed Execution Adapter (`MockOrderExecutionAdapter`):
   Handles timeout, partial fill, insufficient margin/leverage, and min-notional errors
   by rejecting new risk and cancelling all pending orders.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import numpy as np
from config import CONFIG, Phase1Config
from engines.cost_model import CostModel


@dataclass(frozen=True)
class OrderSizingDecision:
    approved: bool
    reason: str
    symbol: str
    side: int
    entry_price: float
    stop_price: float
    stop_distance_pct: float
    notional_usd: float
    quantity: float
    effective_leverage: float
    worst_case_loss_usd: float
    worst_case_loss_pct: float


@dataclass(frozen=True)
class GapLossEvent:
    utc_date: str
    symbol: str
    planned_max_loss_usd: float
    actual_realized_loss_usd: float
    gap_exceedance_usd: float
    equity_before: float
    equity_after: float


@dataclass(frozen=True)
class OrderExecutionResult:
    transmitted: bool
    status: str  # "FILLED", "PARTIAL_FILL_CONTAINED", "REJECTED_FAIL_CLOSED"
    reason: str
    filled_notional_usd: float
    open_orders_cancelled: bool


class DeterministicRiskEngine:
    def __init__(self, cfg: Phase1Config = CONFIG, cost_model: Optional[CostModel] = None):
        self.cfg = cfg
        self.cost_model = cost_model or CostModel(cfg)
        self.equity: float = float(cfg.initial_capital)
        self.current_day: Optional[str] = None
        self.day_start_equity: float = float(cfg.initial_capital)
        self.daily_realized_pnl: float = 0.0
        self.kill_switch_active: bool = False
        self.kill_switch_reason: str = ""
        self.open_symbols: Set[str] = set()
        self.gap_loss_events: List[GapLossEvent] = []

    @property
    def open_positions_count(self) -> int:
        return len(self.open_symbols)

    def on_new_bar_timestamp(self, utc_date_str: str) -> None:
        if self.current_day != utc_date_str:
            self.current_day = utc_date_str
            self.day_start_equity = self.equity
            self.daily_realized_pnl = 0.0
            self.kill_switch_active = False
            self.kill_switch_reason = ""

    @property
    def daily_return_pct(self) -> float:
        if self.day_start_equity <= 0:
            return -1.0
        return self.daily_realized_pnl / self.day_start_equity

    def record_trade_pnl(
        self,
        net_pnl_usd: float,
        utc_date_str: str,
        symbol: str = "UNKNOWN",
        planned_max_loss_usd: Optional[float] = None,
    ) -> None:
        """
        Apply the exact unclipped realized PnL (including any adverse gap slippage)
        to account equity, log a GapLossEvent if loss exceeded planned cap,
        and trip the daily Kill Switch if daily loss reaches max_daily_loss_pct.
        """
        self.on_new_bar_timestamp(utc_date_str)
        eq_before = self.equity
        # NEVER clip actual loss; clipping hides tail risk and distorts MDD!
        self.equity += float(net_pnl_usd)
        self.daily_realized_pnl += float(net_pnl_usd)

        actual_loss_usd = -float(net_pnl_usd) if net_pnl_usd < 0 else 0.0
        cap_usd = (
            float(planned_max_loss_usd)
            if planned_max_loss_usd is not None
            else (eq_before * self.cfg.max_risk_per_trade_cap)
        )
        if actual_loss_usd > cap_usd + 1e-6:
            self.gap_loss_events.append(
                GapLossEvent(
                    utc_date=utc_date_str,
                    symbol=symbol,
                    planned_max_loss_usd=cap_usd,
                    actual_realized_loss_usd=actual_loss_usd,
                    gap_exceedance_usd=actual_loss_usd - cap_usd,
                    equity_before=eq_before,
                    equity_after=self.equity,
                )
            )

        if self.daily_return_pct <= -self.cfg.max_daily_loss_pct + 1e-12:
            self.kill_switch_active = True
            self.kill_switch_reason = (
                f"KILL_SWITCH_DAILY_LOSS_BREACH: daily return {self.daily_return_pct:.4%} "
                f"<= -{self.cfg.max_daily_loss_pct:.4%}"
            )

    def size_and_validate_order(
        self,
        symbol: str,
        side: int,
        entry_price: float,
        stop_price: float,
        utc_date_str: str,
        requested_risk_pct: Optional[float] = None,
    ) -> OrderSizingDecision:
        self.on_new_bar_timestamp(utc_date_str)

        if self.kill_switch_active:
            return self._reject(symbol, side, entry_price, stop_price, 0.0, self.kill_switch_reason or "KILL_SWITCH_ACTIVE")

        if self.daily_return_pct <= -self.cfg.max_daily_loss_pct + 1e-12:
            self.kill_switch_active = True
            self.kill_switch_reason = "KILL_SWITCH_DAILY_LOSS_LIMIT"
            return self._reject(symbol, side, entry_price, stop_price, 0.0, self.kill_switch_reason)

        if symbol in self.open_symbols:
            return self._reject(symbol, side, entry_price, stop_price, 0.0, "REJECT_SYMBOL_ALREADY_OPEN")

        if self.open_positions_count >= self.cfg.max_concurrent_positions:
            return self._reject(symbol, side, entry_price, stop_price, 0.0, "REJECT_MAX_CONCURRENT_POSITIONS")

        if entry_price <= 0 or stop_price <= 0 or self.equity <= 0:
            return self._reject(symbol, side, entry_price, stop_price, 0.0, "REJECT_INVALID_PRICE_OR_EQUITY")

        if side == 1 and stop_price >= entry_price:
            return self._reject(symbol, side, entry_price, stop_price, 0.0, "REJECT_LONG_STOP_ABOVE_ENTRY")
        if side == -1 and stop_price <= entry_price:
            return self._reject(symbol, side, entry_price, stop_price, 0.0, "REJECT_SHORT_STOP_BELOW_ENTRY")

        raw_stop_dist_pct = abs(entry_price - stop_price) / entry_price
        # Enforce pre-registered minimum stop distance floor (0.30%)
        stop_distance_pct = max(raw_stop_dist_pct, self.cfg.min_stop_distance_pct)

        raw_risk_pct = (
            requested_risk_pct
            if requested_risk_pct is not None
            else self.cfg.risk_per_trade_pct
        )
        enforced_risk_pct = float(
            np.clip(
                raw_risk_pct,
                self.cfg.min_risk_per_trade_cap,
                self.cfg.max_risk_per_trade_cap,
            )
        )

        remaining_daily_loss_room_pct = max(
            0.0, self.cfg.max_daily_loss_pct + self.daily_return_pct
        )
        effective_risk_pct = min(enforced_risk_pct, remaining_daily_loss_room_pct)
        if effective_risk_pct <= 1e-6:
            return self._reject(symbol, side, entry_price, stop_price, stop_distance_pct, "REJECT_NO_REMAINING_DAILY_RISK_BUDGET")

        risk_budget_usd = self.equity * effective_risk_pct
        all_in_stop_loss_rate = (
            stop_distance_pct + self.cost_model.conservative_sizing_cost_rate
        )
        raw_notional_usd = risk_budget_usd / all_in_stop_loss_rate

        max_notional_usd = self.equity * self.cfg.max_gross_leverage
        final_notional_usd = float(np.floor(min(raw_notional_usd, max_notional_usd) * 1e6) / 1e6)

        if final_notional_usd < self.cfg.min_order_notional_usd:
            return self._reject(symbol, side, entry_price, stop_price, stop_distance_pct, "REJECT_BELOW_MIN_NOTIONAL")

        quantity = final_notional_usd / entry_price
        effective_leverage = final_notional_usd / self.equity
        worst_case_loss_usd = final_notional_usd * all_in_stop_loss_rate
        worst_case_loss_pct = worst_case_loss_usd / self.equity

        return OrderSizingDecision(
            approved=True,
            reason="APPROVED",
            symbol=symbol,
            side=side,
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            stop_distance_pct=float(stop_distance_pct),
            notional_usd=float(final_notional_usd),
            quantity=float(quantity),
            effective_leverage=float(effective_leverage),
            worst_case_loss_usd=float(worst_case_loss_usd),
            worst_case_loss_pct=float(worst_case_loss_pct),
        )

    @staticmethod
    def _reject(
        symbol: str,
        side: int,
        entry_price: float,
        stop_price: float,
        stop_dist: float,
        reason: str,
    ) -> OrderSizingDecision:
        return OrderSizingDecision(
            approved=False,
            reason=reason,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_pct=stop_dist,
            notional_usd=0.0,
            quantity=0.0,
            effective_leverage=0.0,
            worst_case_loss_usd=0.0,
            worst_case_loss_pct=0.0,
        )


class MockOrderExecutionAdapter:
    """
    Deterministic Order-Layer Fail-Closed Adapter (W2 4th Assert).
    Verifies that exchange/API faults (timeout, partial fill without stop ack,
    leverage/margin rejection, min lot size) trigger fail-closed containment:
    rejecting new risk and cancelling all pending open orders.
    """
    def __init__(self, risk_engine: DeterministicRiskEngine, min_qty: float = 0.001):
        self.risk_engine = risk_engine
        self.min_qty = min_qty
        self.pending_orders: Dict[str, float] = {}

    def submit_order_fail_closed(
        self,
        sizing: OrderSizingDecision,
        simulate_timeout: bool = False,
        simulate_partial_fill_ratio: float = 1.0,
        exchange_max_leverage_allowed: float = 5.0,
    ) -> OrderExecutionResult:
        if not sizing.approved:
            return OrderExecutionResult(
                transmitted=False,
                status="REJECTED_FAIL_CLOSED",
                reason=sizing.reason,
                filled_notional_usd=0.0,
                open_orders_cancelled=False,
            )

        # 1. Minimum Quantity / Notional check at exchange adapter boundary
        if (
            sizing.quantity < self.min_qty
            or sizing.notional_usd < self.risk_engine.cfg.min_order_notional_usd
        ):
            self.pending_orders.clear()
            return OrderExecutionResult(
                transmitted=False,
                status="REJECTED_FAIL_CLOSED",
                reason="FAIL_CLOSED_MIN_QTY_OR_NOTIONAL",
                filled_notional_usd=0.0,
                open_orders_cancelled=True,
            )

        # 2. Exchange Margin / Leverage mismatch check
        if sizing.effective_leverage > exchange_max_leverage_allowed + 1e-9:
            self.pending_orders.clear()
            return OrderExecutionResult(
                transmitted=False,
                status="REJECTED_FAIL_CLOSED",
                reason="FAIL_CLOSED_EXCHANGE_LEVERAGE_REJECT",
                filled_notional_usd=0.0,
                open_orders_cancelled=True,
            )

        # 3. API Timeout / Network Disconnect -> Fail-Closed & Cancel All
        if simulate_timeout:
            self.pending_orders.clear()
            self.risk_engine.kill_switch_active = True
            self.risk_engine.kill_switch_reason = "KILL_SWITCH_ORDER_TIMEOUT_FAIL_CLOSED"
            return OrderExecutionResult(
                transmitted=False,
                status="REJECTED_FAIL_CLOSED",
                reason="FAIL_CLOSED_API_TIMEOUT",
                filled_notional_usd=0.0,
                open_orders_cancelled=True,
            )

        # 4. Partial Fill Handling -> Cancel unfilled remainder immediately, bound exposure
        if simulate_partial_fill_ratio < 1.0:
            if simulate_partial_fill_ratio <= 0.0:
                self.pending_orders.clear()
                return OrderExecutionResult(
                    transmitted=False,
                    status="REJECTED_FAIL_CLOSED",
                    reason="FAIL_CLOSED_ZERO_FILL",
                    filled_notional_usd=0.0,
                    open_orders_cancelled=True,
                )
            filled_notional = sizing.notional_usd * float(simulate_partial_fill_ratio)
            self.pending_orders.clear()  # Cancel remainder
            return OrderExecutionResult(
                transmitted=True,
                status="PARTIAL_FILL_CONTAINED",
                reason="PARTIAL_FILL_REMAINDER_CANCELLED",
                filled_notional_usd=filled_notional,
                open_orders_cancelled=True,
            )

        return OrderExecutionResult(
            transmitted=True,
            status="FILLED",
            reason="OK",
            filled_notional_usd=sizing.notional_usd,
            open_orders_cancelled=False,
        )
