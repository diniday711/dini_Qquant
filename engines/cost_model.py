"""
Deterministic 2-Tier Transaction Cost & Funding Model (Phase 1 v2.1)
Accounts for:
1. Entry & Exit Taker Fees (0.10% roundtrip = 5 bps per side)
2. 2-Tier Execution Slippage:
   - Normal regime: 0.04% roundtrip
   - High-volatility / sweep candle regime (bar_range > 1.5 * ATR): 0.15% roundtrip
3. 8-Hour Funding Rate Payments (00:00, 08:00, 16:00 UTC)
"""
from dataclasses import dataclass
import numpy as np
from config import CONFIG, Phase1Config


@dataclass(frozen=True)
class TradeCostBreakdown:
    notional_usd: float
    is_high_vol_slippage: bool
    slippage_rate_applied: float
    fee_usd: float
    slippage_usd: float
    funding_usd: float
    total_cost_usd: float
    total_cost_pct_of_notional: float


class CostModel:
    def __init__(self, cfg: Phase1Config = CONFIG):
        if cfg.roundtrip_fee_rate < 0 or cfg.normal_roundtrip_slippage_rate < 0:
            raise ValueError("Fee and slippage rates must be non-negative.")
        self.cfg = cfg

    def get_roundtrip_slippage_rate(self, is_high_volatility_bar: bool = False) -> float:
        return (
            float(self.cfg.stressed_roundtrip_slippage_rate)
            if is_high_volatility_bar
            else float(self.cfg.normal_roundtrip_slippage_rate)
        )

    def get_roundtrip_cost_rate(self, is_high_volatility_bar: bool = False) -> float:
        return float(self.cfg.roundtrip_fee_rate) + self.get_roundtrip_slippage_rate(
            is_high_volatility_bar
        )

    @property
    def conservative_sizing_cost_rate(self) -> float:
        """
        Use stressed slippage (0.15%) + fee (0.10%) = 0.25% when sizing positions
        so that even during a sweep candle, normal stop-outs stay inside the risk cap.
        """
        return float(self.cfg.roundtrip_fee_rate + self.cfg.stressed_roundtrip_slippage_rate)

    def compute_funding_cost(
        self,
        side: int,
        notional_usd: float,
        funding_rates_crossed: np.ndarray | list[float],
    ) -> float:
        if notional_usd <= 0 or len(funding_rates_crossed) == 0:
            return 0.0
        rates = np.asarray(funding_rates_crossed, dtype=float)
        return float(side * notional_usd * np.sum(rates))

    def evaluate_trade_cost(
        self,
        side: int,
        notional_usd: float,
        funding_rates_crossed: np.ndarray | list[float] | None = None,
        is_high_volatility_bar: bool = False,
    ) -> TradeCostBreakdown:
        if notional_usd < 0:
            raise ValueError("notional_usd must be non-negative.")
        if side not in (1, -1):
            raise ValueError("side must be +1 (LONG) or -1 (SHORT).")

        slip_rate = self.get_roundtrip_slippage_rate(is_high_volatility_bar)
        fee_usd = notional_usd * self.cfg.roundtrip_fee_rate
        slippage_usd = notional_usd * slip_rate
        funding_usd = (
            self.compute_funding_cost(side, notional_usd, funding_rates_crossed)
            if funding_rates_crossed is not None
            else 0.0
        )
        total_cost_usd = fee_usd + slippage_usd + funding_usd
        total_cost_pct = (total_cost_usd / notional_usd) if notional_usd > 0 else 0.0

        return TradeCostBreakdown(
            notional_usd=float(notional_usd),
            is_high_vol_slippage=bool(is_high_volatility_bar),
            slippage_rate_applied=float(slip_rate),
            fee_usd=float(fee_usd),
            slippage_usd=float(slippage_usd),
            funding_usd=float(funding_usd),
            total_cost_usd=float(total_cost_usd),
            total_cost_pct_of_notional=float(total_cost_pct),
        )
