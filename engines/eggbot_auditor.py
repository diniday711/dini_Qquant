"""
04. DR. EGGBOT — Bear-Case & Pre-Trade Contradiction Auditor (`engines/eggbot_auditor.py`)
Implements `<bear_case>` and `<final_check>` from `audit_bot.xml` + `Lauren Tan's pstack`:
- Actively challenges every regime call and trade candidate before capital is committed.
- Checks:
  1. Breadth Divergence: Is the coin rallying while overall market breadth (<30%) is collapsing?
  2. Funding Rate Crowding: Is funding rate extreme in the same direction (crowded positioning)?
  3. Volatility Tail Risk: Is `vol_pct_rank > 0.92` (Extreme volatility tail)?
  4. BTC Momentum Clash: Is BTC moving strongly opposite to the candidate trade?
  5. Negative Knowledge (Crumb Check): Did this sub-strategy fail historical cost gates?
- Outputs a deterministic `APPROVED` or `ABSTAIN` verdict + structured `"WHAT COULD I BE WRONG ABOUT?"` notes.
"""
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class EggbotAuditVerdict:
    approved: bool
    verdict_code: str  # "APPROVED", "ABSTAIN_BEAR_CASE_BLOCK"
    confidence_score: int
    bull_thesis: str
    bear_case_challenges: List[str]
    what_could_i_be_wrong_about: str


class DrEggbotAuditor:
    def audit_candidate_signal(
        self,
        symbol: str,
        side: int,  # +1 LONG, -1 SHORT
        sub_strategy: str,  # "TREND" or "MEAN_REV"
        adx_14: float,
        vol_pct_rank: float,
        market_breadth_pct: float,
        btc_ret_24: float,
        delta_ratio: float,
        latest_funding_rate: float,
        expected_reward_to_cost: float,
    ) -> EggbotAuditVerdict:
        challenges: List[str] = []
        blocking_errors: List[str] = []

        direction_str = "LONG" if side == 1 else "SHORT"
        bull_thesis = (
            f"{symbol} {direction_str} via {sub_strategy} "
            f"(ADX={adx_14:.1f}, Breadth={market_breadth_pct:.0%}, "
            f"DeltaRatio={delta_ratio:+.2f}, Reward/Cost={expected_reward_to_cost:.1f}x)"
        )

        # 1. Volatility Tail Handbrake Check
        if vol_pct_rank > 0.92:
            blocking_errors.append(
                f"EXTREME_VOLATILITY_TAIL: Volatility rank is {vol_pct_rank:.1%} (>92% ceiling). Handbrake active."
            )

        # 2. Breadth Divergence Check (`audit_bot.xml`: "indices rising while breadth falls")
        if side == 1 and sub_strategy == "TREND" and market_breadth_pct < 0.40:
            blocking_errors.append(
                f"BREADTH_DIVERGENCE: Attempting LONG breakout while only {market_breadth_pct:.0%} of universe is above 200-MA."
            )
        elif side == -1 and sub_strategy == "TREND" and market_breadth_pct > 0.60:
            blocking_errors.append(
                f"BREADTH_DIVERGENCE: Attempting SHORT breakdown while {market_breadth_pct:.0%} of universe is above 200-MA."
            )

        # 3. Funding Rate Crowding Check
        if side == 1 and latest_funding_rate > 0.0005:  # > +0.05% per 8h is heavily crowded long
            challenges.append(
                f"CROWDED_LONG_FUNDING: 8h funding is high ({latest_funding_rate:.4%}); vulnerable to long squeeze."
            )
            if latest_funding_rate > 0.0010:
                blocking_errors.append("EXTREME_LONG_FUNDING_CROWDING")
        elif side == -1 and latest_funding_rate < -0.0005:
            challenges.append(
                f"CROWDED_SHORT_FUNDING: 8h funding is deeply negative ({latest_funding_rate:.4%}); vulnerable to short squeeze."
            )
            if latest_funding_rate < -0.0010:
                blocking_errors.append("EXTREME_SHORT_FUNDING_CROWDING")

        # 4. BTC Directional Clash Check
        if side == 1 and btc_ret_24 <= -0.015:
            blocking_errors.append(
                f"BTC_MACRO_HEADWIND: BTC 24h return is {btc_ret_24:.2%} against LONG intent."
            )
        elif side == -1 and btc_ret_24 >= 0.015:
            blocking_errors.append(
                f"BTC_MACRO_HEADWIND: BTC 24h return is {btc_ret_24:+.2%} against SHORT intent."
            )

        # 5. Cost Erosion Check
        if expected_reward_to_cost < 3.0:
            blocking_errors.append(
                f"COST_EROSION_RISK: Expected reward/cost ({expected_reward_to_cost:.2f}x) is below 3.0x minimum."
            )

        # 6. Always formulate "WHAT COULD I BE WRONG ABOUT?" (`audit_bot.xml` <final_check>)
        if sub_strategy == "TREND":
            wrong_about = (
                "If this breakout is a late-stage exhaustion wick driven by last-minute retail FOMO "
                "or an upcoming macro release (FOMC/CPI), price could snap back inside the 48h range and hit stop."
            )
        else:
            wrong_about = (
                "If the current ranging market is transitioning into a high-momentum liquidation cascade, "
                "fading the band edge will suffer adverse selection."
            )

        all_notes = blocking_errors + challenges
        if not all_notes:
            all_notes.append("No fatal cross-asset or funding contradictions detected; standard stop-loss applies.")

        approved = len(blocking_errors) == 0
        confidence = max(10, min(95, int(75 - 25 * len(blocking_errors) - 10 * len(challenges))))

        return EggbotAuditVerdict(
            approved=approved,
            verdict_code="APPROVED" if approved else "ABSTAIN_BEAR_CASE_BLOCK",
            confidence_score=confidence,
            bull_thesis=bull_thesis,
            bear_case_challenges=all_notes,
            what_could_i_be_wrong_about=wrong_about,
        )
