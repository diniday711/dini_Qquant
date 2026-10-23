"""
Pre-Registered Configuration for Q-Stack Phase 1 (v2.1 - Council Synthesis Complete)
All formerly undefined items from Round 2 (Codex, Claude, Opgo) are locked here.
Includes SHA-256 manifest hashing to mechanically detect any post-hoc parameter tampering.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import List


@dataclass(frozen=True)
class Phase1Config:
    # --- 1. Fixed Pre-Registered Universe (10 Liquid USDT-M Perpetuals + BTC Benchmark) ---
    timeframe: str = "15m"
    bars_per_day: int = 96
    benchmark_symbol: str = "BTCUSDT"
    target_universe: List[str] = field(default_factory=lambda: [
        "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
        "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT", "LTCUSDT"
    ])
    default_backfill_days: int = 180   # Initial 6-month backfill
    max_backfill_days: int = 730       # Period-only expansion ceiling (24m); universe expansion forbidden

    # --- 2. Pre-Registered Indicator Windows (Opgo Round 3 Fixed Values) ---
    beta_window: int = 96              # W_beta = 96 bars (24h) rolling OLS against BTCUSDT
    residual_window: int = 16          # 16 bars (4h) cumulative residual move
    residual_z_window: int = 192       # W_z = 192 bars (48h) residual standardization window
    residual_z_threshold: float = 2.5  # |z| >= 2.5 entry threshold
    swing_lookback: int = 20           # N = 20 bars (5h) prior swing high/low for Failed Breakout
    vwap_window: int = 96              # VWAP(96) (24h rolling VWAP) for TP1 target
    atr_window: int = 20               # ATR(20) for stop distance

    # --- 3. Deterministic Rule-Based Abstention Filter ---
    btc_momentum_window: int = 24      # 24 bars (6h) BTC return window
    btc_clash_return_limit: float = 0.010  # Abstain if |r_BTC(24)| >= 1.0% opposing trade direction
    min_reward_to_cost_mult: float = 3.0   # Expected distance to VWAP(96) must be >= 3x roundtrip cost

    # --- 4. Explicit Pre-Registered Exit Model (Shared by Strategy & Random Control) ---
    stop_atr_mult: float = 1.5         # Stop distance = ATR(20) * 1.5
    min_stop_distance_pct: float = 0.0030  # Floor stop distance = 0.30%
    tp1_fraction: float = 0.50         # Close 50% at VWAP(96) and move remaining stop to Breakeven
    tp2_residual_z_target: float = 0.50    # Close remaining 50% when |residual_z| reverts to <= 0.5
    max_holding_bars: int = 96         # Hard time-stop = 96 bars (24h)

    # --- 5. 2-Tier Realistic Transaction Cost Model ---
    roundtrip_fee_rate: float = 0.0010           # 0.10% roundtrip taker fee (5 bps per side)
    normal_roundtrip_slippage_rate: float = 0.0004   # 0.04% normal regime roundtrip slippage
    stressed_roundtrip_slippage_rate: float = 0.0015 # 0.15% high-volatility / sweep candle slippage
    high_vol_atr_mult_threshold: float = 1.5     # Trigger 0.15% slippage when bar range > 1.5 * ATR(20)

    # --- 6. W2 Deterministic Risk Engine & Order Layer Constraints ---
    initial_capital: float = 1000.0
    risk_per_trade_pct: float = 0.010       # 1.0% target risk per trade
    min_risk_per_trade_cap: float = 0.005   # 0.5% lower bound
    max_risk_per_trade_cap: float = 0.015   # 1.5% upper hard cap
    max_daily_loss_pct: float = 0.020       # -2.0% daily loss trips Kill Switch
    max_gross_leverage: float = 5.0         # Max 5.0x notional leverage
    max_concurrent_positions: int = 3       # Max 3 concurrent positions across universe
    max_positions_per_symbol: int = 1       # Max 1 position per coin
    min_order_notional_usd: float = 5.0     # Binance USDT-M minimum order notional ($5)

    # --- 7. W1 Edge Falsification Gates & Shared Random Control ---
    random_control_shared_execution: bool = True
    random_control_shuffles: int = 1000
    random_control_seed: int = 20261023
    w1_min_sample_size: int = 300
    w1_min_net_profit_to_cost: float = 1.5
    w1_min_oos_sharpe: float = 1.0
    w1_min_oos_trades_for_sharpe: int = 90
    w1_max_drawdown: float = 0.15
    w1_oos_split_ratio: float = 0.30
    w1_max_p_value: float = 0.05

    # --- 8. Paper -> Live Transition Gate (Corrected from OR to AND) ---
    paper_min_trades: int = 30              # >= 30 trades AND >= 8 weeks AND net profit > 0
    paper_min_weeks: int = 8
    paper_require_positive_net_pnl: bool = True
    paper_slippage_tolerance_pct: float = 0.30

    def config_hash(self) -> str:
        """Compute deterministic SHA-256 hash of the pre-registered configuration."""
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


CONFIG = Phase1Config()
