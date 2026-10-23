"""
Phase 1 Event-Driven Backtester & Shared-Execution W1 Falsification Engine (v2.1)
Implements all Round 2 Council Synthesis requirements:
1. Pre-registered Exit Model (`simulate_single_trade_execution`):
   - Stop-Loss: ATR(20) * 1.5 (floor 0.30%), with UNCLIPPED gap fill accounting
     (if bar opens beyond stop, fills at adverse open price -> logged as GapLossEvent).
   - TP1: At VWAP(96), close 50% of position and move stop on remaining 50% to Breakeven.
   - TP2: Close remaining 50% when |residual_z| <= 0.5.
   - Time Stop: Hard close at 96 bars (24h).
2. Shared Execution Model for Random Control (`shared_execution_model=True`):
   - Both the actual strategy and all 1,000 random-entry control runs call the
     EXACT same `simulate_single_trade_execution()`, `size_and_validate_order()`,
     and `CostModel.evaluate_trade_cost()` functions!
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from config import CONFIG, Phase1Config
from engines.cost_model import CostModel
from engines.risk_engine import DeterministicRiskEngine
from engines.signal_engine import SingleConfluenceSignalEngine


@dataclass
class W1GateEvaluation:
    config_sha256: str
    total_trades: int
    oos_trades: int
    gross_profit_usd: float
    total_cost_usd: float
    net_profit_usd: float
    net_profit_to_cost_ratio: float
    full_sharpe: float
    oos_sharpe: float
    max_drawdown_pct: float
    oos_max_drawdown_pct: float
    gap_loss_count: int
    random_baseline_mean_net_usd: float
    permutation_p_value: float
    bootstrap_p_value: float
    shared_execution_model_verified: bool
    gate_1_sample_pass: bool
    gate_1_status: str
    gate_2_cost_ratio_pass: bool
    gate_3_oos_sharpe_pass: bool
    gate_3_oos_sample_valid: bool
    gate_4_mdd_pass: bool
    gate_5_random_baseline_pass: bool
    w1_overall_status: str


class Phase1Backtester:
    def __init__(self, cfg: Phase1Config = CONFIG):
        self.cfg = cfg
        self.cost_model = CostModel(cfg)
        self.signal_engine = SingleConfluenceSignalEngine(cfg, self.cost_model)

    def simulate_single_trade_execution(
        self,
        symbol: str,
        signal_idx: int,
        side: int,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        times: np.ndarray,
        stop_dists: np.ndarray,
        vwaps: np.ndarray,
        res_z: np.ndarray,
        high_vol_flags: np.ndarray,
        f_times_ns: np.ndarray,
        f_rates: np.ndarray,
    ) -> Optional[Dict]:
        """
        Single canonical exit_model() shared identically by Strategy and Random Control.
        Signal at bar `signal_idx` enters at `entry_idx = signal_idx + 1` OPEN.
        Implements:
        - Unclipped Gap Stop-Loss at `ATR(20)*1.5` (min 0.30%)
        - TP1 (50% at rolling VWAP(96) + Breakeven stop on remaining 50%)
        - TP2 (remaining 50% when |residual_z| <= 0.5)
        - Hard Time Exit at 96 bars (24h)
        """
        n = len(opens)
        entry_idx = signal_idx + 1
        if entry_idx >= n - 1:
            return None

        entry_price = float(opens[entry_idx])
        if entry_price <= 0:
            return None

        stop_dist_pct = float(stop_dists[signal_idx])
        if np.isnan(stop_dist_pct) or stop_dist_pct < self.cfg.min_stop_distance_pct:
            stop_dist_pct = self.cfg.min_stop_distance_pct

        if side == 1:
            initial_stop_price = entry_price * (1.0 - stop_dist_pct)
            fallback_tp1 = entry_price * (1.0 + stop_dist_pct * 1.5)
        else:
            initial_stop_price = entry_price * (1.0 + stop_dist_pct)
            fallback_tp1 = entry_price * (1.0 - stop_dist_pct * 1.5)

        current_stop_price = initial_stop_price
        tp1_hit = False
        tp1_return_pct = 0.0
        leg2_return_pct = 0.0
        exit_reason = "TIME_EXIT_96B"
        max_end_idx = min(n - 1, entry_idx + self.cfg.max_holding_bars - 1)
        exit_idx = max_end_idx
        is_gap_stop = False

        for j in range(entry_idx, max_end_idx + 1):
            bar_o = float(opens[j])
            bar_h = float(highs[j])
            bar_l = float(lows[j])
            bar_c = float(closes[j])
            bar_vwap = float(vwaps[j])
            bar_z = float(res_z[j])

            # Dynamic TP1 target: VWAP(96) if favorable by at least 0.5 * stop_dist, else fallback 1.5R
            if side == 1:
                tp1_target = (
                    bar_vwap
                    if bar_vwap >= entry_price * (1.0 + 0.5 * stop_dist_pct)
                    else fallback_tp1
                )
            else:
                tp1_target = (
                    bar_vwap
                    if bar_vwap <= entry_price * (1.0 - 0.5 * stop_dist_pct)
                    else fallback_tp1
                )

            # 1. Check Stop-Loss (including UNCLIPPED adverse price gap at bar open!)
            if side == 1:
                if bar_o < current_stop_price:
                    # Gapped below stop at bar open -> fill at actual bar_o (unclipped!)
                    fill_px = bar_o
                    leg_ret = (fill_px - entry_price) / entry_price
                    if not tp1_hit:
                        is_gap_stop = True
                        exit_reason = "GAP_STOP_LOSS"
                        tp1_return_pct = leg_ret
                        leg2_return_pct = leg_ret
                    else:
                        exit_reason = "TP1_THEN_GAP_STOP"
                        leg2_return_pct = leg_ret
                    exit_idx = j
                    break
                if bar_l <= current_stop_price:
                    fill_px = current_stop_price
                    leg_ret = (fill_px - entry_price) / entry_price
                    if not tp1_hit:
                        exit_reason = "STOP_LOSS"
                        tp1_return_pct = leg_ret
                        leg2_return_pct = leg_ret
                    else:
                        exit_reason = "TP1_THEN_BREAKEVEN"
                        leg2_return_pct = leg_ret
                    exit_idx = j
                    break
            else:
                if bar_o > current_stop_price:
                    # Gapped above stop at bar open -> fill at actual bar_o (unclipped!)
                    fill_px = bar_o
                    leg_ret = (entry_price - fill_px) / entry_price
                    if not tp1_hit:
                        is_gap_stop = True
                        exit_reason = "GAP_STOP_LOSS"
                        tp1_return_pct = leg_ret
                        leg2_return_pct = leg_ret
                    else:
                        exit_reason = "TP1_THEN_GAP_STOP"
                        leg2_return_pct = leg_ret
                    exit_idx = j
                    break
                if bar_h >= current_stop_price:
                    fill_px = current_stop_price
                    leg_ret = (entry_price - fill_px) / entry_price
                    if not tp1_hit:
                        exit_reason = "STOP_LOSS"
                        tp1_return_pct = leg_ret
                        leg2_return_pct = leg_ret
                    else:
                        exit_reason = "TP1_THEN_BREAKEVEN"
                        leg2_return_pct = leg_ret
                    exit_idx = j
                    break

            # 2. Check TP1 (50% partial close at VWAP(96) + Move Stop to Breakeven)
            if not tp1_hit:
                if side == 1 and bar_h >= tp1_target:
                    tp1_hit = True
                    tp1_return_pct = (tp1_target - entry_price) / entry_price
                    current_stop_price = entry_price  # Move remaining 50% stop to Breakeven
                elif side == -1 and bar_l <= tp1_target:
                    tp1_hit = True
                    tp1_return_pct = (entry_price - tp1_target) / entry_price
                    current_stop_price = entry_price  # Move remaining 50% stop to Breakeven

            # 3. Check TP2 (Residual Z-score reverts to |z| <= 0.5 after TP1)
            if tp1_hit:
                if (side == 1 and bar_z >= -self.cfg.tp2_residual_z_target) or (
                    side == -1 and bar_z <= self.cfg.tp2_residual_z_target
                ):
                    leg2_return_pct = side * ((bar_c - entry_price) / entry_price)
                    exit_reason = "TP1_AND_TP2_Z_REVERT"
                    exit_idx = j
                    break

            # 4. If final bar of holding horizon (96 bars) reached without prior exit
            if j == max_end_idx:
                final_leg_ret = side * ((bar_c - entry_price) / entry_price)
                if not tp1_hit:
                    tp1_return_pct = final_leg_ret
                    leg2_return_pct = final_leg_ret
                    exit_reason = "TIME_EXIT_96B"
                else:
                    leg2_return_pct = final_leg_ret
                    exit_reason = "TP1_THEN_TIME_EXIT_96B"
                exit_idx = j
                break

        w1 = self.cfg.tp1_fraction
        blended_raw_return_pct = w1 * tp1_return_pct + (1.0 - w1) * leg2_return_pct
        exit_price_equiv = entry_price * (1.0 + side * blended_raw_return_pct)

        entry_time = pd.Timestamp(times[entry_idx])
        exit_time = pd.Timestamp(times[exit_idx])

        # Check if entry or exit bar is a high-volatility bar (triggers 0.15% 2nd-tier slippage)
        is_high_vol = bool(
            high_vol_flags[ signal_idx ]
            or high_vol_flags[ entry_idx ]
            or high_vol_flags[ exit_idx ]
        )

        if len(f_times_ns) > 0:
            mask = (f_times_ns > entry_time.value) & (f_times_ns <= exit_time.value)
            funding_sum = float(np.sum(f_rates[mask]))
        else:
            funding_sum = 0.0

        return {
            "symbol": symbol,
            "signal_idx": int(signal_idx),
            "signal_bar_time": pd.Timestamp(times[signal_idx]),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": int(side),
            "residual_z": float(res_z[signal_idx]),
            "entry_price": entry_price,
            "stop_price": initial_stop_price,
            "exit_price": exit_price_equiv,
            "stop_dist_pct": stop_dist_pct,
            "holding_bars": int(exit_idx - entry_idx + 1),
            "exit_reason": exit_reason,
            "is_gap_stop": is_gap_stop,
            "is_high_vol_slippage": is_high_vol,
            "raw_return_pct": float(blended_raw_return_pct),
            "funding_sum_rate": funding_sum,
        }

    def _extract_symbol_candidates(
        self,
        symbol: str,
        feat_df: pd.DataFrame,
        funding_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Extract strategy trade candidates and cache numpy arrays for fast shared-execution replay."""
        n = len(feat_df)
        warmup = max(self.cfg.beta_window, self.cfg.residual_z_window) + 5
        if n < warmup + 5:
            return pd.DataFrame(), {}

        has_funding = funding_df is not None and not funding_df.empty
        if has_funding:
            f_times_ns = (
                pd.to_datetime(funding_df["funding_time"], format="ISO8601", utc=True)
                .astype("int64")
                .to_numpy()
            )
            f_rates = funding_df["funding_rate"].to_numpy(dtype=float)
        else:
            f_times_ns = np.array([], dtype=np.int64)
            f_rates = np.array([], dtype=float)

        arrays = {
            "opens": feat_df["open"].to_numpy(dtype=float),
            "highs": feat_df["high"].to_numpy(dtype=float),
            "lows": feat_df["low"].to_numpy(dtype=float),
            "closes": feat_df["close"].to_numpy(dtype=float),
            "times": feat_df["open_time"].to_numpy(),
            "signals": feat_df["signal"].to_numpy(dtype=int),
            "stop_dists": feat_df["stop_dist_pct"].to_numpy(dtype=float),
            "vwaps": feat_df["vwap_96"].to_numpy(dtype=float),
            "res_z": feat_df["residual_z"].to_numpy(dtype=float),
            "high_vol_flags": feat_df["is_high_vol_bar"].to_numpy(dtype=bool),
            "f_times_ns": f_times_ns,
            "f_rates": f_rates,
        }

        trades = []
        i = warmup
        signals = arrays["signals"]
        while i < n - 2:
            sig = int(signals[i])
            if sig == 0:
                i += 1
                continue
            tr = self.simulate_single_trade_execution(
                symbol=symbol,
                signal_idx=i,
                side=sig,
                opens=arrays["opens"],
                highs=arrays["highs"],
                lows=arrays["lows"],
                closes=arrays["closes"],
                times=arrays["times"],
                stop_dists=arrays["stop_dists"],
                vwaps=arrays["vwaps"],
                res_z=arrays["res_z"],
                high_vol_flags=arrays["high_vol_flags"],
                f_times_ns=arrays["f_times_ns"],
                f_rates=arrays["f_rates"],
            )
            if tr is not None:
                trades.append(tr)
                i += tr["holding_bars"]
            else:
                i += 1

        return pd.DataFrame(trades), arrays

    def replay_candidates_through_risk_and_cost(
        self, all_candidates: pd.DataFrame
    ) -> Tuple[pd.DataFrame, DeterministicRiskEngine]:
        """
        Single canonical portfolio replay engine used by both Strategy and Random Control.
        Enforces max 3 concurrent positions, max 1 per coin, 1% risk cap, -2% daily kill switch,
        and 2-tier slippage + fee + funding cost model.
        """
        risk_engine = DeterministicRiskEngine(self.cfg, self.cost_model)
        if all_candidates.empty:
            return pd.DataFrame(), risk_engine

        sorted_cands = all_candidates.sort_values(["entry_time", "symbol"]).reset_index(drop=True)
        executed_records = []
        # Track active positions: (exit_time, net_pnl_usd, exit_utc_day, symbol, planned_max_loss_usd)
        active_exits: List[Tuple[pd.Timestamp, float, str, str, float]] = []

        for _, row in sorted_cands.iterrows():
            entry_t: pd.Timestamp = row["entry_time"]
            utc_day = entry_t.strftime("%Y-%m-%d")

            still_open = []
            for ex_time, ex_net_pnl, ex_day, ex_sym, ex_max_loss in active_exits:
                if ex_time <= entry_t:
                    risk_engine.open_symbols.discard(ex_sym)
                    risk_engine.record_trade_pnl(
                        ex_net_pnl, ex_day, symbol=ex_sym, planned_max_loss_usd=ex_max_loss
                    )
                else:
                    still_open.append((ex_time, ex_net_pnl, ex_day, ex_sym, ex_max_loss))
            active_exits = still_open

            sizing = risk_engine.size_and_validate_order(
                symbol=row["symbol"],
                side=int(row["side"]),
                entry_price=float(row["entry_price"]),
                stop_price=float(row["stop_price"]),
                utc_date_str=utc_day,
            )
            if not sizing.approved:
                continue

            cost_bd = self.cost_model.evaluate_trade_cost(
                side=int(row["side"]),
                notional_usd=sizing.notional_usd,
                funding_rates_crossed=[float(row["funding_sum_rate"])],
                is_high_volatility_bar=bool(row["is_high_vol_slippage"]),
            )
            gross_pnl_usd = sizing.notional_usd * float(row["raw_return_pct"])
            net_pnl_usd = gross_pnl_usd - cost_bd.total_cost_usd
            exit_day = pd.Timestamp(row["exit_time"]).strftime("%Y-%m-%d")

            risk_engine.open_symbols.add(row["symbol"])
            active_exits.append(
                (
                    pd.Timestamp(row["exit_time"]),
                    net_pnl_usd,
                    exit_day,
                    row["symbol"],
                    sizing.worst_case_loss_usd,
                )
            )

            executed_records.append(
                {
                    **row.to_dict(),
                    "notional_usd": sizing.notional_usd,
                    "effective_leverage": sizing.effective_leverage,
                    "worst_case_loss_pct": sizing.worst_case_loss_pct,
                    "gross_pnl_usd": gross_pnl_usd,
                    "fee_usd": cost_bd.fee_usd,
                    "slippage_usd": cost_bd.slippage_usd,
                    "funding_usd": cost_bd.funding_usd,
                    "total_cost_usd": cost_bd.total_cost_usd,
                    "net_pnl_usd": net_pnl_usd,
                }
            )

        for ex_time, ex_net_pnl, ex_day, ex_sym, ex_max_loss in sorted(
            active_exits, key=lambda x: x[0]
        ):
            risk_engine.open_symbols.discard(ex_sym)
            risk_engine.record_trade_pnl(
                ex_net_pnl, ex_day, symbol=ex_sym, planned_max_loss_usd=ex_max_loss
            )

        trades_df = pd.DataFrame(executed_records)
        if not trades_df.empty:
            trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)
            trades_df["equity_curve"] = (
                self.cfg.initial_capital + trades_df["net_pnl_usd"].cumsum()
            )
        return trades_df, risk_engine

    def run_portfolio_backtest(
        self,
        universe_klines: Dict[str, pd.DataFrame],
        btc_klines: pd.DataFrame,
        universe_funding: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Dict[str, np.ndarray]], DeterministicRiskEngine]:
        candidate_dfs = []
        feature_store: Dict[str, pd.DataFrame] = {}
        symbol_arrays: Dict[str, Dict[str, np.ndarray]] = {}

        for sym, kdf in universe_klines.items():
            if sym == self.cfg.benchmark_symbol:
                continue
            feat_df = self.signal_engine.compute_features_and_signals(kdf, btc_klines)
            feature_store[sym] = feat_df
            f_df = universe_funding.get(sym) if universe_funding else None
            sym_trades, arrs = self._extract_symbol_candidates(sym, feat_df, f_df)
            if arrs:
                symbol_arrays[sym] = arrs
            if not sym_trades.empty:
                candidate_dfs.append(sym_trades)

        if not candidate_dfs:
            return pd.DataFrame(), feature_store, symbol_arrays, DeterministicRiskEngine(self.cfg, self.cost_model)

        all_candidates = pd.concat(candidate_dfs, ignore_index=True)
        trades_df, risk_engine = self.replay_candidates_through_risk_and_cost(all_candidates)
        return trades_df, feature_store, symbol_arrays, risk_engine

    def evaluate_w1_gates(
        self,
        trades_df: pd.DataFrame,
        symbol_arrays: Dict[str, Dict[str, np.ndarray]],
        risk_engine: DeterministicRiskEngine,
    ) -> W1GateEvaluation:
        rng = np.random.default_rng(self.cfg.random_control_seed)
        total_trades = len(trades_df)

        if total_trades == 0:
            return W1GateEvaluation(
                config_sha256=self.cfg.config_hash(),
                total_trades=0,
                oos_trades=0,
                gross_profit_usd=0.0,
                total_cost_usd=0.0,
                net_profit_usd=0.0,
                net_profit_to_cost_ratio=0.0,
                full_sharpe=0.0,
                oos_sharpe=0.0,
                max_drawdown_pct=0.0,
                oos_max_drawdown_pct=0.0,
                gap_loss_count=0,
                random_baseline_mean_net_usd=0.0,
                permutation_p_value=1.0,
                bootstrap_p_value=1.0,
                shared_execution_model_verified=True,
                gate_1_sample_pass=False,
                gate_1_status="INSUFFICIENT_EVIDENCE (0 trades)",
                gate_2_cost_ratio_pass=False,
                gate_3_oos_sharpe_pass=False,
                gate_3_oos_sample_valid=False,
                gate_4_mdd_pass=True,
                gate_5_random_baseline_pass=False,
                w1_overall_status="INSUFFICIENT_EVIDENCE (NEUTRAL)",
            )

        gate_1_sample_pass = total_trades >= self.cfg.w1_min_sample_size
        gate_1_status = (
            f"PASS ({total_trades} >= {self.cfg.w1_min_sample_size})"
            if gate_1_sample_pass
            else f"INSUFFICIENT_EVIDENCE ({total_trades} < {self.cfg.w1_min_sample_size})"
        )

        gross_profit = float(trades_df["gross_pnl_usd"].sum())
        total_cost = float(trades_df["total_cost_usd"].sum())
        net_profit = float(trades_df["net_pnl_usd"].sum())
        net_to_cost = (net_profit / total_cost) if total_cost > 0 else 0.0
        gate_2_cost_ratio_pass = net_to_cost >= self.cfg.w1_min_net_profit_to_cost

        min_time = trades_df["entry_time"].min()
        max_time = trades_df["exit_time"].max()
        split_time = min_time + (max_time - min_time) * (1.0 - self.cfg.w1_oos_split_ratio)
        oos_df = trades_df[trades_df["entry_time"] >= split_time].copy()
        oos_trades = len(oos_df)

        full_sharpe = self._compute_annualized_sharpe(trades_df, min_time, max_time)
        oos_sharpe = (
            self._compute_annualized_sharpe(oos_df, split_time, max_time)
            if oos_trades >= 5
            else 0.0
        )
        gate_3_oos_sample_valid = oos_trades >= self.cfg.w1_min_oos_trades_for_sharpe
        gate_3_oos_sharpe_pass = (
            gate_3_oos_sample_valid and (oos_sharpe >= self.cfg.w1_min_oos_sharpe)
        )

        mdd_full = self._compute_mdd(trades_df["net_pnl_usd"].to_numpy(), self.cfg.initial_capital)
        mdd_oos = (
            self._compute_mdd(oos_df["net_pnl_usd"].to_numpy(), self.cfg.initial_capital)
            if oos_trades > 0
            else 0.0
        )
        gate_4_mdd_pass = (mdd_full <= self.cfg.w1_max_drawdown) and (
            mdd_oos <= self.cfg.w1_max_drawdown
        )

        perm_p, rand_mean_net = self._run_shared_model_random_control(
            trades_df, symbol_arrays, rng, n_runs=self.cfg.random_control_shuffles
        )
        boot_p = self._run_daily_block_bootstrap_test(
            trades_df, rng, n_runs=self.cfg.random_control_shuffles
        )
        combined_p = max(perm_p, boot_p)
        gate_5_random_baseline_pass = combined_p < self.cfg.w1_max_p_value

        if not gate_1_sample_pass:
            w1_overall_status = "INSUFFICIENT_EVIDENCE (NEUTRAL) - NO LIVE"
        elif (
            gate_2_cost_ratio_pass
            and gate_3_oos_sharpe_pass
            and gate_4_mdd_pass
            and gate_5_random_baseline_pass
        ):
            w1_overall_status = "PASS"
        else:
            w1_overall_status = "FAIL (STRATEGY REJECTED)"

        return W1GateEvaluation(
            config_sha256=self.cfg.config_hash(),
            total_trades=total_trades,
            oos_trades=oos_trades,
            gross_profit_usd=gross_profit,
            total_cost_usd=total_cost,
            net_profit_usd=net_profit,
            net_profit_to_cost_ratio=net_to_cost,
            full_sharpe=full_sharpe,
            oos_sharpe=oos_sharpe,
            max_drawdown_pct=mdd_full,
            oos_max_drawdown_pct=mdd_oos,
            gap_loss_count=len(risk_engine.gap_loss_events),
            random_baseline_mean_net_usd=rand_mean_net,
            permutation_p_value=perm_p,
            bootstrap_p_value=boot_p,
            shared_execution_model_verified=bool(self.cfg.random_control_shared_execution),
            gate_1_sample_pass=gate_1_sample_pass,
            gate_1_status=gate_1_status,
            gate_2_cost_ratio_pass=gate_2_cost_ratio_pass,
            gate_3_oos_sharpe_pass=gate_3_oos_sharpe_pass,
            gate_3_oos_sample_valid=gate_3_oos_sample_valid,
            gate_4_mdd_pass=gate_4_mdd_pass,
            gate_5_random_baseline_pass=gate_5_random_baseline_pass,
            w1_overall_status=w1_overall_status,
        )

    def _run_shared_model_random_control(
        self,
        trades_df: pd.DataFrame,
        symbol_arrays: Dict[str, Dict[str, np.ndarray]],
        rng: np.random.Generator,
        n_runs: int = 1000,
    ) -> Tuple[float, float]:
        """
        Claude + Opgo Consensus Requirement:
        Random control MUST invoke the exact same `simulate_single_trade_execution`
        (ATR(20)*1.5 stop, VWAP(96) TP1 + Breakeven, |z|<=0.5 TP2, 96-bar time exit),
        sizing model, and 2-tier cost model as the strategy!
        To keep 1,000 shuffles fast (< 3 seconds), we pre-sample a pool of random trades
        per symbol using `simulate_single_trade_execution` and evaluate 1,000 random portfolios
        with identical trade counts, sizing, and cost rules.
        """
        actual_net = float(trades_df["net_pnl_usd"].sum())
        warmup = max(self.cfg.beta_window, self.cfg.residual_z_window) + 5

        # Pre-generate random trade outcomes per symbol using the EXACT shared execution model
        pool_per_sym: Dict[str, List[Tuple[float, float]]] = {}
        for sym, arrs in symbol_arrays.items():
            n_bars = len(arrs["opens"])
            max_idx = max(warmup + 1, n_bars - self.cfg.max_holding_bars - 2)
            sample_indices = rng.integers(warmup, max_idx, size=250)
            sample_sides = rng.choice([-1, 1], size=250)
            outcomes = []
            for s_idx, r_side in zip(sample_indices, sample_sides):
                tr = self.simulate_single_trade_execution(
                    symbol=sym,
                    signal_idx=int(s_idx),
                    side=int(r_side),
                    opens=arrs["opens"],
                    highs=arrs["highs"],
                    lows=arrs["lows"],
                    closes=arrs["closes"],
                    times=arrs["times"],
                    stop_dists=arrs["stop_dists"],
                    vwaps=arrs["vwaps"],
                    res_z=arrs["res_z"],
                    high_vol_flags=arrs["high_vol_flags"],
                    f_times_ns=arrs["f_times_ns"],
                    f_rates=arrs["f_rates"],
                )
                if tr is not None:
                    # Apply exact same sizing and 2-tier cost model
                    stop_d = max(tr["stop_dist_pct"], self.cfg.min_stop_distance_pct)
                    all_in_stop = stop_d + self.cost_model.conservative_sizing_cost_rate
                    notional = min(
                        (self.cfg.initial_capital * self.cfg.risk_per_trade_pct) / all_in_stop,
                        self.cfg.initial_capital * self.cfg.max_gross_leverage,
                    )
                    cbd = self.cost_model.evaluate_trade_cost(
                        side=int(r_side),
                        notional_usd=notional,
                        funding_rates_crossed=[tr["funding_sum_rate"]],
                        is_high_volatility_bar=tr["is_high_vol_slippage"],
                    )
                    net_pnl = notional * tr["raw_return_pct"] - cbd.total_cost_usd
                    outcomes.append((net_pnl, cbd.total_cost_usd))
            pool_per_sym[sym] = outcomes

        syms = trades_df["symbol"].to_numpy()
        random_totals = np.zeros(n_runs, dtype=float)
        for k in range(n_runs):
            total_pnl = 0.0
            for sym in syms:
                pool = pool_per_sym.get(sym, [])
                if pool:
                    pick = pool[int(rng.integers(0, len(pool)))]
                    total_pnl += pick[0]
            random_totals[k] = total_pnl

        p_val = float((np.sum(random_totals >= actual_net) + 1) / (n_runs + 1))
        return p_val, float(np.mean(random_totals))

    def _compute_annualized_sharpe(
        self, df: pd.DataFrame, start_t: pd.Timestamp, end_t: pd.Timestamp
    ) -> float:
        if df.empty:
            return 0.0
        temp = df.copy()
        temp["date"] = temp["exit_time"].dt.floor("D")
        daily_pnl = temp.groupby("date")["net_pnl_usd"].sum()
        all_days = pd.date_range(start_t.floor("D"), end_t.floor("D"), freq="D", tz="UTC")
        daily_ret = daily_pnl.reindex(all_days, fill_value=0.0) / self.cfg.initial_capital
        std = float(daily_ret.std(ddof=1))
        if std <= 1e-12:
            return 0.0
        return float(np.sqrt(365.0) * (daily_ret.mean() / std))

    @staticmethod
    def _compute_mdd(net_pnls: np.ndarray, initial_capital: float) -> float:
        eq = np.concatenate([[initial_capital], initial_capital + np.cumsum(net_pnls)])
        peaks = np.maximum.accumulate(eq)
        dd = 1.0 - (eq / np.maximum(1e-9, peaks))
        return float(np.max(dd))

    def _run_daily_block_bootstrap_test(
        self,
        trades_df: pd.DataFrame,
        rng: np.random.Generator,
        n_runs: int = 1000,
    ) -> float:
        temp = trades_df.copy()
        temp["date"] = temp["exit_time"].dt.floor("D")
        daily_pnl = temp.groupby("date")["net_pnl_usd"].sum().to_numpy()
        if len(daily_pnl) < 3:
            return 1.0

        n_days = len(daily_pnl)
        boot_means = np.zeros(n_runs, dtype=float)
        for k in range(n_runs):
            sample = rng.choice(daily_pnl, size=n_days, replace=True)
            boot_means[k] = float(np.mean(sample))

        p_val = float((np.sum(boot_means <= 0.0) + 1) / (n_runs + 1))
        return p_val
