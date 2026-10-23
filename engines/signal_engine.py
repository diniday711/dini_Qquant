"""
Phase 1 Pre-Registered Single Confluence Signal Engine (v2.1 - Council Spec)
Implements fixed pre-registered parameters:
- Timeframe: 15m
- Rolling Beta Window: W_beta = 96 bars (24h), strictly shifted by 1 bar (`shift(1)`)
- Residual Standardization Window: W_z = 192 bars (48h), strictly shifted by 1 bar
- Swing High/Low Window: N = 20 bars (5h), strictly shifted by 1 bar
- Rolling VWAP Window: 96 bars (24h) for TP1 target
- ATR Window: 20 bars, Stop = max(1.5 * ATR(20), 0.30%)
- Candle Volume Delta: explicitly named `candle_delta_15m` = 2 * taker_buy_volume - volume
- BTC Clash Filter: |r_BTC(24)| >= 1.0% opposing trade direction -> ABSTAIN
"""
import numpy as np
import pandas as pd
from config import CONFIG, Phase1Config
from engines.cost_model import CostModel


class SingleConfluenceSignalEngine:
    def __init__(self, cfg: Phase1Config = CONFIG, cost_model: CostModel | None = None):
        self.cfg = cfg
        self.cost_model = cost_model or CostModel(cfg)

    def compute_features_and_signals(
        self, asset_df: pd.DataFrame, btc_df: pd.DataFrame
    ) -> pd.DataFrame:
        df = pd.merge(
            asset_df[
                [
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "taker_buy_volume",
                ]
            ].copy(),
            btc_df[["open_time", "close"]].rename(columns={"close": "btc_close"}),
            on="open_time",
            how="inner",
        ).sort_values("open_time").reset_index(drop=True)

        # 1. Explicit Candle-Level Volume Delta (Renamed per Council Consensus: `candle_delta_15m`)
        df["candle_delta_15m"] = 2.0 * df["taker_buy_volume"] - df["volume"]
        df["candle_delta_ratio_15m"] = np.where(
            df["volume"] > 0, df["candle_delta_15m"] / df["volume"], 0.0
        )

        # 2. 1-Bar Returns
        df["asset_ret"] = df["close"].pct_change()
        df["btc_ret"] = df["btc_close"].pct_change()

        # 3. Rolling OLS Beta (W_beta = 96) estimated strictly on t-1 (`shift(1)`)
        w_beta = self.cfg.beta_window
        cov_ab = df["asset_ret"].rolling(w_beta).cov(df["btc_ret"]).shift(1)
        var_b = df["btc_ret"].rolling(w_beta).var().shift(1)
        beta_est = np.where(var_b > 1e-12, cov_ab / var_b, 1.0)
        beta_est = np.clip(beta_est, -5.0, 5.0)

        mean_a = df["asset_ret"].rolling(w_beta).mean().shift(1)
        mean_b = df["btc_ret"].rolling(w_beta).mean().shift(1)
        alpha_est = mean_a - beta_est * mean_b

        df["rolling_beta"] = beta_est
        df["residual_ret"] = df["asset_ret"] - (alpha_est + beta_est * df["btc_ret"])

        # 4. Cumulative Residual & Standardization over W_z = 192 bars (48h)
        w_res = self.cfg.residual_window
        w_z = self.cfg.residual_z_window
        df["cum_residual"] = df["residual_ret"].rolling(w_res).sum()

        res_mean_prev = df["cum_residual"].rolling(w_z).mean().shift(1)
        res_std_prev = df["cum_residual"].rolling(w_z).std().shift(1)
        df["residual_z"] = np.where(
            res_std_prev > 1e-8,
            (df["cum_residual"] - res_mean_prev) / res_std_prev,
            0.0,
        )

        # 5. Prior N=20 Swing High / Low (`shift(1)`)
        n_swing = self.cfg.swing_lookback
        df["prior_swing_high"] = df["high"].shift(1).rolling(n_swing).max()
        df["prior_swing_low"] = df["low"].shift(1).rolling(n_swing).min()

        # 6. Rolling VWAP(96) for TP1
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        pv_sum = (typical_price * df["volume"]).rolling(self.cfg.vwap_window).sum()
        v_sum = df["volume"].rolling(self.cfg.vwap_window).sum()
        df["vwap_96"] = np.where(v_sum > 0, pv_sum / v_sum, df["close"])

        # 7. Causal ATR(20) & High-Volatility Bar Flag (for 2-Tier Slippage 0.04% vs 0.15%)
        prev_close = df["close"].shift(1)
        tr = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ),
        )
        df["atr_20"] = tr.rolling(self.cfg.atr_window).mean()
        df["bar_range"] = df["high"] - df["low"]
        df["is_high_vol_bar"] = df["bar_range"] > (
            self.cfg.high_vol_atr_mult_threshold * df["atr_20"]
        )

        # 8. Stop Distance = max(ATR(20) * 1.5 / close, 0.30%)
        raw_atr_stop_pct = (self.cfg.stop_atr_mult * df["atr_20"]) / df["close"]
        df["stop_dist_pct"] = np.maximum(
            raw_atr_stop_pct.fillna(self.cfg.min_stop_distance_pct),
            self.cfg.min_stop_distance_pct,
        )

        # 9. BTC 24-Bar Return for Directional Clash Abstention (|r_BTC(24)| >= 1.0%)
        df["btc_ret_24"] = df["btc_close"].pct_change(self.cfg.btc_momentum_window)

        # 10. Binary Confluence Conditions at close of bar `t`
        z_thresh = self.cfg.residual_z_threshold
        cond_short_z = df["residual_z"] >= z_thresh
        cond_long_z = df["residual_z"] <= -z_thresh

        upper_wick = df["high"] - np.maximum(df["open"], df["close"])
        lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
        candle_range = np.maximum(1e-8, df["bar_range"])

        swept_high_and_failed = (df["high"] >= df["prior_swing_high"]) & (
            (df["close"] < df["prior_swing_high"])
            | (
                (upper_wick / candle_range >= 0.35)
                & (df["candle_delta_ratio_15m"] < 0.0)
            )
        )
        swept_low_and_failed = (df["low"] <= df["prior_swing_low"]) & (
            (df["close"] > df["prior_swing_low"])
            | (
                (lower_wick / candle_range >= 0.35)
                & (df["candle_delta_ratio_15m"] > 0.0)
            )
        )

        raw_short = cond_short_z & swept_high_and_failed
        raw_long = cond_long_z & swept_low_and_failed

        # Expected reward distance to VWAP(96) mean reversion target
        short_vwap_dist_pct = np.maximum(0.0, (df["close"] - df["vwap_96"]) / df["close"])
        long_vwap_dist_pct = np.maximum(0.0, (df["vwap_96"] - df["close"]) / df["close"])
        df["expected_target_pct"] = np.where(
            raw_short,
            np.maximum(short_vwap_dist_pct, df["stop_dist_pct"] * 1.5),
            np.where(
                raw_long,
                np.maximum(long_vwap_dist_pct, df["stop_dist_pct"] * 1.5),
                0.0,
            ),
        )

        # Deterministic Rule-Based Abstention Filter
        abstain_btc_clash = (
            raw_short & (df["btc_ret_24"] >= self.cfg.btc_clash_return_limit)
        ) | (raw_long & (df["btc_ret_24"] <= -self.cfg.btc_clash_return_limit))

        roundtrip_cost_for_bar = np.where(
            df["is_high_vol_bar"],
            self.cost_model.get_roundtrip_cost_rate(True),
            self.cost_model.get_roundtrip_cost_rate(False),
        )
        abstain_cost_erosion = (raw_short | raw_long) & (
            df["expected_target_pct"]
            < (roundtrip_cost_for_bar * self.cfg.min_reward_to_cost_mult)
        )

        df["abstain_reason"] = np.where(
            abstain_btc_clash,
            "ABSTAIN_BTC_DIRECTION_CLASH",
            np.where(abstain_cost_erosion, "ABSTAIN_COST_EROSION", ""),
        )

        df["signal"] = 0
        df.loc[raw_long & (~abstain_btc_clash) & (~abstain_cost_erosion), "signal"] = 1
        df.loc[raw_short & (~abstain_btc_clash) & (~abstain_cost_erosion), "signal"] = -1

        return df
