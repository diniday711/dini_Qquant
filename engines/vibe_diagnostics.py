"""
Vibe-Trading Inspired Diagnostics Engine (`engines/vibe_diagnostics.py`)
Implements two lightweight, zero-LLM deterministic diagnostics adapted from HKUDS/Vibe-Trading:
1. Cross-Coin Correlation Regime & Herd Detector (`compute_correlation_regime`):
   - Measures pairwise 1h return correlations and edge density (`abs(corr) >= 0.80`) over trailing 72 hours.
   - Automatically throttles `recommended_max_slots` from `3 -> 2 -> 1` when altcoins lock into a single herd (`corr -> 1.0`).
2. Rolling Alpha Vitality Monitor (`compute_alpha_vitality_diagnostics`):
   - Computes 30-day rolling Information Coefficient (`IC` & `IC_IR`) strictly on completed historical bars (`t <= len - 5`)
     against 4h forward returns with zero look-ahead leakage.
   - Classifies each strategy family as `ALIVE` (`IC >= +0.015`), `DEAD` (`-0.020 < IC < +0.015`), or `REVERSED` (`IC <= -0.020`).
"""
from typing import Any, Dict, List
import numpy as np
import pandas as pd


STRATEGY_TO_VITALITY_FAMILY: Dict[str, str] = {
    "WONYOTTI_VOLUME_CLUSTER_BREAKOUT": "WONYOTTI_VOL_BREAKOUT",
    "DONCHIAN_EMA_MOMENTUM": "WONYOTTI_VOL_BREAKOUT",
    "SATOSHI_RELATIVE_STRENGTH_ROTATION": "SATOSHI_REL_STRENGTH",
    "WONYOTTI_RANGE_ABSORPTION_REVERSION": "RANGE_ABSORPTION_REVERSION",
    "BBANDS_CVD_MEAN_REVERSION": "RANGE_ABSORPTION_REVERSION",
    "CASH_DEFENSIVE_LOCK": "NONE",
}


def compute_correlation_regime(
    univ_1h: Dict[str, pd.DataFrame],
    lookback_bars: int = 72,
) -> Dict[str, Any]:
    """
    Compute pairwise 1h return correlation matrix and edge density (`abs(corr) >= 0.80`)
    across the universe over the trailing `lookback_bars` closed candles.
    """
    ret_map: Dict[str, np.ndarray] = {}
    min_len = lookback_bars

    for sym, df in sorted(univ_1h.items()):
        if df is None or len(df) < 10:
            continue
        closes = df["close"].astype(float).values
        rets = np.diff(closes) / np.maximum(1e-9, closes[:-1])
        tail = rets[-lookback_bars:] if len(rets) >= lookback_bars else rets
        if len(tail) >= 8:
            ret_map[sym] = tail
            min_len = min(min_len, len(tail))

    symbols = sorted(ret_map.keys())
    if len(symbols) < 2 or min_len < 8:
        return {
            "corr_regime": "DISPERSED_NORMAL",
            "avg_pairwise_corr": 0.0,
            "edge_density_gt_080": 0.0,
            "recommended_max_slots": 3,
            "max_corr_pair": "NONE",
            "max_corr_value": 0.0,
            "korean_summary": "🟢 [개별 장세 (DISPERSED_NORMAL)] 평균 상관계수 0.00 | 동조화 밀도 0% -> 기본 3슬롯 유지",
        }

    mat = np.column_stack([ret_map[s][-min_len:] for s in symbols])
    # Guard against zero-variance series
    stds = np.std(mat, axis=0)
    pair_corrs: List[float] = []
    high_edges = 0
    max_pair = "NONE"
    max_corr_val = -1.0

    n_sym = len(symbols)
    for i in range(n_sym):
        for j in range(i + 1, n_sym):
            if stds[i] <= 1e-12 or stds[j] <= 1e-12:
                c = 0.0
            else:
                c_val = np.corrcoef(mat[:, i], mat[:, j])[0, 1]
                c = float(c_val) if np.isfinite(c_val) else 0.0
            pair_corrs.append(c)
            if abs(c) >= 0.80:
                high_edges += 1
            if c > max_corr_val:
                max_corr_val = c
                max_pair = f"{symbols[i]}-{symbols[j]}"

    avg_corr = float(np.mean(pair_corrs)) if pair_corrs else 0.0
    edge_density = float(high_edges / len(pair_corrs)) if pair_corrs else 0.0

    if avg_corr >= 0.78 or edge_density >= 0.65:
        corr_regime = "HERD_LOCK_EXTREME"
        rec_slots = 1
        badge = "🚨 [극단적 동조화 경보 (HERD_LOCK_EXTREME)] 코인들이 한 덩어리로 움직임 -> 최대 슬롯 3개에서 1개로 강제 축소!"
    elif avg_corr >= 0.65 or edge_density >= 0.45:
        corr_regime = "ELEVATED_SYNC"
        rec_slots = 2
        badge = "🟡 [동조화 주의 (ELEVATED_SYNC)] 코인 간 커플링 심화 -> 최대 슬롯 3개에서 2개로 축소"
    else:
        corr_regime = "DISPERSED_NORMAL"
        rec_slots = 3
        badge = "🟢 [개별 순환매 정상 (DISPERSED_NORMAL)] 코인별 독립 움직임 양호 -> 최대 3슬롯 모두 허용"

    korean_summary = (
        f"{badge} (72h 평균 상관계수 {avg_corr:+.2f}, 강한 동조화 비율 {edge_density:.0%}, "
        f"최고 동조화 쌍 {max_pair} {max_corr_val:+.2f})"
    )

    return {
        "corr_regime": corr_regime,
        "avg_pairwise_corr": round(avg_corr, 4),
        "edge_density_gt_080": round(edge_density, 4),
        "recommended_max_slots": rec_slots,
        "max_corr_pair": max_pair,
        "max_corr_value": round(float(max_corr_val), 4),
        "korean_summary": korean_summary,
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 12:
        return 0.0
    xv = x[mask]
    yv = y[mask]
    if np.std(xv) <= 1e-12 or np.std(yv) <= 1e-12:
        return 0.0
    c = float(np.corrcoef(xv, yv)[0, 1])
    return c if np.isfinite(c) else 0.0


def compute_alpha_vitality_diagnostics(
    univ_1h: Dict[str, pd.DataFrame],
    btc_1h: pd.DataFrame,
    lookback_bars: int = 720,
) -> Dict[str, Any]:
    """
    Evaluate 30-day rolling Information Coefficient (`IC` & `IC_IR`) for the 3 core Q-Stack
    signal families strictly on completed historical windows (`idx <= len - 5`) so that
    the 4-bar forward return (`close[t+4]/close[t] - 1`) never touches the current bar `T` or future bars.
    """
    family_ics: Dict[str, List[float]] = {
        "WONYOTTI_VOL_BREAKOUT": [],
        "SATOSHI_REL_STRENGTH": [],
        "RANGE_ABSORPTION_REVERSION": [],
    }

    btc_close = btc_1h["close"].astype(float).values if btc_1h is not None and len(btc_1h) > 30 else None

    for sym, df in sorted(univ_1h.items()):
        if df is None or len(df) < 40:
            continue
        # Strictly exclude the final bar (n_total - 1) so 4h forward returns are 100% realized prior to current bar T
        n_total = len(df)
        end_idx = n_total - 5
        start_idx = max(25, end_idx - lookback_bars)
        if end_idx - start_idx < 16:
            continue

        closes = df["close"].astype(float).values
        vols = df["volume"].astype(float).values
        deltas = (
            df["delta_ratio"].astype(float).values
            if "delta_ratio" in df.columns
            else np.zeros_like(closes)
        )

        # Forward 4h return realized within [start_idx, end_idx)
        fwd_4h = (closes[start_idx + 4 : end_idx + 4] / np.maximum(1e-9, closes[start_idx:end_idx])) - 1.0

        # 1) WONYOTTI_VOL_BREAKOUT signal at bar t (using data up to bar t):
        #    20-bar volume ratio * 12-bar momentum
        s_vol_brk = np.zeros(end_idx - start_idx)
        # 2) SATOSHI_REL_STRENGTH signal at bar t:
        #    24-bar coin return minus 24-bar BTC return
        s_rel_str = np.zeros(end_idx - start_idx)
        # 3) RANGE_ABSORPTION_REVERSION signal at bar t:
        #    negative 20-bar z-score * positive absorption
        s_rng_rev = np.zeros(end_idx - start_idx)

        for k, t in enumerate(range(start_idx, end_idx)):
            vol_ma20 = float(np.mean(vols[t - 20 : t]))
            vol_ratio = vols[t] / max(1e-9, vol_ma20)
            mom_12 = (closes[t] / max(1e-9, closes[t - 12])) - 1.0
            s_vol_brk[k] = vol_ratio * np.sign(mom_12) * abs(mom_12)

            ret_24 = (closes[t] / max(1e-9, closes[t - 24])) - 1.0
            if btc_close is not None and t < len(btc_close) and t >= 24:
                btc_ret_24 = (btc_close[t] / max(1e-9, btc_close[t - 24])) - 1.0
            else:
                btc_ret_24 = 0.0
            s_rel_str[k] = ret_24 - btc_ret_24

            win_c = closes[t - 20 : t + 1]
            c_mean = float(np.mean(win_c))
            c_std = float(np.std(win_c))
            z_score = (closes[t] - c_mean) / max(1e-9, c_std)
            s_rng_rev[k] = -z_score * (1.0 + 0.5 * deltas[t])

        family_ics["WONYOTTI_VOL_BREAKOUT"].append(_safe_corr(s_vol_brk, fwd_4h))
        family_ics["SATOSHI_REL_STRENGTH"].append(_safe_corr(s_rel_str, fwd_4h))
        family_ics["RANGE_ABSORPTION_REVERSION"].append(_safe_corr(s_rng_rev, fwd_4h))

    families_report: Dict[str, Dict[str, Any]] = {}
    alive_count = 0
    dead_or_reversed: List[str] = []

    korean_labels = {
        "WONYOTTI_VOL_BREAKOUT": "워뇨띠 거래량 클러스터 돌파",
        "SATOSHI_REL_STRENGTH": "사토시 BTC 대비 상대강도",
        "RANGE_ABSORPTION_REVERSION": "박스권 하단 흡수 역추세",
    }

    summary_parts: List[str] = []
    for fam, ic_list in family_ics.items():
        if not ic_list:
            ic_mean = 0.0
            ic_std = 0.0
            ic_ir = 0.0
        else:
            ic_mean = float(np.mean(ic_list))
            ic_std = float(np.std(ic_list))
            ic_ir = float(ic_mean / ic_std) if ic_std > 1e-6 else 0.0

        if ic_mean >= 0.015:
            status = "ALIVE"
            weight_mult = 1.0
            alive_count += 1
            status_kr = "🟢 ALIVE(작동중)"
        elif ic_mean <= -0.020:
            status = "REVERSED"
            weight_mult = 0.0
            dead_or_reversed.append(fam)
            status_kr = "🔴 REVERSED(역작동-차단)"
        else:
            status = "DEAD"
            weight_mult = 0.5
            dead_or_reversed.append(fam)
            status_kr = "🟡 DEAD(수명약화-비중축소)"

        families_report[fam] = {
            "label_kr": korean_labels[fam],
            "ic_mean": round(ic_mean, 4),
            "ic_ir": round(ic_ir, 3),
            "vitality_status": status,
            "status_kr": status_kr,
            "weight_multiplier": weight_mult,
        }
        summary_parts.append(f"{korean_labels[fam]}: {status_kr} (IC {ic_mean:+.3f})")

    return {
        "families": families_report,
        "active_alive_count": alive_count,
        "dead_or_reversed_families": dead_or_reversed,
        "strategy_to_family_map": STRATEGY_TO_VITALITY_FAMILY,
        "korean_summary": " | ".join(summary_parts),
    }
