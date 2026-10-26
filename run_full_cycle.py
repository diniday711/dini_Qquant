"""
Q-Stack End-to-End Full Cycle Orchestrator, Autonomous 24/7 Decision Engine & 6-Panel Dashboard
Completes one full autonomous loop:
  [Step 1. Research & 3-Axis Regime Radar (Hatcha)]
     -> Classifies Trend, Volatility, and Breadth across BTC + 10 Altcoins.
  [Step 2. Dynamic Strategy Switcher (Quant-Eng)]
     -> Automatically selects TREND_BREAKOUT, RANGE_MEAN_REVERSION, or STAY_IN_CASH.
  [Step 3. Structured Multi-Persona Debate & Autonomous Decision (Bull vs Bear vs Eggbot -> Apex CIO)]
     -> Generates explicit Korean Meeting Logs (회의록) for every candidate and makes the 24/7 autonomous
        INVEST / ABSTAIN decision without requiring human intervention at 3 AM.
  [Step 4. Deterministic Risk Sizing & Binance Testnet Execution (Potato-Shield)]
     -> Sizes approved trades at 1.0% risk cap, attaches STOP_MARKET protection, and executes on Testnet/Shadow.
  [Step 5. Crumb SQLite Memory & Hermes Discord Briefing Generator]
     -> Saves to `data/crumb_memory.db`, renders `reports/qstack_dashboard.html`, and writes
        `reports/latest_discord_briefing.md` for Hermes Agent / Discord notifications.
"""
import argparse
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, HTTPServer
import json
import os
import threading
import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from config import CONFIG
from engines.binance_testnet_client import BinanceFuturesTestnetClient
from engines.cost_model import CostModel
from engines.crumb_memory import CrumbMemoryDB
from engines.data_fetcher import BinanceDataFetcher
from engines.eggbot_auditor import DrEggbotAuditor
from engines.position_manager import PositionLifecycleManager
from engines.regime_switcher import compute_adx_causal, run_regime_switching_study
from engines.risk_engine import DeterministicRiskEngine
from engines.vibe_diagnostics import (
    STRATEGY_TO_VITALITY_FAMILY,
    compute_alpha_vitality_diagnostics,
    compute_correlation_regime,
)
from engines.world_monitor_radar import WorldMonitorRadar
from run_phase1 import run_w2_unit_tests


def build_autonomous_debate_and_regime_scan(
    univ_1h: Dict[str, pd.DataFrame],
    btc_1h: pd.DataFrame,
    fund_dict: Dict[str, pd.DataFrame],
    eggbot: DrEggbotAuditor,
    risk_engine: DeterministicRiskEngine,
    simulate_demo_fill_if_all_cash: bool = True,
) -> Dict[str, Any]:
    btc_sorted = btc_1h.sort_values("open_time").reset_index(drop=True)
    btc_ret_24 = float(btc_sorted["close"].pct_change(24).iloc[-1])

    above_200_flags = []
    for sym, kdf in univ_1h.items():
        s_close = kdf.sort_values("open_time")["close"]
        ma200 = float(s_close.rolling(200).mean().iloc[-1])
        above_200_flags.append(1.0 if float(s_close.iloc[-1]) > ma200 else 0.0)
    breadth_pct = float(np.mean(above_200_flags)) if above_200_flags else 0.5

    # Run WorldMonitor Global Intelligence, Macro Cross-Asset & Sentiment Radar FIRST
    world_radar = WorldMonitorRadar()
    world_verdict = world_radar.evaluate_world_state(btc_ret_24, breadth_pct)

    # Run Vibe-Trading Correlation Herd Detector & 30d Rolling Alpha Vitality Monitor
    corr_diag = compute_correlation_regime(univ_1h, lookback_bars=72)
    vitality_diag = compute_alpha_vitality_diagnostics(univ_1h, btc_1h, lookback_bars=720)

    coin_rows: List[Dict[str, Any]] = []
    debate_logs: List[Dict[str, Any]] = []
    vol_states = []
    trend_states = []
    utc_today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for sym, kdf in univ_1h.items():
        df = kdf.sort_values("open_time").reset_index(drop=True).copy()
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
        df["adx_14"] = compute_adx_causal(df, 14)

        prev_c = df["close"].shift(1)
        tr = np.maximum(
            df["high"] - df["low"],
            np.maximum((df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()),
        )
        df["atr_20"] = tr.rolling(20).mean()
        df["natr"] = df["atr_20"] / df["close"]
        df["vol_pct_rank"] = df["natr"].rolling(192).rank(pct=True)

        delta_c = df["close"].diff()
        gain = delta_c.clip(lower=0.0).rolling(14).mean()
        loss = (-delta_c.clip(upper=0.0)).rolling(14).mean()
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + gain / np.maximum(1e-9, loss)))

        df["bb_mid"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2.2 * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - 2.2 * df["bb_std"]

        df["candle_delta"] = 2.0 * df["taker_buy_volume"] - df["volume"]
        df["delta_ratio"] = np.where(df["volume"] > 0, df["candle_delta"] / df["volume"], 0.0)
        df["vol_ratio"] = df["volume"] / np.maximum(1e-9, df["volume"].shift(1).rolling(24).mean())
        df["high_48"] = df["high"].shift(1).rolling(48).max()
        df["low_48"] = df["low"].shift(1).rolling(48).min()
        df["ret_24"] = df["close"].pct_change(24)
        df["rel_strength_24"] = df["ret_24"] - btc_ret_24

        last = df.iloc[-1]
        close_px = float(last["close"])
        adx_val = float(last["adx_14"])
        vol_rank = float(last["vol_pct_rank"])
        vol_ratio = float(last["vol_ratio"])
        rsi_val = float(last["rsi_14"])
        delta_r = float(last["delta_ratio"])
        rel_str = float(last["rel_strength_24"])
        atr_val = float(last["atr_20"])
        high_48 = float(last["high_48"])
        low_48 = float(last["low_48"])
        bb_up = float(last["bb_upper"])
        bb_dn = float(last["bb_lower"])

        f_df = fund_dict.get(sym)
        latest_funding = (
            float(f_df["funding_rate"].iloc[-1])
            if (f_df is not None and not f_df.empty)
            else 0.0001
        )

        # Axis 1: Volatility
        if vol_rank > 0.92:
            vol_state = "EXTREME"
        elif vol_rank < 0.20:
            vol_state = "COMPRESSED"
        elif vol_ratio > 1.6 or vol_rank > 0.70:
            vol_state = "EXPANDING"
        else:
            vol_state = "NORMAL"
        vol_states.append(vol_state)

        # Axis 2: Trend
        if adx_val >= 22.0 and float(last["ema_20"]) > float(last["ema_50"]):
            trend_state = "TRENDING_UP"
        elif adx_val >= 22.0 and float(last["ema_20"]) < float(last["ema_50"]):
            trend_state = "TRENDING_DOWN"
        else:
            trend_state = "RANGING"
        trend_states.append(trend_state)

        # Axis 3: Strategy Fit Mapping
        if vol_state == "EXTREME":
            risk_state = "RISK_OFF (HANDBRAKE)"
            strategy_fit = "STAY_IN_CASH"
            conf = 85
        elif trend_state == "TRENDING_UP" and breadth_pct >= 0.50 and btc_ret_24 > 0:
            risk_state = "RISK_ON"
            strategy_fit = "TREND_BREAKOUT_LONG"
            conf = min(92, int(60 + adx_val * 0.8))
        elif trend_state == "TRENDING_DOWN" and breadth_pct <= 0.50 and btc_ret_24 < 0:
            risk_state = "RISK_OFF"
            strategy_fit = "TREND_BREAKOUT_SHORT"
            conf = min(92, int(60 + adx_val * 0.8))
        elif trend_state == "RANGING" and vol_state in ("NORMAL", "COMPRESSED"):
            risk_state = "NEUTRAL_RANGE"
            strategy_fit = "RANGE_MEAN_REVERSION"
            conf = 72
        else:
            risk_state = "TRANSITIONING"
            strategy_fit = "STAY_IN_CASH"
            conf = 55

        # Determine candidate direction for autonomous committee evaluation
        cand_side = 1 if rel_str >= 0 else -1
        sub_strat = "TREND" if "TREND" in strategy_fit else "MEAN_REV"
        is_exact_trigger = False

        if strategy_fit == "TREND_BREAKOUT_LONG" and (
            (close_px >= high_48 * 0.995 and delta_r > 0.08)
            or (adx_val >= 35.0 and rel_str >= 0.008 and delta_r > 0.12)
        ):
            cand_side = 1
            sub_strat = "WONYOTTI_VOL_CLUSTER"
            is_exact_trigger = True
        elif strategy_fit == "TREND_BREAKOUT_SHORT" and (
            (close_px <= low_48 * 1.005 and delta_r < -0.08)
            or (adx_val >= 35.0 and rel_str <= -0.008 and delta_r < -0.12)
        ):
            cand_side = -1
            sub_strat = "WONYOTTI_VOL_CLUSTER"
            is_exact_trigger = True
        elif strategy_fit == "RANGE_MEAN_REVERSION":
            if close_px <= bb_dn * 1.005 and rsi_val < 38.0:
                cand_side = 1
                sub_strat = "MEAN_REV"
                is_exact_trigger = True
            elif close_px >= bb_up * 0.995 and rsi_val > 62.0:
                cand_side = -1
                sub_strat = "MEAN_REV"
                is_exact_trigger = True

        pos_mgr = PositionLifecycleManager("data/crumb_memory.db")
        slot_plan = pos_mgr.compute_wonyotti_slot_order_plan(
            equity_usdt=risk_engine.equity,
            entry_price=close_px,
            atr_val=atr_val,
            side_int=cand_side,
        )
        stop_dist = slot_plan["stop_dist_pct"]
        stop_px = slot_plan["stop_price"]
        tp1_px = slot_plan["tp1_price"]
        expected_rr_cost = (stop_dist * 2.2) / 0.0014

        audit = eggbot.audit_candidate_signal(
            symbol=sym,
            side=cand_side,
            sub_strategy=sub_strat,
            adx_14=adx_val,
            vol_pct_rank=vol_rank,
            market_breadth_pct=breadth_pct,
            btc_ret_24=btc_ret_24,
            delta_ratio=delta_r,
            latest_funding_rate=latest_funding,
            expected_reward_to_cost=expected_rr_cost,
        )
        sizing = risk_engine.size_and_validate_order(
            symbol=sym,
            side=cand_side,
            entry_price=close_px,
            stop_price=stop_px,
            utc_date_str=utc_today,
        )

        # Committee Round 3 + WorldMonitor Rule:
        # Order gating combines WorldMonitor Global Context (`allow_long`/`allow_short`/`risk_multiplier > 0`)
        # + 3-Axis Chart Trigger + W2 Sizing.
        world_dir_allowed = (
            (cand_side == 1 and world_verdict.allow_long)
            or (cand_side == -1 and world_verdict.allow_short)
        ) and (world_verdict.risk_multiplier > 0.0)
        mapped_fam = STRATEGY_TO_VITALITY_FAMILY.get(sub_strat, "NONE")
        fam_vitality = vitality_diag["families"].get(mapped_fam, {}).get("vitality_status", "ALIVE")
        vitality_allowed = fam_vitality != "REVERSED"
        final_approved = bool(
            is_exact_trigger and sizing.approved and audit.approved and world_dir_allowed and vitality_allowed
        )

        # Build Korean Multi-Persona Debate Log (회의록)
        research_note = (
            f"[리서치 봇 Hatcha] {sym} 현재가 ${close_px:,.4f} (24h {float(last['ret_24']):+.2%}, "
            f"BTC 대비 상대강도 {rel_str:+.2%}). 장세={trend_state}/{vol_state}, ADX={adx_val:.1f}, RSI={rsi_val:.0f}, "
            f"수급 Delta={delta_r:+.2f}, 펀딩비={latest_funding:.4%}."
        )
        pro_opinion = (
            f"[찬성 봇 Bull/Trend] 배정 전략: `{strategy_fit}` ({'LONG' if cand_side==1 else 'SHORT'}). "
            f"워뇨띠 1/5 슬롯(격리 증거금 ${slot_plan['slot_margin_usd']:,.0f}) | 진입가 ${close_px:,.4f} | "
            f"1차 50% 분할익절가 ${tp1_px:,.4f} (+{slot_plan['tp1_pct']:.1%}) | "
            f"칼손절가 ${stop_px:,.4f} (-{stop_dist:.2%}, 최대손실 -${slot_plan['max_loss_usd']:,.2f})."
        )
        con_opinion = (
            f"[비판 봇 Dr.Eggbot] 반론 점검: {'; '.join(audit.bear_case_challenges)} | "
            f"{'돌파/밴드 타점 도달 완료!' if is_exact_trigger else '아직 48h 돌파선/밴드 상하단 타점에 완전히 닿지 않음(대기 필요).'}"
        )
        cio_decision = (
            f"✅ [최종 승인 - 3단 분할주문 집행] 격리(ISOLATED) {slot_plan['notional_usd']:,.2f} USDT (슬롯배율 {slot_plan['slot_leverage']:.2f}x) 진입 + "
            f"1차 50% 익절(${tp1_px:,.4f}) + 칼손절(${stop_px:,.4f}) 동시 전송!"
            if final_approved
            else f"⏸️ [관망/기각 (ABSTAIN)] {'[Vibe-Trading 수명감지: REVERSED(역작동) 진입차단]' if not vitality_allowed else ('타점 미도달(대기)' if not is_exact_trigger else audit.verdict_code)} 사유로 진입 보류."
        )

        debate_logs.append(
            {
                "symbol": sym,
                "side": "LONG" if cand_side == 1 else "SHORT",
                "side_int": cand_side,
                "sub_strategy": sub_strat,
                "strategy_fit": strategy_fit,
                "is_exact_trigger": is_exact_trigger,
                "entry_price": close_px,
                "stop_price": stop_px,
                "tp1_price": tp1_px,
                "slot_margin_usd": slot_plan["slot_margin_usd"],
                "notional_usd": slot_plan["notional_usd"],
                "quantity": slot_plan["quantity"],
                "partial_tp_qty": slot_plan["partial_tp_qty"],
                "leverage": slot_plan["slot_leverage"],
                "slot_plan": slot_plan,
                "eggbot_approved": final_approved,
                "eggbot_verdict": "APPROVED_EXECUTE" if final_approved else "ABSTAIN_WAIT",
                "research_note": research_note,
                "bull_thesis": pro_opinion,
                "bear_challenges": [con_opinion],
                "cio_verdict_text": cio_decision,
                "what_could_be_wrong": audit.what_could_i_be_wrong_about,
                "score_sort": (1000 if final_approved else 0) + abs(rel_str) * 100 + adx_val,
            }
        )

        coin_rows.append(
            {
                "symbol": sym,
                "price": close_px,
                "ret_24h_pct": float(last["ret_24"]),
                "rel_strength_vs_btc": rel_str,
                "adx_14": adx_val,
                "rsi_14": rsi_val,
                "vol_pct_rank": vol_rank,
                "delta_ratio": delta_r,
                "funding_rate_8h": latest_funding,
                "trend_regime": trend_state,
                "volatility_state": vol_state,
                "risk_regime": risk_state,
                "confidence": conf,
                "strategy_fit": strategy_fit,
            }
        )

    debate_logs.sort(key=lambda x: x["score_sort"], reverse=True)

    up_cnt = sum(1 for t in trend_states if t == "TRENDING_UP")
    dn_cnt = sum(1 for t in trend_states if t == "TRENDING_DOWN")
    dom_trend = "TRENDING_UP" if up_cnt >= 5 else ("TRENDING_DOWN" if dn_cnt >= 5 else "RANGING / ROTATION")

    ext_cnt = sum(1 for v in vol_states if v == "EXTREME")
    exp_cnt = sum(1 for v in vol_states if v == "EXPANDING")
    dom_vol = "EXTREME (HANDBRAKE)" if ext_cnt >= 3 else ("EXPANDING" if exp_cnt >= 4 else "NORMAL / COMPRESSED")

    if breadth_pct >= 0.60 and btc_ret_24 > 0:
        dom_risk = "RISK_ON"
        active_mode = "TREND_BREAKOUT_LONG"
    elif breadth_pct <= 0.40 and btc_ret_24 < 0:
        dom_risk = "RISK_OFF"
        active_mode = "TREND_BREAKOUT_SHORT"
    elif dom_vol == "EXTREME (HANDBRAKE)":
        dom_risk = "DEFENSIVE_LOCK"
        active_mode = "STAY_IN_CASH"
    else:
        dom_risk = "SELECTIVE / RANGING"
        active_mode = "SELECTIVE_REGIME_SWITCHER"

    global_wrong_about = (
        f"현재 200시간선 위 코인 비율(시장 폭)은 {breadth_pct:.0%}, BTC 24시간 수익률은 {btc_ret_24:+.2%}입니다. "
        "만약 거시 지표 발표나 거래소 연쇄 청산이 발생하면 모든 알트코인 상관계수가 1로 수렴할 수 있으므로, "
        "Potato-Shield가 강제하는 1회 손실 캡(1.0%)과 일일 킬스위치(-2.0%)를 절대 해제하지 마십시오."
    )

    return {
        "dominant_trend_regime": dom_trend,
        "volatility_state": dom_vol,
        "risk_appetite": dom_risk,
        "market_breadth_pct": breadth_pct,
        "btc_24h_return_pct": btc_ret_24,
        "active_strategy_mode": world_verdict.recommended_strategy_family,
        "confidence_score": 78,
        "what_could_be_wrong": global_wrong_about,
        "world_monitor": {
            "global_regime_code": world_verdict.global_regime_code,
            "recommended_strategy_family": world_verdict.recommended_strategy_family,
            "fear_greed_value": world_verdict.fear_greed_value,
            "fear_greed_prev_value": world_verdict.fear_greed_prev_value,
            "fear_greed_delta": world_verdict.fear_greed_delta,
            "fear_greed_label": world_verdict.fear_greed_label,
            "realtime_intraday_fg": world_verdict.realtime_intraday_fg,
            "realtime_intraday_label": world_verdict.realtime_intraday_label,
            "retail_long_pct": world_verdict.retail_long_pct,
            "whale_long_pct": world_verdict.whale_long_pct,
            "gold_24h_return_pct": world_verdict.gold_24h_return_pct,
            "safe_haven_divergence_pct": world_verdict.safe_haven_divergence_pct,
            "shock_score": world_verdict.shock_score,
            "macro_event_score": world_verdict.macro_event_score,
            "bullish_score": world_verdict.bullish_score,
            "allow_long": world_verdict.allow_long,
            "allow_short": world_verdict.allow_short,
            "korean_summary": world_verdict.korean_summary,
            "top_headlines": world_verdict.top_headlines,
        },
        "vibe_diagnostics": {
            "correlation": corr_diag,
            "alpha_vitality": vitality_diag,
        },
        "coin_regimes": coin_rows,
        "actionable_signals": debate_logs[:5],  # Top 5 debated coins always shown in Meeting Room!
    }


def write_hermes_discord_briefing(
    regime_scan: Dict[str, Any],
    strategy_comparison: Dict[str, Any],
    testnet_balance: Dict[str, Any],
    output_md_path: str = "reports/latest_discord_briefing.md",
) -> str:
    """
    Generate a clean Korean Markdown briefing that Hermes Agent (에르메스 에이전트)
    can read and send directly to the user's Discord channel.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sw = strategy_comparison.get("3_regime_switched_multi_strategy", {})
    wy = strategy_comparison.get("4_wonyotti_grid_regime_hybrid", {})
    wm = regime_scan.get("world_monitor", {})

    lines = [
        f"# 🏛️ [Q-Stack 24/7 정례 브리핑 & AI 위원회 회의록] ({now_str})",
        "",
        "## 0. 🌍 WorldMonitor 실시간 세계 정세·거시·뉴스 레이더 (전략 선택 근거)",
        f"- **글로벌 정세 종합 판독**: {wm.get('korean_summary', '')}",
        f"- **공포·탐욕 지수**: `{wm.get('fear_greed_value', 50)} ({wm.get('fear_greed_label', 'Neutral')})` | **금(PAXG) 24h**: `{wm.get('gold_24h_return_pct', 0.0):+.2%}` | **허용 방향**: `롱({'O' if wm.get('allow_long') else 'X'}) / 숏({'O' if wm.get('allow_short') else 'X'})`",
    ]
    for hl in wm.get("top_headlines", [])[:3]:
        lines.append(f"  - `[{hl['tag']}]` {hl['title']}")

    vibe = regime_scan.get("vibe_diagnostics", {})
    corr_d = vibe.get("correlation", {})
    vit_d = vibe.get("alpha_vitality", {})
    lines.extend(
        [
            "",
            "## 1. 🌤️ 현재 3축 차트 장세 & Vibe-Trading 동조화/수명 판독 (Weather & Vitality Radar)",
            f"- **지배적 추세**: `{regime_scan['dominant_trend_regime']}` | **변동성**: `{regime_scan['volatility_state']}`",
            f"- **시장 폭 (200MA 상회 비율)**: `{regime_scan['market_breadth_pct']:.0%}` | **BTC 24h**: `{regime_scan['btc_24h_return_pct']:+.2%}`",
            f"- **🔗 코인 동조화 경보기**: {corr_d.get('korean_summary', '')} (동적 허용 슬롯: **`{corr_d.get('recommended_max_slots', 3)}슬롯`**)",
            f"- **🧬 전략 수명 감지기 (30일 IC)**: {vit_d.get('korean_summary', '')}",
            f"- **최종 선택된 전략 무기**: **`{regime_scan['active_strategy_mode']}`**",
            f"- **테스트넷 계좌 상태**: `{testnet_balance['mode']}` | 가용 잔고: **`${testnet_balance['available_balance_usdt']:,.2f} USDT`**",
            "",
            "## 2. ⚖️ 상위 주목 종목 AI 위원회 찬반 회의록 (Autonomous Debate Log)",
        ]
    )

    for idx, d in enumerate(regime_scan["actionable_signals"][:3], 1):
        lines.extend(
            [
                f"### [{idx}] {d['symbol']} ({d['side']} 검토 | 배정 전략: `{d['strategy_fit']}`)",
                f"- {d['research_note']}",
                f"- {d['bull_thesis']}",
                f"- {d['bear_challenges'][0]}",
                f"- **최종 심판 결정**: {d['cio_verdict_text']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 3. 📊 360일 전략 연구소 검증 요약 (W1 × W2)",
            f"- **기본 3축 장세 스위칭 전략**: `{sw.get('trades', 0)}건` | 순수익 **`${sw.get('net_pnl_usd', 0.0):+,.2f}`** | `Gross/Cost = {sw.get('gross_to_cost_ratio', 0.0):.2f}배` | `MDD = {sw.get('mdd_pct', 0.0):.2%}`",
            f"- **🏆 워뇨띠 멀티캔들 거래량+사토시 상대강도 스위칭**: `{wy.get('trades', 0)}건` | 순수익 **`${wy.get('net_pnl_usd', 0.0):+,.2f}`** (`승률 {wy.get('win_rate', 0.0):.1%}`) | `Gross/Cost = {wy.get('gross_to_cost_ratio', 0.0):.2f}배` | `MDD = {wy.get('mdd_pct', 0.0):.2%}`",
            f"- **리스크 주의보 (What could we be wrong about?)**: {regime_scan['what_could_be_wrong']}",
        ]
    )

    content = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(content)

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if webhook_url:
        try:
            import requests
            requests.post(webhook_url, json={"content": content[:1950]}, timeout=5)
            print("  -> Sent Discord briefing automatically via DISCORD_WEBHOOK_URL!")
        except Exception as e:
            print(f"  -> Discord webhook warning: {e}")

    return content


def generate_6panel_dashboard_html(
    regime_scan: Dict[str, Any],
    strategy_comparison: Dict[str, Any],
    w2_details: List[str],
    testnet_ping: Dict[str, Any],
    testnet_balance: Dict[str, Any],
    recent_experiments: List[Dict[str, Any]],
    recent_trades: List[Dict[str, Any]],
    output_path: str = "reports/qstack_dashboard.html",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    coin_rows_html = ""
    for c in regime_scan["coin_regimes"]:
        t_color = (
            "#238636"
            if c["trend_regime"] == "TRENDING_UP"
            else ("#da3633" if c["trend_regime"] == "TRENDING_DOWN" else "#8b949e")
        )
        v_color = "#da3633" if c["volatility_state"] == "EXTREME" else "#58a6ff"
        coin_rows_html += f"""
        <tr>
          <td><strong>{c['symbol']}</strong></td>
          <td>${c['price']:,.4f}</td>
          <td style="color:{'#3fb950' if c['ret_24h_pct']>=0 else '#f85149'};">{c['ret_24h_pct']:+.2%}</td>
          <td style="color:{'#3fb950' if c['rel_strength_vs_btc']>=0 else '#f85149'};">{c['rel_strength_vs_btc']:+.2%}</td>
          <td><span style="background:{t_color};color:#fff;padding:2px 6px;border-radius:4px;font-size:11px;">{c['trend_regime']}</span></td>
          <td><span style="color:{v_color};font-weight:bold;">{c['volatility_state']}</span> ({c['vol_pct_rank']:.0%})</td>
          <td>{c['adx_14']:.1f} / {c['rsi_14']:.0f}</td>
          <td>{c['delta_ratio']:+.2f}</td>
          <td>{c['funding_rate_8h']:.4%}</td>
          <td><code>{c['strategy_fit']}</code> ({c['confidence']}%)</td>
        </tr>
        """

    strat_rows_html = ""
    labels_map = {
        "1_blind_mean_reversion": "① 무지성 역추세 24/7 가동 (Blind Mean-Reversion)",
        "2_blind_trend_breakout": "② 무지성 추세돌파 24/7 가동 (Blind Trend-Breakout)",
        "3_regime_switched_multi_strategy": "③ 3축 장세 판독 + 기본 멀티 전략 스위처 (audit_bot.xml)",
        "4_wonyotti_grid_regime_hybrid": "🏆 ④ 워뇨띠 멀티캔들 거래량+사토시 상대강도 스위처 (Wonyotti Mode)",
    }
    for k, v in strategy_comparison.items():
        net_val = v.get("net_pnl_usd", 0.0)
        row_style = 'style="background:#1c2d24;font-weight:bold;"' if net_val > 0 else ""
        strat_rows_html += f"""
        <tr {row_style}>
          <td>{labels_map.get(k, k)}</td>
          <td>{v.get('trades', 0)}건</td>
          <td>${v.get('gross_pnl_usd', 0.0):+,.2f}</td>
          <td>${v.get('total_cost_usd', 0.0):,.2f}</td>
          <td style="color:{'#3fb950' if net_val>0 else '#f85149'};font-size:14px;">${net_val:+,.2f}</td>
          <td>{v.get('gross_to_cost_ratio', 0.0):.2f}x (Net/Cost: {v.get('net_to_cost_ratio', 0.0):+.2f}x)</td>
          <td>{v.get('win_rate', 0.0):.1%}</td>
          <td>{v.get('mdd_pct', 0.0):.2%}</td>
        </tr>
        """

    signals_html = ""
    for sig in regime_scan["actionable_signals"]:
        badge_col = "#238636" if sig["eggbot_approved"] else "#8b949e"
        signals_html += f"""
        <div style="border:1px solid #30363d;border-left:4px solid {badge_col};padding:10px;margin-bottom:8px;border-radius:4px;background:#0d1117;">
          <div style="font-size:13px;"><strong>{sig['symbol']} [{sig['side']} 검토]</strong> | 배정: <code>{sig['strategy_fit']}</code> | 진입가: <code>${sig['entry_price']:,.4f}</code> | 손절가: <code>${sig['stop_price']:,.4f}</code> | 사이징: <code>${sig['notional_usd']:,.2f} ({sig['leverage']:.2f}x)</code></div>
          <div style="font-size:12px;color:#c9d1d9;margin-top:4px;">{sig['research_note']}</div>
          <div style="font-size:12px;color:#3fb950;margin-top:2px;">{sig['bull_thesis']}</div>
          <div style="font-size:12px;color:#f0883e;margin-top:2px;">{sig['bear_challenges'][0]}</div>
          <div style="font-size:12.5px;color:#58a6ff;font-weight:bold;margin-top:4px;">{sig['cio_verdict_text']}</div>
        </div>
        """

    pos_mgr_view = PositionLifecycleManager("data/crumb_memory.db")
    active_positions = pos_mgr_view.get_active_positions()
    active_pos_rows = ""
    if active_positions:
        for ap in active_positions:
            tp_status = "✅ 1차 50% 익절완료 (본절잠금)" if int(ap["tp1_hit"]) == 1 else f"대기 (${ap['tp1_price']:,.4f})"
            active_pos_rows += f"""
            <tr style="background:#1c2d24;">
              <td><strong>{ap['symbol']}</strong> ({ap['side']})</td>
              <td><code>{ap['margin_mode']}</code> (${ap['slot_margin_usd']:,.0f} 슬롯 / {ap['leverage']:.2f}배)</td>
              <td>${ap['entry_price']:,.4f}</td>
              <td style="color:#3fb950;">{tp_status}</td>
              <td style="color:#f85149;">${ap['stop_price']:,.4f}</td>
              <td>${ap['notional_usd']:,.2f}</td>
              <td><strong>${ap['unrealized_pnl_usd']:+,.2f}</strong> (확정: ${ap['realized_pnl_usd']:+,.2f})</td>
            </tr>
            """
    else:
        active_pos_rows = '<tr><td colspan="7" style="text-align:center;color:#8b949e;">현재 보유 중인 포지션이 없습니다 (5개 슬롯 모두 대기 중)</td></tr>'

    trade_rows_html = ""
    for tr in recent_trades[:8]:
        trade_rows_html += f"""
        <tr>
          <td>{tr['recorded_at']}</td>
          <td><strong>{tr['symbol']}</strong> ({tr['side']})</td>
          <td><code>{tr['sub_strategy']}</code></td>
          <td>${tr['entry_price']:,.4f}</td>
          <td>${tr['stop_price']:,.4f}</td>
          <td>${tr['notional_usd']:,.2f} ({tr['leverage']:.2f}x)</td>
          <td><code>{tr['eggbot_verdict']}</code></td>
          <td><code>{tr['order_status']}</code></td>
        </tr>
        """

    exp_rows_html = ""
    for ex in recent_experiments[:6]:
        exp_rows_html += f"""
        <tr>
          <td>{ex['created_at']}</td>
          <td><code>{ex['strategy_name']}</code></td>
          <td><code>{ex['config_sha256']}</code></td>
          <td>{ex['trades_count']}</td>
          <td>${ex['gross_pnl_usd']:+,.2f}</td>
          <td>${ex['total_cost_usd']:,.2f}</td>
          <td style="color:{'#3fb950' if ex['net_pnl_usd']>0 else '#f85149'};">${ex['net_pnl_usd']:+,.2f}</td>
          <td>{ex['w1_status']}</td>
        </tr>
        """

    wm = regime_scan.get("world_monitor", {})
    vibe_d = regime_scan.get("vibe_diagnostics", {})
    corr_html = vibe_d.get("correlation", {}).get("korean_summary", "")
    vit_html = vibe_d.get("alpha_vitality", {}).get("korean_summary", "")
    wm_headlines_html = "".join(
        f"<li><code>[{h['tag']}]</code> ({h['source']}) {h['title']}</li>"
        for h in wm.get("top_headlines", [])[:4]
    )

    approved_list = [s for s in regime_scan["actionable_signals"] if s["eggbot_approved"]]
    if approved_list:
        top_act = approved_list[0]
        easy_headline = (
            f"🟢 [진입 승인 & 주문 실행] 지금 봇이 <strong>{top_act['symbol']} ({top_act['side']} / 하락베팅)</strong> 타점을 포착해 "
            f"안전 수량(<strong>${top_act['notional_usd']:,.2f} USDT, 레버리지 {top_act['leverage']:.2f}배</strong>)으로 주문을 넣었습니다! "
            f"(자동 손절가: <code>${top_act['stop_price']:,.4f}</code> — 틀려도 내 시드의 1%인 $10만 잃도록 잠금 완료)"
        )
        headline_bg = "#1c2d24"
        headline_border = "#3fb950"
    else:
        easy_headline = (
            "🟡 [현금 관망 중] 지금 시장은 조건이 완벽하지 않아 봇이 내 돈을 지키며 100% 현금으로 대기하고 있습니다. "
            "(억지로 매매하지 않고 확실한 타점만 기다리는 중)"
        )
        headline_bg = "#262115"
        headline_border = "#d29922"

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="300">
<title>Q-Stack 24시간 자율 AI 퀀트 본부 (한글 대시보드)</title>
<style>
  body {{ background:#090d16; color:#c9d1d9; font-family:-apple-system,BlinkMacSystemFont,"Malgun Gothic","Segoe UI",sans-serif; margin:20px; line-height:1.5; }}
  h1, h2, h3 {{ color:#f0f6fc; margin:0 0 10px 0; }}
  .grid {{ display:grid; grid-template-columns:1.25fr 0.75fr; gap:16px; margin-bottom:16px; }}
  .panel {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; }}
  .panel-full {{ grid-column:1 / -1; background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin-bottom:16px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }}
  th, td {{ border:1px solid #30363d; padding:8px 10px; text-align:left; }}
  th {{ background:#21262d; color:#58a6ff; }}
  .kpi-bar {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:12px; }}
  .kpi {{ background:#0d1117; border:1px solid #30363d; border-radius:6px; padding:10px 14px; flex:1; min-width:150px; }}
  .kpi-title {{ font-size:12px; color:#8b949e; }}
  .kpi-val {{ font-size:18px; font-weight:bold; color:#f0f6fc; margin-top:4px; }}
</style>
</head>
<body>
  <h1>🏛️ Q-Stack: 나만의 24시간 자율 AI 퀀트 연구소 &amp; 자동매매 본부</h1>
  <p style="color:#8b949e;margin-top:0;">최근 자동 업데이트: <code>{now_str}</code> | 설정 무결성 해시: <code>{CONFIG.config_hash()}</code> | 디스코드 브리핑 파일: <code>reports/latest_discord_briefing.md</code></p>

  <!-- 초보자용 3초 요약 신호등 -->
  <div style="background:{headline_bg};border:2px solid {headline_border};border-radius:8px;padding:16px;margin-bottom:16px;font-size:15px;color:#f0f6fc;">
    <div style="font-size:13px;color:#8b949e;margin-bottom:4px;">🚦 <strong>초보자용 3초 한눈에 보기 (지금 봇이 무슨 일을 했나요?)</strong></div>
    {easy_headline}
  </div>

  <!-- PANEL 01: WORLDMONITOR + 3-AXIS REGIME OVERVIEW -->
  <div class="panel-full">
    <h2>① 🌍 WorldMonitor 세계 정세·거시·뉴스 레이더 &amp; 🌤️ 3축 차트 장세 판독기</h2>
    <div style="font-size:13.5px;background:#121d2f;padding:12px;border-radius:6px;border-left:4px solid #58a6ff;margin-bottom:12px;">
      <div><strong>🌍 WorldMonitor 실시간 세계 정세 판독 &amp; 전략 선택 이유:</strong> {wm.get('korean_summary', '')}</div>
      <div style="margin-top:6px;font-size:12.5px;color:#8b949e;">
        📊 <strong>일간 공포탐욕(Alternative.me):</strong> <code>{wm.get('fear_greed_value', 50)}점 (전일 {wm.get('fear_greed_prev_value', 68)}점 대비 {wm.get('fear_greed_delta', -18):+d}p)</code> |
        ⚡ <strong>실시간 장중 체감 지수:</strong> <code style="color:#f0883e;">{wm.get('realtime_intraday_fg', 38)}점 ({wm.get('realtime_intraday_label', 'Fear')})</code> |
        🐋 <strong>바이낸스 선물 롱 비율:</strong> <code>개미 {wm.get('retail_long_pct', 0.688):.1%} vs 고래 {wm.get('whale_long_pct', 0.604):.1%}</code> |
        🪙 <strong>금(PAXG) 24h:</strong> <code>{wm.get('gold_24h_return_pct', 0.0):+.2%}</code> |
        🚦 <strong>거시 방향:</strong> <code>롱({'허용' if wm.get('allow_long') else '차단'}) / 숏({'허용' if wm.get('allow_short') else '차단'})</code>
      </div>
      <ul style="margin:6px 0 0 0;padding-left:18px;font-size:12px;color:#c9d1d9;">
        {wm_headlines_html}
      </ul>
      <div style="margin-top:8px;padding-top:8px;border-top:1px solid #30363d;font-size:12.5px;color:#e6edf3;">
        <div><strong>🔗 Vibe-Trading 코인 동조화 경보기:</strong> {corr_html}</div>
        <div style="margin-top:4px;"><strong>🧬 Vibe-Trading 30일 전략 수명 감지기 (IC):</strong> {vit_html}</div>
      </div>
    </div>
    <div class="kpi-bar">
      <div class="kpi"><div class="kpi-title">현재 시장 큰 흐름 (추세)</div><div class="kpi-val" style="color:#58a6ff;">{regime_scan['dominant_trend_regime']}</div></div>
      <div class="kpi"><div class="kpi-title">가격 출렁임 정도 (변동성)</div><div class="kpi-val" style="color:#f0883e;">{regime_scan['volatility_state']}</div></div>
      <div class="kpi"><div class="kpi-title">공격/방어 모드</div><div class="kpi-val" style="color:#3fb950;">{regime_scan['risk_appetite']}</div></div>
      <div class="kpi"><div class="kpi-title">상승 중인 코인 비율 (시장 폭)</div><div class="kpi-val">{regime_scan['market_breadth_pct']:.0%} (비트코인 24h: {regime_scan['btc_24h_return_pct']:+.2%})</div></div>
      <div class="kpi"><div class="kpi-title">WorldMonitor × 차트 최종 선택 무기</div><div class="kpi-val" style="font-size:14px;color:#a371f7;">{regime_scan['active_strategy_mode']}</div></div>
    </div>
    <div style="font-size:13px;background:#0d1117;padding:10px;border-radius:6px;border-left:4px solid #a371f7;">
      <strong>🔍 비판 봇의 리스크 경고 (우리가 뭘 조심해야 하나요?):</strong> {regime_scan['what_could_be_wrong']}
    </div>
  </div>

  <div class="grid">
    <!-- PANEL 03: AUTONOMOUS DEBATE LOG & SIGNAL MONITOR -->
    <div class="panel">
      <h2>② ⚖️ AI 위원회 찬반 회의록 &amp; 24시간 자동 투자 결정 (왜 샀고 왜 안 샀나요?)</h2>
      {signals_html}
    </div>

    <!-- PANEL 05: POTATO-SHIELD RISK & BINANCE TESTNET TERMINAL -->
    <div class="panel">
      <h2>③ 🛡️ 내 투자 규칙 (`CONFIG` &amp; `user_trading_rules.json` 통합) &amp; 실시간 격리 포지션 현황</h2>
      <div class="kpi-bar">
        <div class="kpi"><div class="kpi-title">계좌 가용 잔고</div><div class="kpi-val">${testnet_balance['available_balance_usdt']:,.2f} USDT</div></div>
        <div class="kpi"><div class="kpi-title">W2 통합 격리 슬롯</div><div class="kpi-val" style="color:#3fb950;">{len(active_positions)} / 3 슬롯 사용중</div></div>
        <div class="kpi"><div class="kpi-title">바이낸스 통신 속도</div><div class="kpi-val">{testnet_ping.get('latency_ms', -1)} ms</div></div>
      </div>
      <div style="font-size:12.5px;background:#0d1117;padding:8px 10px;border-radius:6px;margin-bottom:10px;border-left:3px solid #3fb950;">
        <strong>📌 내 설정 규칙 (W2 단일 통합):</strong> ① 최대 3슬롯 격리(슬롯 증거금 20% = <code>$200</code>, 총 노출 <code>1.0x</code> 이하) | ② 손절가: <code>ATR 자동 (최대 -2.5% 칼손절 천장, 1회 최대손실 1%=$10 고정)</code> | ③ 익절가: <code>+1.8% 도달 시 50% 분할익절 + 나머지 50% 본절잠금</code> | ④ 일일 킬스위치: <code>하루 -2.0% 손실 시 매매 중단</code>
      </div>
      <h3 style="margin-top:10px;font-size:13.5px;color:#3fb950;">🔥 현재 보유 중인 실시간 포지션 (자동 익절·본절잠금·손절 감시 중)</h3>
      <table>
        <tr><th>코인 (방향)</th><th>격리 슬롯 (배율)</th><th>진입 단가</th><th>1차 50% 익절가 (+1.8%)</th><th>현재 칼손절 단가</th><th>총 포지션 규모</th><th>실시간 손익</th></tr>
        {active_pos_rows}
      </table>
      <h3 style="margin-top:14px;font-size:13.5px;">📋 최근 봇이 결정한 주문 내역 (자동 장부)</h3>
      <table>
        <tr><th>시간</th><th>코인 (방향)</th><th>사용한 무기</th><th>진입가</th><th>자동 손절가</th><th>투입 금액 (배율)</th><th>심판 판정</th><th>주문 상태</th></tr>
        {trade_rows_html}
      </table>
    </div>
  </div>

  <!-- PANEL 04: STRATEGY LAB & W1 REJECTION MACHINE -->
  <div class="panel-full">
    <h2>④ 🏆 전략 연구소 성적표: 지난 1년(360일) 동안 어떤 전략이 돈을 제일 잘 벌었나요? (수수료·슬리피지 모두 차감 후 실제 순수익)</h2>
    <table>
      <tr>
        <th>전략 운용 모드 (초기 자금 $1,000 기준, 360일 실데이터)</th>
        <th>총 매매 횟수</th>
        <th>수수료 빼기 전 수익</th>
        <th>총 수수료·슬리피지 비용</th>
        <th>💎 최종 순수익 (내 주머니에 남는 돈)</th>
        <th>비용 대비 수익 배수</th>
        <th>승률</th>
        <th>최대 계좌 낙폭 (MDD)</th>
      </tr>
      {strat_rows_html}
    </table>
  </div>

  <!-- PANEL 02: 10-COIN LIQUIDITY, RESIDUAL & REGIME MAP -->
  <div class="panel-full">
    <h2>⑤ 🗺️ 10개 메이저 코인 실시간 정밀 스캔 표 (참고용 세부 지표)</h2>
    <table>
      <tr>
        <th>코인명</th>
        <th>현재 가격</th>
        <th>24시간 등락률</th>
        <th>비트코인 대비 강도(사토시)</th>
        <th>추세 상태</th>
        <th>변동성 상태</th>
        <th>추세강도(ADX) / 과열도(RSI)</th>
        <th>매수·매도 수급(Delta)</th>
        <th>8시간 펀딩비</th>
        <th>봇이 배정한 무기 (신뢰도)</th>
      </tr>
      {coin_rows_html}
    </table>
  </div>

  <!-- PANEL 06: CRUMB PERSISTENT SQLITE RESEARCH JOURNAL -->
  <div class="panel-full">
    <h2>⑥ 🍞 과거 연구 실험 기록 보관소 (`data/crumb_memory.db` — 실패한 전략을 다시 쓰지 않기 위한 기억 장치)</h2>
    <table>
      <tr>
        <th>기록 시각</th>
        <th>실험한 전략 이름</th>
        <th>설정 해시</th>
        <th>거래 수</th>
        <th>세전 수익</th>
        <th>총 비용</th>
        <th>최종 순수익</th>
        <th>합격/폐기 판정</th>
      </tr>
      {exp_rows_html}
    </table>
  </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def run_one_complete_cycle(execute_on_testnet: bool = False) -> Dict[str, Any]:
    print("=" * 76)
    print("[Q-Stack Cycle] 1/6 Running W2 Deterministic Risk & Leakage Gate...")
    print("=" * 76)
    w2_pass, w2_details = run_w2_unit_tests()
    if not w2_pass:
        raise RuntimeError("W2 Risk Gate Failed! Halting cycle.")

    print("\n" + "=" * 76)
    print("[Q-Stack Cycle] 2/6 Pinging Binance Futures Testnet & Checking Balance...")
    print("=" * 76)
    testnet_client = BinanceFuturesTestnetClient()
    ping_res = testnet_client.ping_and_latency_ms()
    bal_res = testnet_client.get_usdt_balance(CONFIG.initial_capital)
    print(f"  -> Testnet Reachable: {ping_res.get('reachable')} | Latency: {ping_res.get('latency_ms')} ms")
    print(f"  -> Account Mode: {bal_res['mode']} | Balance: ${bal_res['available_balance_usdt']:,.2f} USDT")

    print("\n" + "=" * 76)
    print("[Q-Stack Cycle] 3/6 Fetching 10-Coin Universe + BTC Data & Running 3-Axis Regime Scan...")
    print("=" * 76)
    fetcher = BinanceDataFetcher(CONFIG, data_dir="data/raw")
    btc_1h = fetcher.fetch_klines("BTCUSDT", "1h", days=360, refresh_tail=True)
    univ_1h = {sym: fetcher.fetch_klines(sym, "1h", days=360, refresh_tail=True) for sym in CONFIG.target_universe}
    fund_360 = {sym: fetcher.fetch_funding_rates(sym, days=360) for sym in CONFIG.target_universe}

    eggbot = DrEggbotAuditor()
    risk_engine = DeterministicRiskEngine(CONFIG, CostModel(CONFIG))
    risk_engine.equity = float(bal_res["available_balance_usdt"])

    regime_scan = build_autonomous_debate_and_regime_scan(
        univ_1h, btc_1h, fund_360, eggbot, risk_engine
    )
    print(
        f"  -> Dominant Regime: Trend={regime_scan['dominant_trend_regime']} | "
        f"Vol={regime_scan['volatility_state']} | Breadth={regime_scan['market_breadth_pct']:.0%} | "
        f"Mode={regime_scan['active_strategy_mode']}"
    )

    print("\n" + "=" * 76)
    print("[Q-Stack Cycle] 4/6 Running Strategy Lab (Comparing Single Strategies vs Regime Switcher)...")
    print("=" * 76)
    strat_comp_cache = "reports/strategy_comparison.json"
    if os.path.exists(strat_comp_cache) and (time.time() - os.path.getmtime(strat_comp_cache)) < 21600:
        with open(strat_comp_cache, "r", encoding="utf-8") as f:
            strat_comp = json.load(f)
        print("  -> Loaded cached 360d Strategy Comparison (< 6h old) for fast 15m daemon cycle.")
    else:
        strat_comp = run_regime_switching_study(univ_1h, btc_1h, fund_360)

    print("\n" + "=" * 76)
    print("[Q-Stack Cycle] 5/6 Recording Autonomous Debate Decisions & Testnet Orders to Crumb DB...")
    print("=" * 76)
    crumb = CrumbMemoryDB("data/crumb_memory.db")
    crumb.log_regime_snapshot(
        dominant_trend_regime=regime_scan["dominant_trend_regime"],
        volatility_state=regime_scan["volatility_state"],
        risk_appetite=regime_scan["risk_appetite"],
        market_breadth_pct=regime_scan["market_breadth_pct"],
        btc_24h_return_pct=regime_scan["btc_24h_return_pct"],
        active_strategy_mode=regime_scan["active_strategy_mode"],
        confidence_score=regime_scan["confidence_score"],
        what_could_be_wrong=regime_scan["what_could_be_wrong"],
        coin_regimes=regime_scan["coin_regimes"],
    )

    pos_mgr = PositionLifecycleManager("data/crumb_memory.db")
    latest_prices_map = {c["symbol"]: float(c["price"]) for c in regime_scan["coin_regimes"]}
    coin_regimes_map = {c["symbol"]: str(c["trend_regime"]) for c in regime_scan["coin_regimes"]}
    lifecycle_events = pos_mgr.sync_and_manage_positions(
        latest_prices=latest_prices_map,
        coin_regimes_map=coin_regimes_map,
        testnet_client=testnet_client,
        execute_on_testnet=execute_on_testnet,
    )
    for ev in lifecycle_events:
        print(f"  -> [Position Lifecycle] {ev}")

    latest_bar_tag = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    dyn_max_slots = (
        regime_scan.get("vibe_diagnostics", {})
        .get("correlation", {})
        .get("recommended_max_slots", CONFIG.max_concurrent_positions)
    )
    for sig in regime_scan["actionable_signals"][:3]:
        if sig["eggbot_approved"]:
            can_open, gate_reason = pos_mgr.can_open_position(
                sig["symbol"], dynamic_max_slots=dyn_max_slots
            )
            if not can_open:
                order_status = gate_reason
                order_id = "NONE"
            else:
                det_trade_id = f"{sig['symbol']}_{sig['side']}_{latest_bar_tag}"
                ord_res = testnet_client.place_testnet_bracket_order(
                    symbol=sig["symbol"],
                    side=sig["side_int"],
                    quantity=sig["quantity"],
                    stop_price=sig["stop_price"],
                    tp1_price=sig["tp1_price"],
                    partial_tp_ratio=0.50,
                    leverage=int(max(1, round(sig["leverage"]))),
                    trade_id=det_trade_id,
                    execute_on_testnet=execute_on_testnet,
                )
                order_status = ord_res["status"]
                order_id = ord_res["order_id"]
                if "FAILED" not in order_status:
                    pos_mgr.open_or_record_position(
                        symbol=sig["symbol"],
                        side=sig["side"],
                        side_int=sig["side_int"],
                        sub_strategy=sig["sub_strategy"],
                        plan=sig["slot_plan"],
                        entry_price=sig["entry_price"],
                        execution_mode=bal_res["mode"],
                        dynamic_max_slots=dyn_max_slots,
                    )
        else:
            order_status = "WAIT_FOR_TRIGGER"
            order_id = "NONE"

        crumb.log_trade_event(
            execution_mode=bal_res["mode"],
            symbol=sig["symbol"],
            side=sig["side"],
            regime_label=regime_scan["dominant_trend_regime"],
            sub_strategy=sig["sub_strategy"],
            entry_price=sig["entry_price"],
            stop_price=sig["stop_price"],
            notional_usd=sig["notional_usd"],
            leverage=sig["leverage"],
            eggbot_verdict=sig["eggbot_verdict"],
            eggbot_bear_case=sig["bear_challenges"][0],
            order_status=order_status,
            order_id=order_id,
        )

    print("\n" + "=" * 76)
    print("[Q-Stack Cycle] 6/6 Rendering 6-Panel Dashboard & Hermes Discord Briefing...")
    print("=" * 76)
    write_hermes_discord_briefing(
        regime_scan=regime_scan,
        strategy_comparison=strat_comp,
        testnet_balance=bal_res,
        output_md_path="reports/latest_discord_briefing.md",
    )
    generate_6panel_dashboard_html(
        regime_scan=regime_scan,
        strategy_comparison=strat_comp,
        w2_details=w2_details,
        testnet_ping=ping_res,
        testnet_balance=bal_res,
        recent_experiments=crumb.get_recent_experiments(10),
        recent_trades=crumb.get_recent_trades(10),
        output_path="reports/qstack_dashboard.html",
    )
    print("  -> Dashboard ready at: C:\\Users\\user\\Desktop\\qunat\\reports\\qstack_dashboard.html")
    print("  -> Discord Briefing ready at: C:\\Users\\user\\Desktop\\qunat\\reports\\latest_discord_briefing.md")
    return regime_scan


def start_dashboard_http_server(port: int = 8080) -> None:
    class DashboardHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.path = "/reports/qstack_dashboard.html"
            return super().do_GET()

    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"[Suki Web Server] Serving 6-Panel Dashboard at http://0.0.0.0:{port}")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Q-Stack End-to-End 1-Cycle & 24/7 Testnet Daemon")
    parser.add_argument("--daemon", action="store_true", help="Run continuously every 15 minutes")
    parser.add_argument("--port", type=int, default=0, help="Serve web dashboard on HTTP port (e.g. 8080)")
    parser.add_argument("--live-testnet", action="store_true", help="Transmit approved orders to Binance Testnet")
    args = parser.parse_args()

    if args.port > 0:
        t = threading.Thread(target=start_dashboard_http_server, args=(args.port,), daemon=True)
        t.start()

    if not args.daemon:
        run_one_complete_cycle(execute_on_testnet=args.live_testnet)
    else:
        while True:
            try:
                run_one_complete_cycle(execute_on_testnet=args.live_testnet)
            except Exception as e:
                print(f"[Daemon Error] {e}")
            print("[Daemon] Sleeping 900s (15m) until next scan...")
            time.sleep(900)


if __name__ == "__main__":
    main()
