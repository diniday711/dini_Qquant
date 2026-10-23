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
from engines.regime_switcher import compute_adx_causal, run_regime_switching_study
from engines.risk_engine import DeterministicRiskEngine
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

        stop_dist = max(0.015, (2.2 * atr_val) / close_px)
        stop_px = close_px * (1.0 - stop_dist) if cand_side == 1 else close_px * (1.0 + stop_dist)
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

        final_approved = bool(is_exact_trigger and audit.approved and sizing.approved)

        # Build Korean Multi-Persona Debate Log (회의록)
        research_note = (
            f"[리서치 봇 Hatcha] {sym} 현재가 ${close_px:,.4f} (24h {float(last['ret_24']):+.2%}, "
            f"BTC 대비 상대강도 {rel_str:+.2%}). 장세={trend_state}/{vol_state}, ADX={adx_val:.1f}, RSI={rsi_val:.0f}, "
            f"수급 Delta={delta_r:+.2f}, 펀딩비={latest_funding:.4%}."
        )
        pro_opinion = (
            f"[찬성 봇 Bull/Trend] 현재 배정 전략은 `{strategy_fit}`입니다. "
            f"{'LONG' if cand_side==1 else 'SHORT'} 방향 진입 시 손절폭 {stop_dist:.2%}(${stop_px:,.4f}), "
            f"비용 대비 기대수익비 {expected_rr_cost:.1f}배로 손익비가 우수합니다."
        )
        con_opinion = (
            f"[비판 봇 Dr.Eggbot] 반론 점검: {'; '.join(audit.bear_case_challenges)} | "
            f"{'돌파/밴드 타점 도달 완료!' if is_exact_trigger else '아직 48h 돌파선/밴드 상하단 타점에 완전히 닿지 않음(대기 필요).'}"
        )
        cio_decision = (
            f"✅ [최종 승인 - 즉시 주문 집행] 타점 도달 및 리스크 감사 통과! {sizing.notional_usd:,.2f} USDT ({sizing.effective_leverage:.2f}x) 주문 전송."
            if final_approved
            else f"⏸️ [관망/기각 (ABSTAIN)] {'타점 미도달(대기)' if not is_exact_trigger else audit.verdict_code} 사유로 진입 보류."
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
                "notional_usd": sizing.notional_usd,
                "quantity": sizing.quantity,
                "leverage": sizing.effective_leverage,
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
        "active_strategy_mode": active_mode,
        "confidence_score": 78,
        "what_could_be_wrong": global_wrong_about,
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

    lines = [
        f"# 🏛️ [Q-Stack 24/7 정례 브리핑 & AI 위원회 회의록] ({now_str})",
        "",
        "## 1. 🌤️ 현재 3축 시장 장세 판독 (Weather Radar)",
        f"- **지배적 추세**: `{regime_scan['dominant_trend_regime']}` | **변동성**: `{regime_scan['volatility_state']}`",
        f"- **시장 폭 (200MA 상회 비율)**: `{regime_scan['market_breadth_pct']:.0%}` | **BTC 24h**: `{regime_scan['btc_24h_return_pct']:+.2%}`",
        f"- **자동 선택된 전략 모드**: **`{regime_scan['active_strategy_mode']}`**",
        f"- **테스트넷 계좌 상태**: `{testnet_balance['mode']}` | 가용 잔고: **`${testnet_balance['available_balance_usdt']:,.2f} USDT`**",
        "",
        "## 2. ⚖️ 상위 주목 종목 AI 위원회 찬반 회의록 (Autonomous Debate Log)",
    ]

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

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="300">
<title>Q-Stack 6-Panel AI Quant Workstation (Suki Dashboard)</title>
<style>
  body {{ background:#090d16; color:#c9d1d9; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",monospace; margin:20px; line-height:1.45; }}
  h1, h2, h3 {{ color:#f0f6fc; margin:0 0 10px 0; }}
  .grid {{ display:grid; grid-template-columns:1.25fr 0.75fr; gap:16px; margin-bottom:16px; }}
  .panel {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; }}
  .panel-full {{ grid-column:1 / -1; background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin-bottom:16px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; margin-top:8px; }}
  th, td {{ border:1px solid #30363d; padding:7px 9px; text-align:left; }}
  th {{ background:#21262d; color:#58a6ff; }}
  .kpi-bar {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:12px; }}
  .kpi {{ background:#0d1117; border:1px solid #30363d; border-radius:6px; padding:10px 14px; flex:1; min-width:150px; }}
  .kpi-title {{ font-size:11px; color:#8b949e; text-transform:uppercase; }}
  .kpi-val {{ font-size:18px; font-weight:bold; color:#f0f6fc; margin-top:4px; }}
</style>
</head>
<body>
  <h1>🏛️ Q-Stack: 1-Person 24/7 AI Quant Desk &amp; Regime Switcher (6-Panel Terminal)</h1>
  <p style="color:#8b949e;margin-top:0;">Last Updated: <code>{now_str}</code> | Config SHA-256: <code>{CONFIG.config_hash()}</code> | Hermes Discord Briefing: <code>reports/latest_discord_briefing.md</code></p>

  <!-- PANEL 01: MARKET & 3-AXIS REGIME OVERVIEW -->
  <div class="panel-full">
    <h2>PANEL 01 — 🌤️ 3-Axis Market Regime Radar (audit_bot.xml: Trend × Volatility × Breadth)</h2>
    <div class="kpi-bar">
      <div class="kpi"><div class="kpi-title">Dominant Trend</div><div class="kpi-val" style="color:#58a6ff;">{regime_scan['dominant_trend_regime']}</div></div>
      <div class="kpi"><div class="kpi-title">Volatility State</div><div class="kpi-val" style="color:#f0883e;">{regime_scan['volatility_state']}</div></div>
      <div class="kpi"><div class="kpi-title">Risk Appetite</div><div class="kpi-val" style="color:#3fb950;">{regime_scan['risk_appetite']}</div></div>
      <div class="kpi"><div class="kpi-title">Market Breadth (&gt;200 MA)</div><div class="kpi-val">{regime_scan['market_breadth_pct']:.0%} (BTC 24h: {regime_scan['btc_24h_return_pct']:+.2%})</div></div>
      <div class="kpi"><div class="kpi-title">Active Strategy Mode</div><div class="kpi-val" style="color:#a371f7;">{regime_scan['active_strategy_mode']}</div></div>
    </div>
    <div style="font-size:12.5px;background:#0d1117;padding:10px;border-radius:6px;border-left:4px solid #a371f7;">
      <strong>🔍 Final Check — WHAT COULD I BE WRONG ABOUT?:</strong> {regime_scan['what_could_be_wrong']}
    </div>
  </div>

  <!-- PANEL 02: 10-COIN LIQUIDITY, RESIDUAL & REGIME MAP -->
  <div class="panel-full">
    <h2>PANEL 02 — 🗺️ 10-Coin Universe Regime &amp; Strategy Fit Map (Hatcha Intelligence)</h2>
    <table>
      <tr>
        <th>Symbol</th>
        <th>Price</th>
        <th>24h Return</th>
        <th>Rel Strength vs BTC</th>
        <th>Trend Axis</th>
        <th>Volatility Axis (Rank)</th>
        <th>ADX / RSI</th>
        <th>Candle Delta</th>
        <th>8h Funding</th>
        <th>Mapped Strategy Fit (Conf)</th>
      </tr>
      {coin_rows_html}
    </table>
  </div>

  <div class="grid">
    <!-- PANEL 03: AUTONOMOUS DEBATE LOG & SIGNAL MONITOR -->
    <div class="panel">
      <h2>PANEL 03 — ⚖️ 24/7 Autonomous Committee Debate Log (회의록: 리서치 vs 찬성 vs 비판 → 심판 결정)</h2>
      {signals_html}
    </div>

    <!-- PANEL 05: POTATO-SHIELD RISK & BINANCE TESTNET TERMINAL -->
    <div class="panel">
      <h2>PANEL 05 — 🛡️ Potato-Shield Risk &amp; Binance Testnet Terminal</h2>
      <div class="kpi-bar">
        <div class="kpi"><div class="kpi-title">Execution Mode</div><div class="kpi-val" style="font-size:13px;color:#58a6ff;">{testnet_balance['mode']}</div></div>
        <div class="kpi"><div class="kpi-title">Testnet Balance</div><div class="kpi-val">${testnet_balance['available_balance_usdt']:,.2f} USDT</div></div>
        <div class="kpi"><div class="kpi-title">Testnet Ping</div><div class="kpi-val">{testnet_ping.get('latency_ms', -1)} ms</div></div>
      </div>
      <p style="font-size:12.5px;margin:6px 0;"><strong>Hard Risk Constitution (W2 Verified):</strong> Per-Trade Cap = <code>1.0% (Max 1.5%)</code> | Daily Kill Switch = <code>-2.0%</code> | Max Gross Leverage = <code>5.0x</code></p>
      <ul style="font-size:12px;color:#3fb950;margin:6px 0;padding-left:18px;">
        {"".join(f"<li><code>{d}</code></li>" for d in w2_details)}
      </ul>
      <h3 style="margin-top:14px;font-size:13px;">Recent Autonomous Decision &amp; Order Audit Ledger</h3>
      <table>
        <tr><th>Time</th><th>Symbol</th><th>Strat</th><th>Entry</th><th>Stop</th><th>Size</th><th>Verdict</th><th>Order</th></tr>
        {trade_rows_html}
      </table>
    </div>
  </div>

  <!-- PANEL 04: STRATEGY LAB & W1 REJECTION MACHINE -->
  <div class="panel-full">
    <h2>PANEL 04 — 🔬 Strategy Lab: Single Strategy vs. 3-Axis Regime Switcher (360-Day Real Binance Proof)</h2>
    <table>
      <tr>
        <th>Strategy Mode (360 Days, 10 Coins, Full Fees+Slippage+Funding)</th>
        <th>Trades</th>
        <th>Gross PnL</th>
        <th>Total Cost</th>
        <th>Net PnL (After Cost)</th>
        <th>Gross/Cost Ratio</th>
        <th>Win Rate</th>
        <th>Max Drawdown</th>
      </tr>
      {strat_rows_html}
    </table>
  </div>

  <!-- PANEL 06: CRUMB PERSISTENT SQLITE RESEARCH JOURNAL -->
  <div class="panel-full">
    <h2>PANEL 06 — 🍞 Crumb SQLite Institutional Memory (`data/crumb_memory.db` Experiment History)</h2>
    <table>
      <tr>
        <th>Recorded At</th>
        <th>Strategy Name</th>
        <th>Config SHA-256</th>
        <th>Trades</th>
        <th>Gross PnL</th>
        <th>Total Cost</th>
        <th>Net PnL</th>
        <th>Verdict</th>
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
    btc_1h = fetcher.fetch_klines("BTCUSDT", "1h", days=360)
    univ_1h = {sym: fetcher.fetch_klines(sym, "1h", days=360) for sym in CONFIG.target_universe}
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

    for sig in regime_scan["actionable_signals"][:3]:
        if sig["eggbot_approved"]:
            ord_res = testnet_client.place_testnet_bracket_order(
                symbol=sig["symbol"],
                side=sig["side_int"],
                quantity=sig["quantity"],
                stop_price=sig["stop_price"],
                leverage=int(max(1, round(sig["leverage"]))),
                execute_on_testnet=execute_on_testnet,
            )
            order_status = ord_res["status"]
            order_id = ord_res["order_id"]
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
