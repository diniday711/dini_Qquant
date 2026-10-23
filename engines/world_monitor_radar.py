"""
WorldMonitor Global Intelligence, Macro Cross-Asset & Sentiment Radar (`engines/world_monitor_radar.py`)
Inspired by `koala73/worldmonitor` + Jane Street `audit_bot.xml` + Wonyotti Macro Context:
Fetches real-time global context (zero paid API keys required) to drive Top-Down Strategy Selection:
1. Global Macro & Geopolitical Headlines (Live RSS from Yahoo Finance Macro, CoinDesk, Cointelegraph)
   - Detects Geopolitical/Systemic Shocks (War, Tariff, Hack, SEC/Regulation, Emergency)
   - Detects Macro Volatility Catalysts (CPI, FOMC, Fed Powell, Nonfarm Payrolls, Rate Decisions)
   - Detects Institutional Bullish Catalysts (ETF Inflows, Rate Cut, Strategic Reserve)
2. Crypto Fear & Greed Index (`https://api.alternative.me/fng/` — 0 to 100)
3. Cross-Asset Safe-Haven Flow (`PAXGUSDT` Gold Token on Binance vs `BTCUSDT` & Altcoin Breadth)
4. Synthesizes a deterministic `WorldMonitorVerdict` that selects the optimal Strategy Family
   and adjusts risk exposure BEFORE any single-coin chart signal is evaluated.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List
import xml.etree.ElementTree as ET
import requests


RSS_FEEDS = [
    ("Global_Macro_Markets", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC,DX-Y.NYB,GC=F,BTC-USD&region=US&lang=en-US"),
    ("Crypto_Macro_News", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Global_Crypto_Flow", "https://cointelegraph.com/rss"),
]

SHOCK_KEYWORDS = [
    "war", "missile", "invasion", "airstrike", "nuclear", "hack", "exploited",
    "bankrupt", "insolvency", "default", "emergency", "halt", "ban", "sanction",
    "tariff", "crash", "plunge", "black swan", "liquidation cascade"
]

MACRO_EVENT_KEYWORDS = [
    "fomc", "cpi", "inflation", "powell", "fed ", "federal reserve", "interest rate",
    "nonfarm", "payroll", "jobs report", "pce", "treasury yield", "recession"
]

BULLISH_KEYWORDS = [
    "etf inflow", "rate cut", "approval", "approved", "bullish", "surge",
    "rally", "institutional buying", "record high", "stimulus", "adoption"
]


@dataclass
class WorldMonitorVerdict:
    timestamp_utc: str
    fear_greed_value: int
    fear_greed_label: str
    gold_24h_return_pct: float
    btc_24h_return_pct: float
    safe_haven_divergence_pct: float  # Gold 24h return - BTC 24h return
    shock_score: int
    macro_event_score: int
    bullish_score: int
    global_regime_code: str           # e.g. "DEFCON1_EVENT_LOCK", "GLOBAL_RISK_OFF_SHORT_BIAS", "GLOBAL_RISK_ON_LONG_BIAS", "EXTREME_FEAR_REVERSAL", "NORMAL_SELECTIVE"
    recommended_strategy_family: str  # Which weapon to equip based on world state
    risk_multiplier: float            # 0.0 (Cash lock), 0.5 (Defensive half-size), 1.0 (Normal W2 cap)
    allow_long: bool
    allow_short: bool
    korean_summary: str
    top_headlines: List[Dict[str, str]] = field(default_factory=list)


class WorldMonitorRadar:
    def __init__(self, timeout_sec: int = 6):
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; QStackWorldMonitor/3.1)"}
        )

    def fetch_fear_and_greed(self) -> Dict[str, Any]:
        try:
            r = self.session.get("https://api.alternative.me/fng/?limit=1", timeout=self.timeout_sec)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    val = int(data[0].get("value", 50))
                    cls = str(data[0].get("value_classification", "Neutral"))
                    return {"value": val, "label": cls}
        except Exception:
            pass
        return {"value": 50, "label": "Neutral (Fallback)"}

    def fetch_gold_paxg_24h_return(self) -> float:
        """
        Fetch 24h return of PAXGUSDT (Gold backed token on Binance Futures/Spot)
        as a real-time Safe-Haven proxy (Gold vs Crypto risk appetite).
        """
        try:
            r = self.session.get(
                "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=PAXGUSDT",
                timeout=self.timeout_sec,
            )
            if r.status_code == 200:
                return float(r.json().get("priceChangePercent", 0.0)) / 100.0
        except Exception:
            pass
        try:
            r2 = self.session.get(
                "https://api.binance.com/api/v3/ticker/24hr?symbol=PAXGUSDT",
                timeout=self.timeout_sec,
            )
            if r2.status_code == 200:
                return float(r2.json().get("priceChangePercent", 0.0)) / 100.0
        except Exception:
            pass
        return 0.0

    def fetch_global_headlines(self, max_per_feed: int = 5) -> List[Dict[str, str]]:
        headlines: List[Dict[str, str]] = []
        for source_name, url in RSS_FEEDS:
            try:
                r = self.session.get(url, timeout=self.timeout_sec)
                if r.status_code != 200:
                    continue
                root = ET.fromstring(r.content)
                items = root.findall(".//item")
                for it in items[:max_per_feed]:
                    title = (it.findtext("title") or "").strip()
                    if not title:
                        continue
                    tag = "일반"
                    t_low = title.lower()
                    if any(k in t_low for k in SHOCK_KEYWORDS):
                        tag = "🚨충격/위기"
                    elif any(k in t_low for k in MACRO_EVENT_KEYWORDS):
                        tag = "🏦거시/금리"
                    elif any(k in t_low for k in BULLISH_KEYWORDS):
                        tag = "🚀호재/유입"
                    headlines.append({"source": source_name, "tag": tag, "title": title})
            except Exception:
                continue
        return headlines[:12]

    def evaluate_world_state(
        self, btc_24h_return_pct: float, market_breadth_pct: float
    ) -> WorldMonitorVerdict:
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        fng = self.fetch_fear_and_greed()
        fg_val = int(fng["value"])
        fg_label = str(fng["label"])

        gold_ret_24 = self.fetch_gold_paxg_24h_return()
        safe_haven_div = gold_ret_24 - btc_24h_return_pct

        headlines = self.fetch_global_headlines()
        shock_score = 0
        macro_score = 0
        bull_score = 0

        for h in headlines:
            t_low = h["title"].lower()
            if any(k in t_low for k in SHOCK_KEYWORDS):
                shock_score += 2
            if any(k in t_low for k in MACRO_EVENT_KEYWORDS):
                macro_score += 1
            if any(k in t_low for k in BULLISH_KEYWORDS):
                bull_score += 1

        # --- Decision Matrix: Global Context -> Strategy Family & Direction Filter ---
        if shock_score >= 6 and abs(btc_24h_return_pct) > 0.04:
            regime_code = "DEFCON1_GEOPOLITICAL_OR_SYSTEMIC_SHOCK"
            rec_strat = "STAY_IN_CASH_EVENT_LOCK (지정학/시스템 위기 현금 대피)"
            risk_mult = 0.0
            allow_long = False
            allow_short = False
            kr_summary = (
                f"🚨 [비상 대피] 글로벌 속보에서 심각한 충격 키워드(위기점수 {shock_score}점)와 급격한 가격 변동이 동시 감지되었습니다. "
                "모든 신규 진입을 잠그고 100% 현금으로 관망합니다."
            )
        elif safe_haven_div >= 0.012 and btc_24h_return_pct < -0.005 and market_breadth_pct <= 0.45:
            regime_code = "GLOBAL_RISK_OFF_SAFE_HAVEN_FLIGHT"
            rec_strat = "WONYOTTI_SHORT_BREAKOUT_ONLY (안전자산 대피장 -> 워뇨띠 하락돌파 숏 전용)"
            risk_mult = 0.8 if macro_score >= 3 else 1.0
            allow_long = False
            allow_short = True
            kr_summary = (
                f"📉 [글로벌 Risk-Off (위험회피)] 안전자산 금(PAXG {gold_ret_24:+.2%})으로 자금이 대피하고 비트코인({btc_24h_return_pct:+.2%}) 및 알트 시장 폭({market_breadth_pct:.0%})이 약세입니다. "
                f"공포탐욕지수={fg_val}({fg_label}). 어설픈 저점 롱(매수)을 전면 금지하고, '워뇨띠 거래량 군집 하락돌파(SHORT)' 전략만 허용합니다."
            )
        elif fg_val <= 22 and btc_24h_return_pct > -0.02:
            regime_code = "EXTREME_FEAR_CAPITULATION_ZONE"
            rec_strat = "PANIC_ABSORPTION_MEAN_REVERSION (극단적 공포 투매 진정 -> 수급 흡수 역추세)"
            risk_mult = 0.8
            allow_long = True
            allow_short = False
            kr_summary = (
                f"😱 [극단적 공포 역발상 구간] 크립토 공포탐욕지수가 {fg_val}({fg_label})로 대중이 패닉에 빠진 바닥권입니다. "
                "추격 숏을 금지하고, 매수 수급(Delta)이 들어오는 메이저 코인의 '볼린저 하단 수급 흡수 역추세(LONG)' 전략을 우선 가동합니다."
            )
        elif (btc_24h_return_pct > 0.008 and market_breadth_pct >= 0.55) or (bull_score >= 3 and market_breadth_pct >= 0.50):
            regime_code = "GLOBAL_RISK_ON_EXPANSION"
            rec_strat = "WONYOTTI_LONG_VOL_CLUSTER (글로벌 Risk-On -> 워뇨띠 거래량 군집 상승돌파 주력)"
            risk_mult = 1.0
            allow_long = True
            allow_short = False
            kr_summary = (
                f"🚀 [글로벌 Risk-On (위험선호 유입)] 비트코인({btc_24h_return_pct:+.2%})과 시장 폭({market_breadth_pct:.0%})이 강세이며 "
                f"공포탐욕지수는 {fg_val}({fg_label}), 호재 점수 {bull_score}점입니다. '워뇨띠 거래량 군집 + 사토시 상대강도 상승돌파(LONG)' 전략을 주력 가동합니다."
            )
        else:
            regime_code = "MACRO_NEUTRAL_SELECTIVE_SWITCHER"
            rec_strat = "HYBRID_REGIME_SWITCHER (거시 중립 -> 코인별 사토시 상대강도 & 장세 맞춤 선별)"
            risk_mult = 0.8 if macro_score >= 3 else 1.0
            allow_long = market_breadth_pct >= 0.35
            allow_short = market_breadth_pct <= 0.65
            kr_summary = (
                f"🌍 [글로벌 정세 중립/선별장] 공포탐욕지수={fg_val}({fg_label}), 금(PAXG) 24h={gold_ret_24:+.2%}, BTC 24h={btc_24h_return_pct:+.2%} "
                f"(거시 뉴스 {macro_score}건 / 충격 뉴스 {shock_score}건). 거시적 쏠림이 크지 않으므로 코인별 3축 차트 장세와 사토시 상대강도에 따라 롱/숏 무기를 선별 적용합니다."
            )

        return WorldMonitorVerdict(
            timestamp_utc=now_str,
            fear_greed_value=fg_val,
            fear_greed_label=fg_label,
            gold_24h_return_pct=gold_ret_24,
            btc_24h_return_pct=btc_24h_return_pct,
            safe_haven_divergence_pct=safe_haven_div,
            shock_score=shock_score,
            macro_event_score=macro_score,
            bullish_score=bull_score,
            global_regime_code=regime_code,
            recommended_strategy_family=rec_strat,
            risk_multiplier=risk_mult,
            allow_long=allow_long,
            allow_short=allow_short,
            korean_summary=kr_summary,
            top_headlines=headlines[:6],
        )


if __name__ == "__main__":
    wm = WorldMonitorRadar()
    v = wm.evaluate_world_state(btc_24h_return_pct=-0.0124, market_breadth_pct=0.30)
    print(v.korean_summary)
    for hl in v.top_headlines:
        print(f"  - [{hl['tag']}] ({hl['source']}) {hl['title']}")
