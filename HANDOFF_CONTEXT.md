# 🏛️ Q-Stack v3.2: 24/7 Autonomous Multi-Agent Crypto Quant Desk (Handoff Context)

- **GitHub Repository**: `https://github.com/diniday711/dini_Qquant.git`
- **Conversation ID**: `9f276f93-dce7-40a8-99f0-5fc0490a7d9b`
- **Target Market**: Binance USDT-M Perpetual Futures (10-Coin Fixed Liquid Universe + `BTCUSDT` Benchmark + `PAXGUSDT` Safe-Haven Gold Proxy)
- **Target Deployment**: Oracle Cloud Server (Singapore `ap-southeast-1`) + Hermes Agent (Discord Worker Bots using `codex`, `claude`, `opgo`)

---

## 1. Complete 6-Stage End-to-End Architecture (`run_full_cycle.py` & `qstack_cli.py`)

Every 15-minute cycle executes the following 6 stages deterministically:

1. **Stage 0 (`engines/world_monitor_radar.py`) — 🌍 WorldMonitor Real-Time Global Intelligence & Strategy Selector**:
   - Pulls live global macro/geopolitical/crypto headlines via RSS (Yahoo Finance S&P500/DXY/Gold/BTC, CoinDesk, Cointelegraph).
   - Pulls **3-Tier Sentiment**:
     - Daily `Alternative.me` Fear & Greed Index (with yesterday-vs-today delta)
     - Real-Time Intraday Sentiment (`0-100`, computed every 15m from 10-coin market breadth & BTC 24h momentum)
     - Live Binance Futures Retail (`globalLongShortAccountRatio`) vs Whale (`topLongShortPositionRatio`) Long/Short Ratios
   - Pulls Cross-Asset Safe-Haven Flow (`PAXGUSDT` 24h return vs `BTCUSDT` 24h return).
   - Outputs `WorldMonitorVerdict` (`allow_long`, `allow_short`, `risk_multiplier`, `recommended_strategy_family`).
2. **Stage 1 (`tests/test_w2_and_leakage.py`) — 🛡️ W2 Deterministic Risk & Leakage Gate (7/7 Unit Tests PASS)**:
   - Verifies 2-tier cost model (`0.10%` fee + `0.04%/0.15%` slippage + 8h funding), `shift(1)` zero look-ahead leakage, `1.0%` per-trade risk cap, `-2.0%` daily loss Kill Switch, Fail-Closed order adapter, shared random execution model, and Round 3 Unified Config + `clientOrderId` idempotency + Exchange-First Reconciliation.
3. **Stage 2 (`engines/position_manager.py` & `user_trading_rules.json`) — 💼 Exchange-First Reconciliation & Wonyotti Isolated Position Lifecycle Manager**:
   - Queries Binance `/fapi/v2/positionRisk` and `/fapi/v1/openOrders` as the single Source of Truth (`reconcile_exchange_vs_db`). Any mismatch with local SQLite `active_positions` or missing `STOP_MARKET` protection trips `reconciliation_halted = True` (`Fail-Closed`).
   - Enforces Unified W2 Slot Rules: Max `3` concurrent isolated slots (`20%` slot margin = `$200`, max `0.333x` notional per slot so total gross leverage `<= 1.0x`), `-2.5%` max stop distance ceiling (`<= $10` max loss per trade), and `+1.8%` 1st Partial Take-Profit (`50%` scale-out + automatic Break-Even Stop move).
4. **Stage 3 (`engines/regime_switcher.py`) — 🌤️ 3-Axis Market Regime Switcher & Strategy Lab**:
   - Classifies every coin along Trend (`EMA 20/50/200`, `ADX 14`), Volatility (`192h NATR percentile rank`), and Breadth/Flow (`200MA breadth`, `Satoshi relative strength vs BTC`, `3-bar Volume Cluster`).
   - 360-Day Real Binance Backtest Summary:
     - `1_blind_mean_reversion`: 603 trades, Net `-$588.26`, MDD `60.56%`
     - `2_blind_trend_breakout`: 236 trades, Net `-$72.47`, MDD `24.85%`
     - `3_regime_switched_multi_strategy`: 117 trades, Net `+$136.08`, `Gross/Cost = 3.09x`, MDD `18.06%`
     - `4_wonyotti_grid_regime_hybrid` (Wonyotti Multi-Candle Volume Cluster + Satoshi Coupling): **97 trades, Win Rate `41.2%`, Gross `+$349.88`, Cost `$72.43`, Net `+$277.45 (+27.7%)`, `Gross/Cost = 4.83x`, MDD `12.74% (<= 15% Gate PASS)`, Sharpe `0.96`**.
5. **Stage 4 & 5 (`engines/eggbot_auditor.py` & `engines/binance_testnet_client.py`) — ⚖️ Pre-Trade Audit Log & 3-Leg Idempotent Testnet Bracket Execution**:
   - Generates Korean Multi-Persona Debate Logs (`[리서치 봇 Hatcha]`, `[찬성 봇 Bull/Trend]`, `[비판 봇 Dr.Eggbot]`, `[최종 심판 결정]`).
   - Submits 3-Leg Bracket Orders (`MARKET` Entry + `STOP_MARKET` `closePosition=true` + 50% `TAKE_PROFIT_MARKET` `reduceOnly=true`) under `ISOLATED` margin with deterministic `newClientOrderId = QSTACK_{trade_id}_{ENTRY|SL|TP1}` and post-condition verification.
6. **Stage 6 (`engines/crumb_memory.py`, `reports/qstack_dashboard.html`, `reports/latest_discord_briefing.md`) — 🍞 Crumb SQLite Memory, Korean Traffic-Light Dashboard & Discord/Hermes Briefing**:
   - Saves all state to `data/crumb_memory.db`, renders the Korean 6-Panel Dashboard, and outputs/sends the Discord Markdown briefing.

---

## 2. Quick Commands (Home PC or Oracle Cloud Server)

```bash
# 1. Run 1 full autonomous cycle + print Discord briefing
python qstack_cli.py cycle

# 2. Print latest Discord briefing only
python qstack_cli.py briefing

# 3. Generate role-locked prompt + live data for Hermes Discord Worker Bots
python qstack_cli.py worker-prompt --role research
python qstack_cli.py worker-prompt --role critic
python qstack_cli.py worker-prompt --role judge

# 4. Run 24/7 daemon (every 15m) + Web Dashboard on port 8080 + Live Testnet orders
python run_full_cycle.py --daemon --port 8080 --live-testnet
```
