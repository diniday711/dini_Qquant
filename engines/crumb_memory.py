"""
07. CRUMB — Persistent Institutional Memory & SQLite Ledger (`engines/crumb_memory.py`)
Implements Muse Wonder / Crumb from the One-Person AI Hedge Fund paper & Lauren Tan's pstack:
- Stores every backtest experiment (`experiments` table) with config SHA-256, W1/W2 results, and rejection reason.
- Stores every 3-axis Market Regime snapshot (`regime_snapshots` table).
- Stores every Testnet / Paper trade intent, Dr. Eggbot audit, and execution fill (`trade_ledger` table).
- Prevents repeating failed ideas by querying prior contradictions ("Negative Knowledge").
"""
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class CrumbMemoryDB:
    def __init__(self, db_path: str = "data/crumb_memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    config_sha256 TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    days_tested INTEGER NOT NULL,
                    trades_count INTEGER NOT NULL,
                    gross_pnl_usd REAL NOT NULL,
                    total_cost_usd REAL NOT NULL,
                    net_pnl_usd REAL NOT NULL,
                    gross_to_cost_ratio REAL NOT NULL,
                    sharpe REAL NOT NULL,
                    mdd_pct REAL NOT NULL,
                    w1_status TEXT NOT NULL,
                    w2_status TEXT NOT NULL,
                    rejection_reason TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS regime_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    dominant_trend_regime TEXT NOT NULL,
                    volatility_state TEXT NOT NULL,
                    risk_appetite TEXT NOT NULL,
                    market_breadth_pct REAL NOT NULL,
                    btc_24h_return_pct REAL NOT NULL,
                    active_strategy_mode TEXT NOT NULL,
                    confidence_score INTEGER NOT NULL,
                    what_could_be_wrong TEXT NOT NULL,
                    coin_regimes_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    regime_label TEXT NOT NULL,
                    sub_strategy TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    notional_usd REAL NOT NULL,
                    leverage REAL NOT NULL,
                    eggbot_verdict TEXT NOT NULL,
                    eggbot_bear_case TEXT NOT NULL,
                    order_status TEXT NOT NULL,
                    order_id TEXT NOT NULL
                );
                """
            )

    def log_experiment(
        self,
        strategy_name: str,
        config_sha256: str,
        timeframe: str,
        days_tested: int,
        trades_count: int,
        gross_pnl_usd: float,
        total_cost_usd: float,
        net_pnl_usd: float,
        gross_to_cost_ratio: float,
        sharpe: float,
        mdd_pct: float,
        w1_status: str,
        w2_status: str,
        rejection_reason: str,
    ) -> int:
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO experiments (
                    created_at, strategy_name, config_sha256, timeframe, days_tested,
                    trades_count, gross_pnl_usd, total_cost_usd, net_pnl_usd,
                    gross_to_cost_ratio, sharpe, mdd_pct, w1_status, w2_status, rejection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_str,
                    strategy_name,
                    config_sha256,
                    timeframe,
                    int(days_tested),
                    int(trades_count),
                    float(gross_pnl_usd),
                    float(total_cost_usd),
                    float(net_pnl_usd),
                    float(gross_to_cost_ratio),
                    float(sharpe),
                    float(mdd_pct),
                    w1_status,
                    w2_status,
                    rejection_reason,
                ),
            )
            return int(cur.lastrowid)

    def log_regime_snapshot(
        self,
        dominant_trend_regime: str,
        volatility_state: str,
        risk_appetite: str,
        market_breadth_pct: float,
        btc_24h_return_pct: float,
        active_strategy_mode: str,
        confidence_score: int,
        what_could_be_wrong: str,
        coin_regimes: List[Dict[str, Any]],
    ) -> int:
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO regime_snapshots (
                    recorded_at, dominant_trend_regime, volatility_state, risk_appetite,
                    market_breadth_pct, btc_24h_return_pct, active_strategy_mode,
                    confidence_score, what_could_be_wrong, coin_regimes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_str,
                    dominant_trend_regime,
                    volatility_state,
                    risk_appetite,
                    float(market_breadth_pct),
                    float(btc_24h_return_pct),
                    active_strategy_mode,
                    int(confidence_score),
                    what_could_be_wrong,
                    json.dumps(coin_regimes, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def log_trade_event(
        self,
        execution_mode: str,
        symbol: str,
        side: str,
        regime_label: str,
        sub_strategy: str,
        entry_price: float,
        stop_price: float,
        notional_usd: float,
        leverage: float,
        eggbot_verdict: str,
        eggbot_bear_case: str,
        order_status: str,
        order_id: str = "SIM-000",
    ) -> int:
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO trade_ledger (
                    recorded_at, execution_mode, symbol, side, regime_label, sub_strategy,
                    entry_price, stop_price, notional_usd, leverage,
                    eggbot_verdict, eggbot_bear_case, order_status, order_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_str,
                    execution_mode,
                    symbol,
                    side,
                    regime_label,
                    sub_strategy,
                    float(entry_price),
                    float(stop_price),
                    float(notional_usd),
                    float(leverage),
                    eggbot_verdict,
                    eggbot_bear_case,
                    order_status,
                    order_id,
                ),
            )
            return int(cur.lastrowid)

    def get_recent_experiments(self, limit: int = 15) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trade_ledger ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_latest_regime_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM regime_snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["coin_regimes"] = json.loads(d["coin_regimes_json"])
            return d
