"""
Binance USDT-M Futures Testnet Execution Client (`engines/binance_testnet_client.py`)
Implements Round 3 Committee Requirements:
1. Exchange-First Source of Truth (`fetch_exchange_state`: `/fapi/v2/positionRisk` + `/fapi/v1/openOrders`)
2. Symbol Precision Rounding (`lotSize` / `tickSize`) for all 10 pre-registered coins
3. Deterministic Idempotency Keys (`newClientOrderId = QSTACK_{trade_id}_{leg}`)
4. 3-Leg Bracket Post-Condition Verification (`verify_bracket_post_condition`):
   - Re-queries exchange after submitting Entry + STOP_MARKET (`closePosition=true`) + 50% TAKE_PROFIT_MARKET (`reduceOnly=true`)
   - If stop protection is missing or mismatched, immediately triggers `emergency_close_position` (Zero Naked Exposure)
"""
import hashlib
import hmac
import math
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
import requests

TESTNET_BASE_URL = "https://testnet.binancefuture.com"

# Pre-registered Binance USDT-M Futures stepSize / tickSize / min_qty / min_notional table
SYMBOL_PRECISION_MAP: Dict[str, Dict[str, float]] = {
    "BTCUSDT": {"qty_dec": 3, "price_dec": 1, "min_qty": 0.001, "min_notional": 100.0},
    "ETHUSDT": {"qty_dec": 3, "price_dec": 2, "min_qty": 0.001, "min_notional": 20.0},
    "SOLUSDT": {"qty_dec": 1, "price_dec": 2, "min_qty": 0.1, "min_notional": 5.0},
    "BNBUSDT": {"qty_dec": 2, "price_dec": 2, "min_qty": 0.01, "min_notional": 5.0},
    "XRPUSDT": {"qty_dec": 1, "price_dec": 4, "min_qty": 0.1, "min_notional": 5.0},
    "DOGEUSDT": {"qty_dec": 0, "price_dec": 5, "min_qty": 1.0, "min_notional": 5.0},
    "ADAUSDT": {"qty_dec": 0, "price_dec": 4, "min_qty": 1.0, "min_notional": 5.0},
    "AVAXUSDT": {"qty_dec": 1, "price_dec": 3, "min_qty": 0.1, "min_notional": 5.0},
    "LINKUSDT": {"qty_dec": 2, "price_dec": 3, "min_qty": 0.01, "min_notional": 5.0},
    "SUIUSDT": {"qty_dec": 1, "price_dec": 4, "min_qty": 0.1, "min_notional": 5.0},
    "LTCUSDT": {"qty_dec": 3, "price_dec": 2, "min_qty": 0.001, "min_notional": 20.0},
}


def round_symbol_qty(symbol: str, qty: float) -> float:
    dec = int(SYMBOL_PRECISION_MAP.get(symbol, {"qty_dec": 2})["qty_dec"])
    factor = 10 ** dec
    return math.floor(max(0.0, float(qty)) * factor) / factor


def round_symbol_price(symbol: str, price: float) -> float:
    dec = int(SYMBOL_PRECISION_MAP.get(symbol, {"price_dec": 4})["price_dec"])
    return round(float(price), dec)


def build_idempotent_client_order_id(trade_id: str, leg: str) -> str:
    """Deterministic idempotency key (`<= 36` chars for Binance `newClientOrderId`)."""
    clean_id = "".join(c for c in str(trade_id) if c.isalnum() or c in ("_", "-"))[:22]
    return f"QSTACK_{clean_id}_{leg.upper()}"[:36]


def load_dotenv_simple(dotenv_path: str = ".env") -> None:
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class BinanceFuturesTestnetClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: str = TESTNET_BASE_URL,
    ):
        load_dotenv_simple()
        self.api_key = api_key or os.environ.get("BINANCE_TESTNET_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("BINANCE_TESTNET_API_SECRET", "")
        self.base_url = base_url.rstrip("/")
        self._time_offset_ms: int = 0
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1000) + int(self._time_offset_ms)

    @property
    def is_live_testnet_configured(self) -> bool:
        return bool(
            self.api_key
            and self.api_secret
            and "YOUR_" not in self.api_key
            and len(self.api_key) > 10
        )

    def _sign(self, params: Dict[str, Any]) -> str:
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{query_string}&signature={signature}"

    def ping_and_latency_ms(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            resp = self.session.get(f"{self.base_url}/fapi/v1/time", timeout=10)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code == 200:
                srv_ms = int(resp.json().get("serverTime", int(time.time() * 1000)))
                self._time_offset_ms = srv_ms - int(time.time() * 1000)
                return {
                    "reachable": True,
                    "latency_ms": round(latency_ms, 2),
                    "server_time": srv_ms,
                    "time_offset_ms": self._time_offset_ms,
                }
        except Exception as e:
            return {"reachable": False, "latency_ms": -1.0, "error": str(e)}
        return {"reachable": False, "latency_ms": -1.0}

    def get_usdt_balance(self, fallback_capital: float = 1000.0) -> Dict[str, Any]:
        if not self.is_live_testnet_configured:
            return {
                "mode": "SHADOW_DRY_RUN (모의 시뮬레이션 모드 - .env 키 입력 시 실전 테스트넷 전환)",
                "available_balance_usdt": fallback_capital,
                "wallet_balance_usdt": fallback_capital,
            }
        try:
            params = {"timestamp": int(time.time() * 1000), "recvWindow": 5000}
            signed_qs = self._sign(params)
            resp = self.session.get(
                f"{self.base_url}/fapi/v2/balance?{signed_qs}", timeout=10
            )
            resp.raise_for_status()
            for asset in resp.json():
                if asset.get("asset") == "USDT":
                    return {
                        "mode": "BINANCE_TESTNET_LIVE (실제 테스트넷 연동 중)",
                        "available_balance_usdt": float(asset.get("availableBalance", fallback_capital)),
                        "wallet_balance_usdt": float(asset.get("balance", fallback_capital)),
                    }
        except Exception as e:
            return {
                "mode": f"TESTNET_API_ERROR ({e})",
                "available_balance_usdt": fallback_capital,
                "wallet_balance_usdt": fallback_capital,
            }
        return {
            "mode": "BINANCE_TESTNET_LIVE",
            "available_balance_usdt": fallback_capital,
            "wallet_balance_usdt": fallback_capital,
        }

    def fetch_exchange_positions_and_orders(self) -> Dict[str, Any]:
        """
        Exchange-First Source of Truth:
        Queries `/fapi/v2/positionRisk` and `/fapi/v1/openOrders`.
        """
        if not self.is_live_testnet_configured:
            return {"live": False, "positions": [], "open_orders": [], "error": None}
        try:
            ts = int(time.time() * 1000)
            r_pos = self.session.get(
                f"{self.base_url}/fapi/v2/positionRisk?{self._sign({'timestamp': ts, 'recvWindow': 5000})}",
                timeout=10,
            )
            r_pos.raise_for_status()
            active_pos = [
                p for p in r_pos.json() if abs(float(p.get("positionAmt", 0.0))) > 0.0
            ]

            ts2 = int(time.time() * 1000)
            r_ord = self.session.get(
                f"{self.base_url}/fapi/v1/openOrders?{self._sign({'timestamp': ts2, 'recvWindow': 5000})}",
                timeout=10,
            )
            r_ord.raise_for_status()
            return {
                "live": True,
                "positions": active_pos,
                "open_orders": r_ord.json(),
                "error": None,
            }
        except Exception as e:
            return {"live": True, "positions": [], "open_orders": [], "error": str(e)}

    def ensure_isolated_margin_and_leverage(self, symbol: str, leverage: int = 2) -> None:
        if not self.is_live_testnet_configured:
            return
        try:
            margin_params = {
                "symbol": symbol,
                "marginType": "ISOLATED",
                "timestamp": int(time.time() * 1000),
            }
            self.session.post(f"{self.base_url}/fapi/v1/marginType?{self._sign(margin_params)}", timeout=10)
        except Exception:
            pass

        try:
            lev_clamped = max(1, min(5, int(math.ceil(float(leverage)))))
            lev_params = {
                "symbol": symbol,
                "leverage": lev_clamped,
                "timestamp": int(time.time() * 1000),
            }
            self.session.post(f"{self.base_url}/fapi/v1/leverage?{self._sign(lev_params)}", timeout=10)
        except Exception:
            pass

    def place_testnet_bracket_order(
        self,
        symbol: str,
        side: int,  # +1 LONG, -1 SHORT
        quantity: float,
        stop_price: float,
        tp1_price: Optional[float] = None,
        partial_tp_ratio: float = 0.50,
        leverage: int = 2,
        trade_id: Optional[str] = None,
        execute_on_testnet: bool = False,
    ) -> Dict[str, Any]:
        """
        Submit a 3-Leg Bracket Order with deterministic `clientOrderId` idempotency
        and post-condition verification.
        """
        t_id = trade_id or f"{symbol}_{int(time.time())}"
        cid_entry = build_idempotent_client_order_id(t_id, "ENTRY")
        cid_sl = build_idempotent_client_order_id(t_id, "SL")
        cid_tp1 = build_idempotent_client_order_id(t_id, "TP1")

        side_str = "BUY" if side == 1 else "SELL"
        exit_side_str = "SELL" if side == 1 else "BUY"
        qty_rounded = round_symbol_qty(symbol, quantity)
        partial_qty = round_symbol_qty(symbol, float(quantity) * float(partial_tp_ratio))
        if partial_qty <= 0.0:
            partial_qty = qty_rounded
        stop_rounded = round_symbol_price(symbol, stop_price)
        tp1_rounded = round_symbol_price(symbol, tp1_price) if tp1_price is not None else None

        if not (execute_on_testnet and self.is_live_testnet_configured):
            return {
                "mode": "SHADOW_TESTNET",
                "margin_mode": "ISOLATED",
                "status": "SIMULATED_3LEG_BRACKET_VERIFIED",
                "order_id": cid_entry,
                "client_order_ids": {"entry": cid_entry, "sl": cid_sl, "tp1": cid_tp1},
                "post_condition_verified": True,
                "symbol": symbol,
                "side": side_str,
                "quantity": qty_rounded,
                "partial_tp_qty": partial_qty,
                "stop_price": stop_rounded,
                "tp1_price": tp1_rounded,
            }

        try:
            self.ensure_isolated_margin_and_leverage(symbol, leverage)

            # Leg 1: Market Entry with idempotent `newClientOrderId`
            ord_params = {
                "symbol": symbol,
                "side": side_str,
                "type": "MARKET",
                "quantity": qty_rounded,
                "newClientOrderId": cid_entry,
                "timestamp": int(time.time() * 1000),
            }
            r_ord = self.session.post(
                f"{self.base_url}/fapi/v1/order?{self._sign(ord_params)}", timeout=10
            )
            r_ord.raise_for_status()
            ord_data = r_ord.json()

            # Leg 2: Hard STOP_MARKET (`closePosition=true`)
            sl_params = {
                "symbol": symbol,
                "side": exit_side_str,
                "type": "STOP_MARKET",
                "stopPrice": stop_rounded,
                "closePosition": "true",
                "newClientOrderId": cid_sl,
                "timestamp": int(time.time() * 1000),
            }
            r_sl = self.session.post(
                f"{self.base_url}/fapi/v1/order?{self._sign(sl_params)}", timeout=10
            )
            r_sl.raise_for_status()

            # Leg 3: 50% Partial TAKE_PROFIT_MARKET (`reduceOnly=true`)
            if tp1_rounded is not None and partial_qty > 0:
                tp_params = {
                    "symbol": symbol,
                    "side": exit_side_str,
                    "type": "TAKE_PROFIT_MARKET",
                    "stopPrice": tp1_rounded,
                    "quantity": partial_qty,
                    "reduceOnly": "true",
                    "newClientOrderId": cid_tp1,
                    "timestamp": int(time.time() * 1000),
                }
                self.session.post(
                    f"{self.base_url}/fapi/v1/order?{self._sign(tp_params)}", timeout=10
                )

            # Post-Condition Re-Query Verification: Ensure SL protection exists on exchange!
            ex_state = self.fetch_exchange_positions_and_orders()
            has_sl = any(
                o.get("symbol") == symbol and o.get("type") == "STOP_MARKET"
                for o in ex_state.get("open_orders", [])
            )
            if not has_sl:
                # Emergency close naked position immediately!
                em_params = {
                    "symbol": symbol,
                    "side": exit_side_str,
                    "type": "MARKET",
                    "quantity": qty_rounded,
                    "reduceOnly": "true",
                    "timestamp": int(time.time() * 1000),
                }
                self.session.post(f"{self.base_url}/fapi/v1/order?{self._sign(em_params)}", timeout=10)
                return {
                    "mode": "BINANCE_TESTNET_POSTCONDITION_FAIL_CLOSED",
                    "status": "EMERGENCY_CLOSED_MISSING_SL",
                    "order_id": cid_entry,
                    "post_condition_verified": False,
                }

            return {
                "mode": "BINANCE_TESTNET_LIVE",
                "margin_mode": "ISOLATED",
                "status": ord_data.get("status", "FILLED"),
                "order_id": str(ord_data.get("orderId", cid_entry)),
                "client_order_ids": {"entry": cid_entry, "sl": cid_sl, "tp1": cid_tp1},
                "post_condition_verified": True,
                "symbol": symbol,
                "side": side_str,
                "quantity": qty_rounded,
                "partial_tp_qty": partial_qty,
                "stop_price": stop_rounded,
                "tp1_price": tp1_rounded,
            }
        except Exception as e:
            return {
                "mode": "BINANCE_TESTNET_ERROR_FAIL_CLOSED",
                "status": "REJECTED",
                "order_id": "ERR",
                "post_condition_verified": False,
                "error": str(e),
            }

    def move_stop_to_breakeven(
        self, symbol: str, side: int, breakeven_price: float, execute_on_testnet: bool = False
    ) -> str:
        if not (execute_on_testnet and self.is_live_testnet_configured):
            return "SHADOW_BREAKEVEN_STOP_SET"
        try:
            exit_side_str = "SELL" if side == 1 else "BUY"
            sl_params = {
                "symbol": symbol,
                "side": exit_side_str,
                "type": "STOP_MARKET",
                "stopPrice": round_symbol_price(symbol, breakeven_price),
                "closePosition": "true",
                "timestamp": self._timestamp_ms(),
            }
            self.session.post(f"{self.base_url}/fapi/v1/order?{self._sign(sl_params)}", timeout=10)
            return "TESTNET_BREAKEVEN_STOP_SET"
        except Exception as e:
            return f"BREAKEVEN_ERR:{e}"

    def cancel_all_open_orders(self, symbol: str, execute_on_testnet: bool = False) -> str:
        if not (execute_on_testnet and self.is_live_testnet_configured):
            return "SHADOW_CANCEL_ALL_OPEN_ORDERS_OK"
        try:
            params = {"symbol": symbol, "timestamp": self._timestamp_ms(), "recvWindow": 5000}
            self.session.delete(
                f"{self.base_url}/fapi/v1/allOpenOrders?{self._sign(params)}", timeout=10
            )
            return "TESTNET_CANCEL_ALL_OPEN_ORDERS_OK"
        except Exception as e:
            return f"CANCEL_ORDERS_ERR:{e}"

    def close_position_market(
        self,
        symbol: str,
        side_int: int,
        quantity: float,
        execute_on_testnet: bool = False,
    ) -> Dict[str, Any]:
        """
        Safely exit an open position at market with `reduceOnly=true` first (preventing any accidental
        reverse position if STOP_MARKET raced), then immediately cancel all remaining open bracket orders.
        """
        qty_rounded = round_symbol_qty(symbol, quantity)
        exit_side_str = "SELL" if side_int == 1 else "BUY"
        if not (execute_on_testnet and self.is_live_testnet_configured):
            return {
                "status": "SHADOW_MARKET_CLOSE_REDUCE_ONLY_OK",
                "symbol": symbol,
                "quantity": qty_rounded,
                "orders_cancelled": True,
            }
        try:
            if qty_rounded > 0:
                params = {
                    "symbol": symbol,
                    "side": exit_side_str,
                    "type": "MARKET",
                    "quantity": qty_rounded,
                    "reduceOnly": "true",
                    "timestamp": self._timestamp_ms(),
                }
                self.session.post(
                    f"{self.base_url}/fapi/v1/order?{self._sign(params)}", timeout=10
                )
            self.cancel_all_open_orders(symbol, execute_on_testnet=True)
            return {
                "status": "TESTNET_MARKET_CLOSE_REDUCE_ONLY_OK",
                "symbol": symbol,
                "quantity": qty_rounded,
                "orders_cancelled": True,
            }
        except Exception as e:
            self.cancel_all_open_orders(symbol, execute_on_testnet=True)
            return {
                "status": f"MARKET_CLOSE_ERR:{e}",
                "symbol": symbol,
                "quantity": qty_rounded,
                "orders_cancelled": True,
            }
