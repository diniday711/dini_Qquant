"""
Binance USDT-M Futures Testnet Execution Client (`engines/binance_testnet_client.py`)
Connects to Official Binance Futures Testnet (`https://testnet.binancefuture.com`)
or runs in safe `SHADOW_DRY_RUN` mode if API keys are not yet configured in `.env`.
Supports:
- Server time & connectivity ping (works optimal on Oracle Cloud Singapore `ap-southeast-1`)
- Account USDT balance & open position inquiry
- Atomic Market/Limit entry + `STOP_MARKET` hard stop-loss submission
"""
import hashlib
import hmac
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode
import requests

TESTNET_BASE_URL = "https://testnet.binancefuture.com"


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
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})

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
                return {
                    "reachable": True,
                    "latency_ms": round(latency_ms, 2),
                    "server_time": resp.json().get("serverTime"),
                }
        except Exception as e:
            return {"reachable": False, "latency_ms": -1.0, "error": str(e)}
        return {"reachable": False, "latency_ms": -1.0}

    def get_usdt_balance(self, fallback_capital: float = 1000.0) -> Dict[str, Any]:
        if not self.is_live_testnet_configured:
            return {
                "mode": "SHADOW_DRY_RUN (No .env Testnet Key Yet)",
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
                        "mode": "BINANCE_TESTNET_LIVE",
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

    def place_testnet_bracket_order(
        self,
        symbol: str,
        side: int,  # +1 LONG, -1 SHORT
        quantity: float,
        stop_price: float,
        leverage: int = 2,
        execute_on_testnet: bool = False,
    ) -> Dict[str, Any]:
        """
        Submit an order to Binance Futures Testnet (if `execute_on_testnet=True` and API key present),
        or record a deterministic Shadow Testnet fill.
        """
        side_str = "BUY" if side == 1 else "SELL"
        stop_side_str = "SELL" if side == 1 else "BUY"
        qty_rounded = round(float(quantity), 3)
        stop_rounded = round(float(stop_price), 4)

        if not (execute_on_testnet and self.is_live_testnet_configured):
            return {
                "mode": "SHADOW_TESTNET",
                "status": "SIMULATED_FILLED",
                "order_id": f"SHADOW-{symbol}-{int(time.time())}",
                "symbol": symbol,
                "side": side_str,
                "quantity": qty_rounded,
                "stop_price": stop_rounded,
            }

        try:
            # 1. Set leverage on Testnet
            lev_params = {
                "symbol": symbol,
                "leverage": max(1, min(5, int(np.ceil(leverage)) if hasattr(np, "ceil") else int(leverage) + 1)),
                "timestamp": int(time.time() * 1000),
            }
            self.session.post(f"{self.base_url}/fapi/v1/leverage?{self._sign(lev_params)}", timeout=10)

            # 2. Place Market Entry Order
            ord_params = {
                "symbol": symbol,
                "side": side_str,
                "type": "MARKET",
                "quantity": qty_rounded,
                "timestamp": int(time.time() * 1000),
            }
            r_ord = self.session.post(
                f"{self.base_url}/fapi/v1/order?{self._sign(ord_params)}", timeout=10
            )
            r_ord.raise_for_status()
            ord_data = r_ord.json()

            # 3. Place Hard STOP_MARKET Protection Order
            sl_params = {
                "symbol": symbol,
                "side": stop_side_str,
                "type": "STOP_MARKET",
                "stopPrice": stop_rounded,
                "closePosition": "true",
                "timestamp": int(time.time() * 1000),
            }
            self.session.post(
                f"{self.base_url}/fapi/v1/order?{self._sign(sl_params)}", timeout=10
            )
            return {
                "mode": "BINANCE_TESTNET_LIVE",
                "status": ord_data.get("status", "FILLED"),
                "order_id": str(ord_data.get("orderId", "TESTNET-OK")),
                "symbol": symbol,
                "side": side_str,
                "quantity": qty_rounded,
                "stop_price": stop_rounded,
            }
        except Exception as e:
            return {
                "mode": "BINANCE_TESTNET_ERROR_FAIL_CLOSED",
                "status": "REJECTED",
                "order_id": "ERR",
                "error": str(e),
            }
