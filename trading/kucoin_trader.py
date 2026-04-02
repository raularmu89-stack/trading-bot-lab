"""
kucoin_trader.py

Cliente KuCoin autenticado — ejecución de órdenes reales + paper trading.

PAPER MODE (default): simula órdenes sin tocar dinero real.
  → Perfecto para validar la estrategia en tiempo real antes de arriesgar.

REAL MODE: requiere API Key, Secret y Passphrase de KuCoin.
  → Activa con: trader = KuCoinTrader(paper=False, api_key=..., ...)
  → Genera las claves en: KuCoin → API Management → Create API
  → Permisos necesarios: "Trade" (NO necesitas "Transfer")

Endpoints usados:
  POST /api/v1/orders          → crear orden
  DELETE /api/v1/orders/{id}   → cancelar orden
  GET  /api/v1/orders/{id}     → estado de orden
  GET  /api/v1/accounts        → balance de cuenta
  GET  /api/v1/positions       → posiciones abiertas (futures)

Uso:
    from trading.kucoin_trader import KuCoinTrader

    # Paper trading (seguro)
    trader = KuCoinTrader(paper=True)
    order = trader.market_buy("BTC-USDT", usdt_amount=50)

    # Real trading
    trader = KuCoinTrader(
        paper=False,
        api_key="tu_key",
        api_secret="tu_secret",
        api_passphrase="tu_passphrase",
    )
"""

import time
import hmac
import base64
import hashlib
import json
import uuid
import requests
import pandas as pd
from typing import Optional


BASE_URL = "https://api.kucoin.com"
FEE_TAKER = 0.001   # 0.1% taker fee


class KuCoinTrader:
    """
    Ejecutor de órdenes KuCoin con modo paper trading integrado.

    Parámetros
    ----------
    paper          : True = simular sin dinero real (default)
    api_key        : KuCoin API Key (solo real mode)
    api_secret     : KuCoin API Secret (solo real mode)
    api_passphrase : KuCoin API Passphrase (solo real mode)
    initial_balance: Balance inicial para paper trading (USDT)
    """

    def __init__(
        self,
        paper: bool = True,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        initial_balance: float = 1000.0,
    ):
        self.paper          = paper
        self._api_key       = api_key
        self._api_secret    = api_secret
        self._api_passphrase = api_passphrase

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "trading-bot-lab/1.0"})

        # Paper trading state
        self._paper_balance  = {"USDT": initial_balance}
        self._paper_orders   = {}   # order_id → order dict
        self._paper_history  = []   # fills históricos

        mode = "PAPER" if paper else "REAL"
        print(f"  [KuCoinTrader] Modo: {mode}  |  Balance inicial: {initial_balance} USDT")

    # ── Autenticación ──────────────────────────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> dict:
        """Genera headers de autenticación KuCoin."""
        msg     = f"{timestamp}{method.upper()}{path}{body}"
        sig     = base64.b64encode(
            hmac.new(self._api_secret.encode(), msg.encode(), hashlib.sha256).digest()
        ).decode()
        pp_enc  = base64.b64encode(
            hmac.new(self._api_secret.encode(),
                     self._api_passphrase.encode(), hashlib.sha256).digest()
        ).decode()
        return {
            "KC-API-KEY":         self._api_key,
            "KC-API-SIGN":        sig,
            "KC-API-TIMESTAMP":   timestamp,
            "KC-API-PASSPHRASE":  pp_enc,
            "KC-API-KEY-VERSION": "2",
            "Content-Type":       "application/json",
        }

    def _request(self, method: str, path: str, body: dict = None) -> dict:
        """Realiza una petición autenticada a la API real."""
        timestamp = str(int(time.time() * 1000))
        body_str  = json.dumps(body) if body else ""
        headers   = self._sign(timestamp, method, path, body_str)
        url       = BASE_URL + path

        for attempt in range(3):
            try:
                if method.upper() == "GET":
                    r = self._session.get(url, headers=headers, timeout=10)
                elif method.upper() == "POST":
                    r = self._session.post(url, headers=headers,
                                          data=body_str, timeout=10)
                elif method.upper() == "DELETE":
                    r = self._session.delete(url, headers=headers, timeout=10)
                else:
                    raise ValueError(f"Método no soportado: {method}")

                r.raise_for_status()
                data = r.json()
                if data.get("code") != "200000":
                    raise ValueError(f"KuCoin error: {data.get('msg')}")
                return data.get("data", {})
            except requests.exceptions.RequestException as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise ConnectionError(f"KuCoin API error: {e}")

    # ── Precio actual ──────────────────────────────────────────────────────

    def get_price(self, symbol: str) -> float:
        """Precio bid/ask actual del símbolo."""
        try:
            r    = self._session.get(
                f"{BASE_URL}/api/v1/market/orderbook/level1",
                params={"symbol": symbol}, timeout=10
            )
            data = r.json().get("data", {})
            return float(data.get("price", 0))
        except Exception:
            return 0.0

    # ── Balance ────────────────────────────────────────────────────────────

    def get_balance(self, currency: str = "USDT") -> float:
        """Balance disponible de una moneda."""
        if self.paper:
            return self._paper_balance.get(currency, 0.0)

        data     = self._request("GET", "/api/v1/accounts")
        accounts = [a for a in data if a["currency"] == currency
                    and a["type"] == "trade"]
        return float(accounts[0]["available"]) if accounts else 0.0

    def get_all_balances(self) -> dict:
        """Todos los balances (solo monedas con valor > 0)."""
        if self.paper:
            return {k: v for k, v in self._paper_balance.items() if v > 0}

        data = self._request("GET", "/api/v1/accounts")
        return {a["currency"]: float(a["available"])
                for a in data if float(a["available"]) > 0
                and a["type"] == "trade"}

    # ── Órdenes ────────────────────────────────────────────────────────────

    def market_buy(self, symbol: str, usdt_amount: float) -> dict:
        """
        Compra a mercado por importe en USDT.

        Devuelve dict con: order_id, symbol, side, size, price, fee, timestamp
        """
        price = self.get_price(symbol)
        if price <= 0:
            raise ValueError(f"No se pudo obtener precio de {symbol}")

        size = usdt_amount / price
        fee  = usdt_amount * FEE_TAKER

        if self.paper:
            return self._paper_fill("buy", symbol, size, price, fee, usdt_amount)

        body = {
            "clientOid": str(uuid.uuid4()),
            "side":      "buy",
            "symbol":    symbol,
            "type":      "market",
            "funds":     str(round(usdt_amount, 4)),
        }
        data = self._request("POST", "/api/v1/orders", body)
        return {
            "order_id":  data.get("orderId"),
            "symbol":    symbol,
            "side":      "buy",
            "size":      size,
            "price":     price,
            "fee":       fee,
            "usdt_cost": usdt_amount,
            "timestamp": time.time(),
        }

    def market_sell(self, symbol: str, size: float) -> dict:
        """
        Venta a mercado por cantidad de la moneda.

        Devuelve dict con: order_id, symbol, side, size, price, fee, usdt_received
        """
        price        = self.get_price(symbol)
        usdt_received = price * size
        fee          = usdt_received * FEE_TAKER

        if self.paper:
            return self._paper_fill("sell", symbol, size, price, fee, usdt_received)

        body = {
            "clientOid": str(uuid.uuid4()),
            "side":      "sell",
            "symbol":    symbol,
            "type":      "market",
            "size":      str(round(size, 8)),
        }
        data = self._request("POST", "/api/v1/orders", body)
        return {
            "order_id":      data.get("orderId"),
            "symbol":        symbol,
            "side":          "sell",
            "size":          size,
            "price":         price,
            "fee":           fee,
            "usdt_received": usdt_received - fee,
            "timestamp":     time.time(),
        }

    def cancel_order(self, order_id: str) -> bool:
        """Cancela una orden pendiente."""
        if self.paper:
            self._paper_orders.pop(order_id, None)
            return True
        try:
            self._request("DELETE", f"/api/v1/orders/{order_id}")
            return True
        except Exception:
            return False

    def get_order(self, order_id: str) -> dict:
        """Estado de una orden."""
        if self.paper:
            return self._paper_orders.get(order_id, {})
        return self._request("GET", f"/api/v1/orders/{order_id}")

    # ── Paper trading ──────────────────────────────────────────────────────

    def _paper_fill(self, side: str, symbol: str, size: float,
                    price: float, fee: float, value: float) -> dict:
        """Simula un fill inmediato en paper mode."""
        base_currency = symbol.split("-")[0]
        order_id      = f"PAPER-{uuid.uuid4().hex[:8].upper()}"

        if side == "buy":
            cost = value + fee
            if self._paper_balance.get("USDT", 0) < cost:
                raise ValueError(
                    f"Paper: balance insuficiente. "
                    f"Necesitas {cost:.2f} USDT, tienes "
                    f"{self._paper_balance.get('USDT', 0):.2f}"
                )
            self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0) - cost
            self._paper_balance[base_currency] = (
                self._paper_balance.get(base_currency, 0) + size
            )
            result = {
                "order_id":  order_id,
                "symbol":    symbol,
                "side":      "buy",
                "size":      size,
                "price":     price,
                "fee":       fee,
                "usdt_cost": cost,
                "timestamp": time.time(),
            }
        else:
            received = value - fee
            if self._paper_balance.get(base_currency, 0) < size:
                raise ValueError(
                    f"Paper: balance insuficiente de {base_currency}. "
                    f"Necesitas {size:.8f}, tienes "
                    f"{self._paper_balance.get(base_currency, 0):.8f}"
                )
            self._paper_balance[base_currency] = (
                self._paper_balance.get(base_currency, 0) - size
            )
            self._paper_balance["USDT"] = (
                self._paper_balance.get("USDT", 0) + received
            )
            result = {
                "order_id":      order_id,
                "symbol":        symbol,
                "side":          "sell",
                "size":          size,
                "price":         price,
                "fee":           fee,
                "usdt_received": received,
                "timestamp":     time.time(),
            }

        self._paper_orders[order_id] = result
        self._paper_history.append(result)

        bal_usdt = self._paper_balance.get("USDT", 0)
        print(f"  [PAPER] {side.upper():4s} {symbol}  "
              f"size={size:.6f}  price={price:.4f}  "
              f"fee={fee:.4f}  USDT balance={bal_usdt:.2f}")
        return result

    def paper_summary(self) -> dict:
        """Resumen del paper trading: P&L, trades, balance."""
        trades   = len(self._paper_history)
        buys     = [o for o in self._paper_history if o["side"] == "buy"]
        sells    = [o for o in self._paper_history if o["side"] == "sell"]
        fees     = sum(o["fee"] for o in self._paper_history)
        usdt_bal = self._paper_balance.get("USDT", 0)

        # Valorar posiciones abiertas al último precio conocido
        open_val = 0.0
        for cur, amt in self._paper_balance.items():
            if cur != "USDT" and amt > 0:
                price    = self.get_price(f"{cur}-USDT")
                open_val += price * amt

        total_value = usdt_bal + open_val
        return {
            "usdt_balance": round(usdt_bal, 2),
            "open_positions_value": round(open_val, 2),
            "total_value": round(total_value, 2),
            "total_trades": trades,
            "buys": len(buys),
            "sells": len(sells),
            "total_fees": round(fees, 4),
            "balances": {k: round(v, 8) for k, v in self._paper_balance.items() if v > 0},
        }

    def trade_history_df(self) -> pd.DataFrame:
        """Historial de trades como DataFrame."""
        if not self._paper_history:
            return pd.DataFrame()
        df = pd.DataFrame(self._paper_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df
