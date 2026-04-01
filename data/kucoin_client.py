"""
kucoin_client.py

Cliente de datos KuCoin — solo lectura, sin operar.
Usa la API pública (sin autenticación) para obtener datos OHLCV reales.

Endpoints usados (todos públicos):
  GET /api/v1/market/candles          → OHLCV histórico
  GET /api/v1/market/stats            → stats 24h (precio, volumen, cambio)
  GET /api/v1/market/orderbook/level1 → bid/ask actuales
  GET /api/v2/symbols                 → lista de símbolos disponibles

Uso:
    from data.kucoin_client import KuCoinClient

    client = KuCoinClient()

    # OHLCV de las últimas 200 velas de 1h
    df = client.get_ohlcv("BTC-USDT", interval="1hour", limit=200)

    # Stats de mercado en tiempo real
    stats = client.get_ticker("BTC-USDT")

    # Múltiples pares a la vez
    dfs = client.get_multi_ohlcv(["BTC-USDT","ETH-USDT"], interval="15min", limit=500)
"""

import time
import requests
import pandas as pd
import numpy as np
from typing import Optional

# ── Constantes ────────────────────────────────────────────────────────────────

BASE_URL = "https://api.kucoin.com"

# Intervalos válidos en KuCoin
VALID_INTERVALS = {
    "1min", "3min", "5min", "15min", "30min",
    "1hour", "2hour", "4hour", "6hour", "8hour", "12hour",
    "1day", "1week",
}

# Mapeo de aliases comunes → formato KuCoin
INTERVAL_ALIASES = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1hour", "2h": "2hour", "4h": "4hour", "6h": "6hour",
    "8h": "8hour", "12h": "12hour", "1d": "1day", "1w": "1week",
}

# Segundos por intervalo (para calcular startAt)
INTERVAL_SECONDS = {
    "1min": 60, "3min": 180, "5min": 300, "15min": 900, "30min": 1800,
    "1hour": 3600, "2hour": 7200, "4hour": 14400, "6hour": 21600,
    "8hour": 28800, "12hour": 43200, "1day": 86400, "1week": 604800,
}

MAX_CANDLES_PER_REQUEST = 1500   # límite de la API
DEFAULT_TIMEOUT = 10
DEFAULT_RETRIES = 3


class KuCoinClient:
    """
    Cliente de datos KuCoin (solo lectura).

    Parámetros
    ----------
    timeout  : segundos de timeout por request
    retries  : reintentos en caso de error de red
    verbose  : imprimir logs de requests
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT,
                 retries: int = DEFAULT_RETRIES,
                 verbose: bool = False):
        self.timeout = timeout
        self.retries = retries
        self.verbose = verbose
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "trading-bot-lab/1.0"})

    # ── Request base ──────────────────────────────────────────────────

    def _get(self, path: str, params: dict = None) -> dict:
        url = BASE_URL + path
        for attempt in range(self.retries):
            try:
                r = self._session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                if data.get("code") != "200000":
                    raise ValueError(f"KuCoin error: {data.get('msg', 'unknown')}")
                return data.get("data", {})
            except requests.exceptions.RequestException as e:
                if attempt < self.retries - 1:
                    wait = 2 ** attempt
                    if self.verbose:
                        print(f"  [KuCoin] Reintento {attempt+1} en {wait}s — {e}")
                    time.sleep(wait)
                else:
                    raise ConnectionError(f"KuCoin no disponible tras {self.retries} intentos: {e}")

    # ── OHLCV ─────────────────────────────────────────────────────────

    def get_ohlcv(self, symbol: str,
                  interval: str = "1hour",
                  limit: int = 200,
                  start_time: Optional[int] = None,
                  end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Obtiene datos OHLCV históricos.

        Parámetros
        ----------
        symbol   : par de trading, e.g. "BTC-USDT"
        interval : timeframe (1min, 15min, 1hour, 4hour, 1day…)
        limit    : número de velas a obtener (máx 1500)
        start_time : timestamp Unix de inicio (opcional)
        end_time   : timestamp Unix de fin (opcional, default = ahora)

        Retorna
        -------
        DataFrame con columnas: open, high, low, close, volume
        índice: DatetimeIndex UTC
        """
        interval = INTERVAL_ALIASES.get(interval, interval)
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Intervalo inválido: {interval}. Usa: {sorted(VALID_INTERVALS)}")

        limit = min(limit, MAX_CANDLES_PER_REQUEST)

        # Calcular ventana temporal
        now = int(time.time())
        if end_time is None:
            end_time = now
        if start_time is None:
            step = INTERVAL_SECONDS[interval]
            start_time = end_time - step * limit

        params = {
            "symbol":  symbol,
            "type":    interval,
            "startAt": start_time,
            "endAt":   end_time,
        }

        if self.verbose:
            print(f"  [KuCoin] GET {symbol} {interval} limit={limit}")

        raw = self._get("/api/v1/market/candles", params)

        if not raw:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # KuCoin devuelve [timestamp, open, close, high, low, volume, turnover]
        # más reciente primero → invertimos
        rows = []
        for candle in reversed(raw):
            ts, o, c, h, l, vol, _ = candle
            rows.append({
                "timestamp": int(ts),
                "open":      float(o),
                "high":      float(h),
                "low":       float(l),
                "close":     float(c),
                "volume":    float(vol),
            })

        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.drop(columns=["timestamp"])
        df = df.sort_index()

        # Limitar al número pedido
        if len(df) > limit:
            df = df.iloc[-limit:]

        return df

    def get_multi_ohlcv(self, symbols: list,
                         interval: str = "1hour",
                         limit: int = 200) -> dict:
        """
        Obtiene OHLCV para múltiples símbolos.

        Retorna dict {symbol: DataFrame}
        """
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.get_ohlcv(sym, interval, limit)
                if self.verbose:
                    print(f"  [KuCoin] {sym}: {len(result[sym])} velas OK")
            except Exception as e:
                print(f"  [KuCoin] ERROR {sym}: {e}")
                result[sym] = None
        return result

    # ── Ticker / stats ────────────────────────────────────────────────

    def get_ticker(self, symbol: str) -> dict:
        """
        Stats de las últimas 24h: precio, volumen, cambio %.

        Retorna dict con: symbol, high, low, vol, volValue,
                          last, buy, sell, changeRate, changePrice, averagePrice
        """
        data = self._get("/api/v1/market/stats", {"symbol": symbol})
        return {
            "symbol":      symbol,
            "price":       float(data.get("last", 0)),
            "high_24h":    float(data.get("high", 0)),
            "low_24h":     float(data.get("low", 0)),
            "volume_24h":  float(data.get("vol", 0)),
            "change_pct":  float(data.get("changeRate", 0)) * 100,
            "bid":         float(data.get("buy", 0)),
            "ask":         float(data.get("sell", 0)),
        }

    def get_orderbook_top(self, symbol: str) -> dict:
        """Best bid/ask actuales."""
        data = self._get("/api/v1/market/orderbook/level1", {"symbol": symbol})
        return {
            "symbol": symbol,
            "price":  float(data.get("price", 0)),
            "bid":    float(data.get("bestBid", 0)),
            "ask":    float(data.get("bestAsk", 0)),
            "time":   int(data.get("time", 0)),
        }

    # ── Símbolos disponibles ──────────────────────────────────────────

    def list_usdt_symbols(self, min_volume: float = 0) -> list:
        """Lista todos los pares USDT disponibles en KuCoin."""
        data = self._get("/api/v2/symbols")
        symbols = [
            s["symbol"] for s in data
            if s.get("quoteCurrency") == "USDT"
            and s.get("enableTrading", False)
        ]
        return sorted(symbols)

    # ── Utilidades ────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Verifica conectividad con KuCoin."""
        try:
            self._get("/api/v1/market/stats", {"symbol": "BTC-USDT"})
            return True
        except Exception:
            return False

    def get_live_snapshot(self, symbols: list, interval: str = "15min",
                           limit: int = 300) -> dict:
        """
        Snapshot completo para el dashboard/paper trader:
        {symbol: {"ohlcv": df, "ticker": dict}}
        """
        result = {}
        for sym in symbols:
            try:
                df     = self.get_ohlcv(sym, interval, limit)
                ticker = self.get_ticker(sym)
                result[sym] = {"ohlcv": df, "ticker": ticker}
            except Exception as e:
                result[sym] = {"ohlcv": None, "ticker": None, "error": str(e)}
        return result

    def __repr__(self):
        return f"KuCoinClient(timeout={self.timeout}, retries={self.retries})"
