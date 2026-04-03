"""
pair_scanner.py

Escáner dinámico de pares KuCoin — selecciona los mejores N pares en tiempo real
basándose en momentum, volatilidad y volumen.

Score por par = 0.4×momentum + 0.3×volumen_rank + 0.3×volatilidad_rank
  - Momentum    : cambio de precio % en las últimas `momentum_period` velas
  - Volumen     : volumen relativo vs su media (`vol_period` velas)
  - Volatilidad : ATR% normalizado (buscamos mercados con movimiento suficiente)

Uso:
    from trading.pair_scanner import PairScanner
    from data.kucoin_client import KuCoinClient

    scanner = PairScanner(client=KuCoinClient(), top_n=10)
    top_pairs = scanner.scan()
    # → ["BTC-USDT", "SOL-USDT", "ETH-USDT", ...]

    # Con caché (para no spamear la API en el bucle de trading):
    top_pairs = scanner.scan(use_cache_seconds=300)
"""

import time
import numpy as np
import pandas as pd
from typing import Optional


# ── Universo por defecto ──────────────────────────────────────────────────────

DEFAULT_UNIVERSE = [
    # Majors
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "XRP-USDT", "ADA-USDT",
    "SOL-USDT", "DOGE-USDT", "DOT-USDT", "AVAX-USDT", "LTC-USDT",
    # Mid-caps con buen volumen en KuCoin
    "LINK-USDT", "UNI-USDT", "ATOM-USDT", "NEAR-USDT", "APT-USDT",
    "ARB-USDT",  "OP-USDT",  "FIL-USDT",  "INJ-USDT",  "SUI-USDT",
    "TIA-USDT",  "JUP-USDT", "WIF-USDT",  "PEPE-USDT", "SHIB-USDT",
]


class PairScanner:
    """
    Escanea un universo de pares USDT y devuelve el ranking por score.

    Parámetros
    ----------
    client          : KuCoinClient (o cualquier objeto con get_ohlcv())
    universe        : lista de símbolos a escanear
    top_n           : número de pares a retornar
    interval        : timeframe para el escáner (default "1hour")
    candles         : cuántas velas cargar por par (default 50)
    momentum_period : velas para calcular momentum (default 20)
    vol_period      : velas para volumen relativo (default 20)
    atr_period      : periodo ATR para volatilidad (default 14)
    min_volume_usdt : volumen mínimo 24h para considerar el par (default 500k)
    """

    def __init__(
        self,
        client,
        universe: Optional[list] = None,
        top_n: int = 10,
        interval: str = "1hour",
        candles: int = 50,
        momentum_period: int = 20,
        vol_period: int = 20,
        atr_period: int = 14,
        min_volume_usdt: float = 500_000,
    ):
        self.client          = client
        self.universe        = universe or DEFAULT_UNIVERSE
        self.top_n           = top_n
        self.interval        = interval
        self.candles         = candles
        self.momentum_period = momentum_period
        self.vol_period      = vol_period
        self.atr_period      = atr_period
        self.min_volume_usdt = min_volume_usdt

        self._cache_pairs: list = []
        self._cache_scores: pd.DataFrame = pd.DataFrame()
        self._cache_ts: float = 0.0

    # ── Escaneo principal ──────────────────────────────────────────────────

    def scan(self, use_cache_seconds: float = 0) -> list[str]:
        """
        Escanea el universo y retorna los top_n pares ordenados por score.

        use_cache_seconds > 0 → usa caché si tiene menos de N segundos.
        """
        if use_cache_seconds > 0:
            age = time.time() - self._cache_ts
            if age < use_cache_seconds and self._cache_pairs:
                return self._cache_pairs

        scores = self._score_all()
        if scores.empty:
            return self._cache_pairs or self.universe[:self.top_n]

        top = scores.head(self.top_n)["symbol"].tolist()

        self._cache_pairs  = top
        self._cache_scores = scores
        self._cache_ts     = time.time()

        return top

    def scan_with_scores(self, use_cache_seconds: float = 0) -> pd.DataFrame:
        """
        Como scan() pero retorna el DataFrame completo con scores.

        Columnas: symbol, momentum_pct, vol_ratio, atr_pct, score, rank
        """
        self.scan(use_cache_seconds=use_cache_seconds)
        return self._cache_scores

    # ── Scoring ──────────────────────────────────────────────────────────

    def _score_all(self) -> pd.DataFrame:
        rows = []
        for symbol in self.universe:
            try:
                row = self._score_pair(symbol)
                if row:
                    rows.append(row)
            except Exception:
                pass
            time.sleep(0.08)   # rate limit KuCoin

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Rank cada métrica 0→1 (mayor = mejor)
        df["mom_rank"] = _rank_norm(df["momentum_pct"].abs())
        df["vol_rank"] = _rank_norm(df["vol_ratio"])
        df["atr_rank"] = _rank_norm(df["atr_pct"])

        # Score compuesto
        df["score"] = (
            0.40 * df["mom_rank"] +
            0.30 * df["vol_rank"] +
            0.30 * df["atr_rank"]
        )

        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        return df[["symbol", "momentum_pct", "vol_ratio", "atr_pct",
                   "mom_rank", "vol_rank", "atr_rank", "score", "rank"]]

    def _score_pair(self, symbol: str) -> Optional[dict]:
        df = self.client.get_ohlcv(symbol, interval=self.interval, limit=self.candles)
        if df is None or len(df) < max(self.momentum_period, self.vol_period, self.atr_period) + 5:
            return None

        c = df["close"].values
        h = df["high"].values
        lo = df["low"].values
        v = df["volume"].values

        # Momentum: cambio % en momentum_period velas
        if len(c) < self.momentum_period + 1:
            return None
        momentum_pct = (c[-1] - c[-self.momentum_period - 1]) / c[-self.momentum_period - 1] * 100

        # Volumen relativo: último vs media
        if len(v) < self.vol_period + 1:
            return None
        vol_avg = v[-self.vol_period - 1:-1].mean()
        vol_ratio = v[-1] / vol_avg if vol_avg > 0 else 1.0

        # Filtro mínimo de volumen USDT (last candle)
        vol_usdt = v[-1] * c[-1]
        if vol_usdt < self.min_volume_usdt / (24 if "hour" in self.interval else 1):
            return None

        # ATR% (volatilidad relativa)
        atr = _calc_atr(h, lo, c, self.atr_period)
        atr_pct = atr / c[-1] * 100 if c[-1] > 0 else 0.0

        return {
            "symbol":       symbol,
            "momentum_pct": round(momentum_pct, 4),
            "vol_ratio":    round(vol_ratio, 4),
            "atr_pct":      round(atr_pct, 4),
        }

    # ── Escaneo de todo KuCoin (modo avanzado) ────────────────────────────

    def scan_all_kucoin(self, min_vol_24h_usdt: float = 1_000_000) -> list[str]:
        """
        Descarga la lista completa de pares USDT de KuCoin y filtra por volumen.
        Solo disponible con conexión real a la API.
        """
        try:
            import requests
            r = requests.get(
                "https://api.kucoin.com/api/v1/market/allTickers",
                timeout=15
            )
            data = r.json().get("data", {}).get("ticker", [])
        except Exception as e:
            print(f"  [Scanner] Error obteniendo tickers: {e}")
            return self.universe

        pairs = []
        for t in data:
            sym = t.get("symbol", "")
            if not sym.endswith("-USDT"):
                continue
            try:
                vol = float(t.get("volValue", 0))
            except (ValueError, TypeError):
                vol = 0.0
            if vol >= min_vol_24h_usdt:
                pairs.append(sym)

        # Actualizar universo y hacer scan
        if len(pairs) > 5:
            self.universe = sorted(pairs)
            print(f"  [Scanner] Universo actualizado: {len(pairs)} pares")

        return self.scan()

    def print_top(self, n: Optional[int] = None):
        """Imprime tabla de los top pares."""
        df = self._cache_scores
        if df.empty:
            self.scan()
            df = self._cache_scores
        if df.empty:
            print("  [Scanner] Sin datos disponibles")
            return

        top = df.head(n or self.top_n)
        print(f"\n  {'─'*65}")
        print(f"  PAIR SCANNER — Top {len(top)} pares  "
              f"(universo: {len(self.universe)})")
        print(f"  {'─'*65}")
        print(f"  {'#':>3}  {'Symbol':<12} {'Momentum%':>10} "
              f"{'VolRatio':>9} {'ATR%':>7} {'Score':>7}")
        print(f"  {'─'*65}")
        for _, row in top.iterrows():
            print(f"  {int(row['rank']):>3}  {row['symbol']:<12} "
                  f"{row['momentum_pct']:>+10.2f} "
                  f"{row['vol_ratio']:>9.2f} "
                  f"{row['atr_pct']:>7.3f} "
                  f"{row['score']:>7.4f}")
        print(f"  {'─'*65}\n")


# ── Funciones auxiliares ──────────────────────────────────────────────────────

def _rank_norm(series: pd.Series) -> pd.Series:
    """Normaliza un Series a rango [0, 1] (mayor → 1)."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def _calc_atr(h, l, c, period: int) -> float:
    n = len(c)
    if n < 2:
        return float(h[-1] - l[-1]) if n else 0.0
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-period:])) if len(tr) >= period else float(np.mean(tr))


# ── Integración con LiveTrader ────────────────────────────────────────────────

class AdaptivePairManager:
    """
    Wrapper para LiveTrader: mantiene el top-N de pares actualizado
    y añade pares que han entrado/salido del ranking.

    Uso en LiveTrader:
        self.pair_mgr = AdaptivePairManager(scanner, refresh_hours=4)
        # En cada ciclo:
        pairs = self.pair_mgr.get_active_pairs()
    """

    def __init__(self, scanner: PairScanner, refresh_hours: float = 4.0,
                 fallback_pairs: Optional[list] = None):
        self.scanner        = scanner
        self.refresh_secs   = refresh_hours * 3600
        self.fallback       = fallback_pairs or DEFAULT_UNIVERSE[:10]
        self._last_scan: float = 0.0
        self._active_pairs: list = list(self.fallback)

    def get_active_pairs(self) -> list[str]:
        age = time.time() - self._last_scan
        if age >= self.refresh_secs:
            try:
                new_pairs = self.scanner.scan()
                if new_pairs:
                    added   = [p for p in new_pairs if p not in self._active_pairs]
                    removed = [p for p in self._active_pairs if p not in new_pairs]
                    self._active_pairs = new_pairs
                    self._last_scan    = time.time()
                    if added or removed:
                        print(f"  [PairMgr] Actualizado → +{added} -{removed}")
            except Exception as e:
                print(f"  [PairMgr] Error en scan: {e} — usando pares anteriores")
        return self._active_pairs

    def force_refresh(self):
        self._last_scan = 0.0
        return self.get_active_pairs()
