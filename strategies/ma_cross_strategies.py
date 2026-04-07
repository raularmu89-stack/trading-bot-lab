"""
ma_cross_strategies.py

Estrategias basadas en medias móviles, incluyendo la MA300/MA1000
(inspirada en estrategias populares de Freqtrade).

  1. MA300_1000Strategy  — precio > MA300 y MA1000 para entrar, MA300 para salir
  2. EMAStackStrategy    — triple EMA stacking (rápida > media > lenta)
  3. GoldenCrossStrategy — SMA50/200 golden cross clásico
  4. DynamicMAStrategy   — MA adaptativa por volatilidad (ATR regula periodo)
"""

import numpy as np
import pandas as pd


def _sma(arr, p):
    out = np.full(len(arr), np.nan)
    for i in range(p - 1, len(arr)):
        out[i] = arr[i - p + 1 : i + 1].mean()
    return out


def _ema(arr, p):
    a = 2 / (p + 1)
    out = np.full(len(arr), np.nan)
    if len(arr) < p:
        return out
    out[p - 1] = arr[:p].mean()
    for i in range(p, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out


def _atr(h, l, c, p):
    n = len(c)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    out = np.full(n, np.nan)
    if n < p:
        return out
    out[p - 1] = tr[:p].mean()
    for i in range(p, n):
        out[i] = (out[i - 1] * (p - 1) + tr[i]) / p
    return out


class MA300_1000Strategy:
    """
    Adaptación de la estrategia Freqtrade MA300/MA1000 a nuestro framework.

    Lógica original:
      LONG  cuando close > MA300 AND close > MA1000
      EXIT  cuando close < MA300

    Adaptación para señales batch:
      "buy"  → close cruza al alza MA300 estando ya sobre MA1000
      "sell" → close cruza a la baja MA300 (salida/short)
    """

    def __init__(
        self,
        ma_fast: int = 300,
        ma_slow: int = 1000,
        ma_type: str = "sma",   # "sma" | "ema"
    ):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.ma_type = ma_type

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        c = df["close"].values
        fn = _sma if self.ma_type == "sma" else _ema

        fast = fn(c, self.ma_fast)
        slow = fn(c, self.ma_slow)

        n = len(c)
        sigs = ["hold"] * n

        for i in range(1, n):
            if np.isnan(fast[i]) or np.isnan(slow[i]):
                continue
            # Entrada: precio cruza al alza MA_fast Y está sobre MA_slow
            if c[i] > fast[i] and c[i - 1] <= fast[i - 1] and c[i] > slow[i]:
                sigs[i] = "buy"
            # Salida/short: precio cruza a la baja MA_fast
            elif c[i] < fast[i] and c[i - 1] >= fast[i - 1]:
                sigs[i] = "sell"

        return sigs

    def __repr__(self):
        return f"MA{self.ma_fast}/{self.ma_slow}({self.ma_type.upper()})"


class EMAStackStrategy:
    """
    Triple EMA stacking: fast > mid > slow → buy, fast < mid < slow → sell.
    Más reactivo que SMA300/1000 gracias a la ponderación exponencial.
    """

    def __init__(self, fast: int = 50, mid: int = 100, slow: int = 200):
        self.fast = fast
        self.mid  = mid
        self.slow = slow

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        c    = df["close"].values
        ef   = _ema(c, self.fast)
        em   = _ema(c, self.mid)
        es   = _ema(c, self.slow)
        n    = len(c)
        sigs = ["hold"] * n

        for i in range(1, n):
            if np.isnan(ef[i]) or np.isnan(em[i]) or np.isnan(es[i]):
                continue
            bull_now  = ef[i] > em[i] > es[i]
            bull_prev = ef[i-1] > em[i-1] > es[i-1]
            bear_now  = ef[i] < em[i] < es[i]
            bear_prev = ef[i-1] < em[i-1] < es[i-1]

            if bull_now and not bull_prev:
                sigs[i] = "buy"
            elif bear_now and not bear_prev:
                sigs[i] = "sell"

        return sigs


class GoldenCrossStrategy:
    """
    Golden Cross SMA50/200 — clásico institucional.
    Buy: SMA50 cruza al alza SMA200.
    Sell: SMA50 cruza a la baja SMA200.
    """

    def __init__(self, fast: int = 50, slow: int = 200):
        self.fast = fast
        self.slow = slow

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        c    = df["close"].values
        sf   = _sma(c, self.fast)
        ss   = _sma(c, self.slow)
        n    = len(c)
        sigs = ["hold"] * n

        for i in range(1, n):
            if np.isnan(sf[i]) or np.isnan(ss[i]):
                continue
            if sf[i] > ss[i] and sf[i-1] <= ss[i-1]:
                sigs[i] = "buy"
            elif sf[i] < ss[i] and sf[i-1] >= ss[i-1]:
                sigs[i] = "sell"

        return sigs


class DynamicMAStrategy:
    """
    MA adaptativa — el periodo de la MA rápida se ajusta según volatilidad ATR.
    Mercados tranquilos → MA lenta (más señales filtradas).
    Mercados volátiles → MA rápida (más reactividad).
    """

    def __init__(
        self,
        fast_min: int   = 20,
        fast_max: int   = 100,
        slow:     int   = 200,
        atr_period: int = 14,
        vol_smooth: int = 50,
    ):
        self.fast_min   = fast_min
        self.fast_max   = fast_max
        self.slow       = slow
        self.atr_period = atr_period
        self.vol_smooth = vol_smooth

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        c  = df["close"].values
        h  = df["high"].values
        lo = df["low"].values
        n  = len(c)

        atr_arr  = _atr(h, lo, c, self.atr_period)
        atr_pct  = np.where(c > 0, atr_arr / c * 100, np.nan)
        atr_med  = _sma(np.nan_to_num(atr_pct, nan=0.0), self.vol_smooth)
        atr_std  = np.full(n, np.nan)
        for i in range(self.vol_smooth - 1, n):
            atr_std[i] = np.std(atr_pct[i - self.vol_smooth + 1 : i + 1])

        slow_ma = _sma(c, self.slow)

        # Precalcular todas las fast MAs necesarias
        fast_arrays = {}
        for p in range(self.fast_min, self.fast_max + 1, 5):
            fast_arrays[p] = _sma(c, p)

        sigs = ["hold"] * n

        for i in range(1, n):
            if np.isnan(slow_ma[i]) or np.isnan(atr_pct[i]):
                continue

            # Elegir periodo dinámico: más alta volatilidad → MA más rápida
            vol_rank = (atr_pct[i] - atr_med[i]) / (atr_std[i] + 1e-8) \
                       if not np.isnan(atr_std[i]) else 0.0
            # norm -2..2 → 0..1
            norm = min(max((vol_rank + 2) / 4, 0.0), 1.0)
            fast_p = int(self.fast_max - norm * (self.fast_max - self.fast_min))
            fast_p = max(self.fast_min,
                         min(self.fast_max,
                             round(fast_p / 5) * 5))

            fast_ma  = fast_arrays.get(fast_p, fast_arrays[self.fast_min])

            if np.isnan(fast_ma[i]) or np.isnan(fast_ma[i-1]):
                continue

            if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
                sigs[i] = "buy"
            elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
                sigs[i] = "sell"

        return sigs
