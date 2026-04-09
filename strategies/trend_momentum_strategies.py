"""
trend_momentum_strategies.py

5 estrategias de alta frecuencia optimizadas para 1H.
Diseñadas para generar 30-100 trades/mes (el sweet spot de Kelly en 1H).

  1. DualMomentumStrategy     — momentum corto vs largo plazo (cross)
  2. TrendStrengthStrategy    — ADX + DI con entrada en pull-back
  3. VolumeTrendStrategy      — tendencia confirmada por volumen relativo
  4. AdaptiveChannelStrategy  — canal dinámico ATR con breakout en tendencia
  5. MultiSignalStrategy      — scoring de 5 indicadores, entra con ≥3 de 5
"""

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ema(arr, p):
    a = 2 / (p + 1)
    out = np.full(len(arr), np.nan)
    if len(arr) < p:
        return out
    out[p - 1] = arr[:p].mean()
    for i in range(p, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out

def _sma(arr, p):
    out = np.full(len(arr), np.nan)
    for i in range(p - 1, len(arr)):
        out[i] = arr[i - p + 1 : i + 1].mean()
    return out

def _atr(h, l, c, p):
    n = len(c)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    out = np.full(n, np.nan)
    if n < p:
        return out
    out[p - 1] = tr[:p].mean()
    for i in range(p, n):
        out[i] = (out[i-1] * (p-1) + tr[i]) / p
    return out

def _rsi(c, p=14):
    n = len(c)
    out = np.full(n, np.nan)
    if n < p + 1:
        return out
    d = np.diff(c)
    g = np.where(d > 0, d, 0.0)
    lo = np.where(d < 0, -d, 0.0)
    ag, al = g[:p].mean(), lo[:p].mean()
    for i in range(p, n - 1):
        ag = (ag * (p-1) + g[i]) / p
        al = (al * (p-1) + lo[i]) / p
        out[i+1] = 100 - 100 / (1 + ag / (al + 1e-10))
    return out

def _adx_pdi_ndi(h, l, c, p=14):
    """Retorna (ADX, +DI, -DI) arrays length=n."""
    n = len(c)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))

    pdm = np.where((h[1:]-h[:-1]) > (l[:-1]-l[1:]),
                   np.maximum(h[1:]-h[:-1], 0), 0.0)
    ndm = np.where((l[:-1]-l[1:]) > (h[1:]-h[:-1]),
                   np.maximum(l[:-1]-l[1:], 0), 0.0)

    def _ws(arr):          # Wilder smooth
        out = np.zeros(len(arr)+1)
        out[p] = arr[:p].sum()
        for i in range(p, len(arr)):
            out[i+1] = out[i] - out[i]/p + arr[i]
        return out[1:]

    atr14 = _ws(tr[1:])
    pdm14 = _ws(pdm)
    ndm14 = _ws(ndm)
    pdi = np.where(atr14 > 0, 100*pdm14/atr14, 0.0)
    ndi = np.where(atr14 > 0, 100*ndm14/atr14, 0.0)
    dx  = np.where(pdi+ndi > 0, 100*np.abs(pdi-ndi)/(pdi+ndi), 0.0)

    adx = np.zeros(len(dx))
    if len(dx) >= p:
        adx[p-1] = dx[:p].mean()
        for i in range(p, len(dx)):
            adx[i] = (adx[i-1]*(p-1)+dx[i])/p

    pad = np.zeros(1)
    return (np.concatenate([pad, adx])[:n],
            np.concatenate([pad, pdi])[:n],
            np.concatenate([pad, ndi])[:n])


# ── 1. Dual Momentum ──────────────────────────────────────────────────────────

class DualMomentumStrategy:
    """
    Momentum dual: compara retorno de corto plazo vs largo plazo.

    mom_fast = (close / close[fast_p] - 1) * 100
    mom_slow = (close / close[slow_p] - 1) * 100

    Buy:  mom_fast > mom_slow > 0  (aceleración alcista)
    Sell: mom_fast < mom_slow < 0  (aceleración bajista)

    Genera señal en cada cruce → ~40-60 señales/mes en 1H.
    """

    def __init__(self, fast_p: int = 12, slow_p: int = 48,
                 ema_trend: int = 50):
        self.fast_p    = fast_p
        self.slow_p    = slow_p
        self.ema_trend = ema_trend

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        c    = df["close"].values
        n    = len(c)
        ema  = _ema(c, self.ema_trend)
        sigs = ["hold"] * n

        for i in range(self.slow_p, n):
            if np.isnan(ema[i]):
                continue
            mf = (c[i] / c[i - self.fast_p] - 1) * 100
            ms = (c[i] / c[i - self.slow_p] - 1) * 100

            # Estado anterior
            mf_p = (c[i-1] / c[i-1-self.fast_p] - 1)*100 if i >= self.fast_p+1 else 0
            ms_p = (c[i-1] / c[i-1-self.slow_p] - 1)*100 if i >= self.slow_p+1 else 0

            bull_now  = mf > ms > 0 and c[i] > ema[i]
            bull_prev = mf_p > ms_p > 0
            bear_now  = mf < ms < 0 and c[i] < ema[i]
            bear_prev = mf_p < ms_p < 0

            if bull_now and not bull_prev:
                sigs[i] = "buy"
            elif bear_now and not bear_prev:
                sigs[i] = "sell"

        return sigs


# ── 2. Trend Strength (ADX Pull-back) ────────────────────────────────────────

class TrendStrengthStrategy:
    """
    Entra en pull-backs dentro de tendencias fuertes (ADX>25).

    Buy:  ADX>25 + +DI>-DI + RSI retrocede a 40-55 (pull-back en tendencia)
    Sell: ADX>25 + -DI>+DI + RSI sube a 45-60 (pull-back en tendencia bajista)

    Muy alta frecuencia en mercados trending.
    """

    def __init__(self, adx_min: float = 20.0, adx_period: int = 14,
                 rsi_period: int = 14,
                 buy_rsi_lo: float = 38.0, buy_rsi_hi: float = 55.0,
                 sell_rsi_lo: float = 45.0, sell_rsi_hi: float = 62.0):
        self.adx_min    = adx_min
        self.adx_period = adx_period
        self.rsi_period = rsi_period
        self.buy_rsi_lo  = buy_rsi_lo
        self.buy_rsi_hi  = buy_rsi_hi
        self.sell_rsi_lo = sell_rsi_lo
        self.sell_rsi_hi = sell_rsi_hi

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values

        adx, pdi, ndi = _adx_pdi_ndi(h, l, c, self.adx_period)
        rsi = _rsi(c, self.rsi_period)
        n   = len(c)
        sigs = ["hold"] * n

        for i in range(1, n):
            if np.isnan(adx[i]) or np.isnan(rsi[i]):
                continue
            if (adx[i] > self.adx_min and pdi[i] > ndi[i] and
                    self.buy_rsi_lo <= rsi[i] <= self.buy_rsi_hi and
                    rsi[i] > rsi[i-1]):
                sigs[i] = "buy"
            elif (adx[i] > self.adx_min and ndi[i] > pdi[i] and
                    self.sell_rsi_lo <= rsi[i] <= self.sell_rsi_hi and
                    rsi[i] < rsi[i-1]):
                sigs[i] = "sell"

        return sigs


# ── 3. Volume Trend ───────────────────────────────────────────────────────────

class VolumeTrendStrategy:
    """
    Tendencia de precio confirmada por expansión de volumen.

    Buy:  EMA_fast > EMA_slow (tendencia alcista)
          + volumen actual > vol_ma * vol_threshold (volumen expandido)
          + precio > EMA_fast (confirmación)
    Sell: EMA_fast < EMA_slow
          + volumen expandido
          + precio < EMA_fast

    El volumen como confirmador reduce las señales falsas significativamente.
    """

    def __init__(self, ema_fast: int = 12, ema_slow: int = 26,
                 vol_period: int = 20, vol_threshold: float = 1.3):
        self.ema_fast      = ema_fast
        self.ema_slow      = ema_slow
        self.vol_period    = vol_period
        self.vol_threshold = vol_threshold

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        c    = df["close"].values
        v    = df["volume"].values
        n    = len(c)

        ef   = _ema(c, self.ema_fast)
        es   = _ema(c, self.ema_slow)
        vol_ma = _sma(v, self.vol_period)
        sigs = ["hold"] * n

        for i in range(1, n):
            if np.isnan(ef[i]) or np.isnan(es[i]) or np.isnan(vol_ma[i]):
                continue
            vol_exp = v[i] > vol_ma[i] * self.vol_threshold

            bull_now  = ef[i] > es[i] and c[i] > ef[i] and vol_exp
            bull_prev = ef[i-1] > es[i-1] and c[i-1] > ef[i-1]
            bear_now  = ef[i] < es[i] and c[i] < ef[i] and vol_exp
            bear_prev = ef[i-1] < es[i-1] and c[i-1] < ef[i-1]

            if bull_now and not bull_prev:
                sigs[i] = "buy"
            elif bear_now and not bear_prev:
                sigs[i] = "sell"

        return sigs


# ── 4. Adaptive Channel ───────────────────────────────────────────────────────

class AdaptiveChannelStrategy:
    """
    Canal adaptativo ATR — breakout del canal en dirección de tendencia.

    mid   = EMA(close, period)
    upper = mid + atr_mult * ATR(period)
    lower = mid - atr_mult * ATR(period)

    Buy:  precio rompe upper (en tendencia alcista EMA fast > slow)
    Sell: precio rompe lower (en tendencia bajista EMA fast < slow)

    El canal se estrecha en tendencia (ATR bajo) → más señales.
    El canal se expande en volatilidad → menos señales (protección).
    """

    def __init__(self, period: int = 20, atr_mult: float = 1.0,
                 trend_fast: int = 20, trend_slow: int = 50):
        self.period     = period
        self.atr_mult   = atr_mult
        self.trend_fast = trend_fast
        self.trend_slow = trend_slow

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        n = len(c)

        mid   = _ema(c, self.period)
        atr   = _atr(h, l, c, self.period)
        upper = mid + self.atr_mult * atr
        lower = mid - self.atr_mult * atr
        ef    = _ema(c, self.trend_fast)
        es    = _ema(c, self.trend_slow)
        sigs  = ["hold"] * n

        for i in range(1, n):
            if np.isnan(upper[i]) or np.isnan(ef[i]) or np.isnan(es[i]):
                continue
            trend_up   = ef[i] > es[i]
            trend_down = ef[i] < es[i]

            if c[i] > upper[i] and c[i-1] <= upper[i-1] and trend_up:
                sigs[i] = "buy"
            elif c[i] < lower[i] and c[i-1] >= lower[i-1] and trend_down:
                sigs[i] = "sell"

        return sigs


# ── 5. Multi-Signal Scoring ───────────────────────────────────────────────────

class MultiSignalStrategy:
    """
    Sistema de scoring: 5 indicadores votan, entra cuando ≥ score_min coinciden.

    Indicadores:
      1. EMA trend (fast > slow)
      2. RSI zona (>55 bullish / <45 bearish)
      3. MACD histogram positivo/negativo
      4. Precio > / < EMA50
      5. ADX > 20 con DI alignment

    Score 0-5 por dirección. Entra con score >= score_min (default 3).
    Más selectivo = más WR, menos trades.
    """

    def __init__(self, ema_fast: int = 12, ema_slow: int = 26,
                 ema_trend: int = 50, rsi_period: int = 14,
                 adx_period: int = 14, score_min: int = 3):
        self.ema_fast   = ema_fast
        self.ema_slow   = ema_slow
        self.ema_trend  = ema_trend
        self.rsi_period = rsi_period
        self.adx_period = adx_period
        self.score_min  = score_min

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        n = len(c)

        ef   = _ema(c, self.ema_fast)
        es   = _ema(c, self.ema_slow)
        et   = _ema(c, self.ema_trend)
        rsi  = _rsi(c, self.rsi_period)
        adx, pdi, ndi = _adx_pdi_ndi(h, l, c, self.adx_period)

        # MACD histogram
        macd_line = ef - es
        signal    = _ema(macd_line, 9)
        hist      = macd_line - signal

        sigs = ["hold"] * n

        for i in range(1, n):
            if any(np.isnan(x) for x in [ef[i], es[i], et[i], rsi[i],
                                          adx[i], hist[i]]):
                continue

            bull_score = sum([
                ef[i] > es[i],                  # 1. EMA cross alcista
                rsi[i] > 55,                    # 2. RSI bullish zone
                hist[i] > 0 and hist[i] > hist[i-1],  # 3. MACD acelerando
                c[i] > et[i],                   # 4. Sobre EMA50
                adx[i] > 20 and pdi[i] > ndi[i],  # 5. ADX trend up
            ])
            bear_score = sum([
                ef[i] < es[i],
                rsi[i] < 45,
                hist[i] < 0 and hist[i] < hist[i-1],
                c[i] < et[i],
                adx[i] > 20 and ndi[i] > pdi[i],
            ])

            # Cruce de umbral
            prev_bull = sum([
                ef[i-1] > es[i-1], rsi[i-1] > 55,
                hist[i-1] > 0, c[i-1] > et[i-1],
                adx[i-1] > 20 and pdi[i-1] > ndi[i-1],
            ])
            prev_bear = sum([
                ef[i-1] < es[i-1], rsi[i-1] < 45,
                hist[i-1] < 0, c[i-1] < et[i-1],
                adx[i-1] > 20 and ndi[i-1] > pdi[i-1],
            ])

            if bull_score >= self.score_min and prev_bull < self.score_min:
                sigs[i] = "buy"
            elif bear_score >= self.score_min and prev_bear < self.score_min:
                sigs[i] = "sell"

        return sigs
