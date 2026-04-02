"""
swing_strategies.py

3 estrategias de swing nuevas, diseñadas para 4H con pocas señales
y alta ganancia media por trade — superan la barrera de comisiones del 0.2%.

  1. ChandelierExitStrategy  — Chandelier Exit (ATR trailing stop flip)
  2. TurtleBreakoutStrategy  — Ruptura Turtle Trader (N-bar high/low)
  3. MultiEMASwingStrategy   — 3 EMAs largas + ADX confirma tendencia
"""

import numpy as np
import pandas as pd


# ── Indicadores ───────────────────────────────────────────────────────────────

def _ema(arr, p):
    a   = 2 / (p + 1)
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
        out[i] = arr[i - p + 1:i + 1].mean()
    return out


def _atr(h, l, c, p):
    n  = len(c)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    out = np.full(n, np.nan)
    if n < p:
        return out
    out[p-1] = tr[:p].mean()
    for i in range(p, n):
        out[i] = (out[i-1] * (p-1) + tr[i]) / p
    return out


def _adx(h, l, c, p=14):
    """Retorna (ADX, +DI, -DI) como arrays."""
    n    = len(c)
    tr   = np.zeros(n)
    pdm  = np.zeros(n)
    ndm  = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i]  = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        up     = h[i] - h[i-1]
        dn     = l[i-1] - l[i]
        pdm[i] = up  if up > dn and up > 0 else 0.0
        ndm[i] = dn  if dn > up and dn > 0 else 0.0

    def _smooth(x):
        out = np.full(n, np.nan)
        if n < p:
            return out
        out[p-1] = x[:p].sum()
        for i in range(p, n):
            out[i] = out[i-1] - out[i-1]/p + x[i]
        return out

    atr14 = _smooth(tr)
    pdm14 = _smooth(pdm)
    ndm14 = _smooth(ndm)
    pdi   = np.full(n, np.nan)
    ndi   = np.full(n, np.nan)
    dx    = np.full(n, np.nan)
    for i in range(p-1, n):
        if atr14[i] > 0:
            pdi[i] = 100 * pdm14[i] / atr14[i]
            ndi[i] = 100 * ndm14[i] / atr14[i]
            diff   = abs(pdi[i] - ndi[i])
            summ   = pdi[i] + ndi[i]
            dx[i]  = 100 * diff / (summ + 1e-10)

    adx = np.full(n, np.nan)
    # ADX = smoothed DX
    start = 2 * p - 2
    if n > start:
        adx[start] = np.nanmean(dx[p-1:start+1])
        for i in range(start+1, n):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * (p-1) + dx[i]) / p
    return adx, pdi, ndi


# ── Base ──────────────────────────────────────────────────────────────────────

class _Base:
    swing_window    = 5
    require_fvg     = False
    use_choch_filter = False
    _min_bars        = 60

    def generate_signal(self, data):
        if data is None or len(data) < self._min_bars:
            return {"signal": "hold", "reason": "insufficient_data"}
        s = self.generate_signals_batch(data)
        return {"signal": s[-1] if s else "hold"}

    def generate_signals_batch(self, data):
        raise NotImplementedError


# ════════════════════════════════════════════════════════════════════════════
# 1. Chandelier Exit
# ════════════════════════════════════════════════════════════════════════════

class ChandelierExitStrategy(_Base):
    """
    Chandelier Exit: trailing stop basado en ATR desde el máximo/mínimo reciente.
    Señal cuando el precio cruza el nivel Chandelier (flip de tendencia).

    Long CE  = Highest_High(p) - mult * ATR(p)
    Short CE = Lowest_Low(p)  + mult * ATR(p)

    Flip long→short cuando close < Long CE  → señal SELL
    Flip short→long cuando close > Short CE → señal BUY
    """

    def __init__(self, period=22, mult=3.0, ema_trend=100):
        self.period    = period
        self.mult      = mult
        self.ema_trend = ema_trend
        self._min_bars = max(period, ema_trend) + 5
        self.swing_window = period // 3

    def generate_signals_batch(self, data):
        h  = data["high"].values
        l  = data["low"].values
        c  = data["close"].values
        n  = len(c)

        atr14 = _atr(h, l, c, self.period)
        et    = _ema(c, self.ema_trend)

        # Highest High y Lowest Low rodantes
        hh = np.full(n, np.nan)
        ll = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            hh[i] = h[i - self.period + 1:i + 1].max()
            ll[i] = l[i - self.period + 1:i + 1].min()

        long_ce  = hh - self.mult * atr14   # soporte trailing long
        short_ce = ll + self.mult * atr14   # resistencia trailing short

        # Estado de tendencia: 1 = bullish, -1 = bearish
        trend  = 0
        sigs   = ["hold"] * n

        for i in range(self.period, n):
            if np.isnan(long_ce[i]) or np.isnan(short_ce[i]) or np.isnan(et[i]):
                continue

            prev_trend = trend
            if c[i] > short_ce[i]:
                trend = 1
            elif c[i] < long_ce[i]:
                trend = -1

            if prev_trend != 1 and trend == 1 and c[i] > et[i]:
                sigs[i] = "buy"
            elif prev_trend != -1 and trend == -1 and c[i] < et[i]:
                sigs[i] = "sell"

        return sigs

    def __repr__(self):
        return f"ChandelierExit(p={self.period},m={self.mult},et={self.ema_trend})"


# ════════════════════════════════════════════════════════════════════════════
# 2. Turtle Breakout
# ════════════════════════════════════════════════════════════════════════════

class TurtleBreakoutStrategy(_Base):
    """
    Sistema Turtle Trader simplificado:
    - Entry: cierre por encima del máximo de N_entry velas previas (long)
             cierre por debajo del mínimo de N_entry velas previas (short)
    - Exit:  cierre por debajo del mínimo de N_exit velas previas (long)
             cierre por encima del máximo de N_exit velas previas (short)
    - Filtro: EMA confirma dirección de tendencia
    """

    def __init__(self, entry_period=20, exit_period=10, ema_trend=100):
        self.entry_period = entry_period
        self.exit_period  = exit_period
        self.ema_trend    = ema_trend
        self._min_bars    = max(entry_period, ema_trend) + 10
        self.swing_window = entry_period // 3

    def generate_signals_batch(self, data):
        h  = data["high"].values
        l  = data["low"].values
        c  = data["close"].values
        n  = len(c)
        et = _ema(c, self.ema_trend)
        ep = self.entry_period
        xp = self.exit_period

        sigs = ["hold"] * n
        for i in range(ep, n):
            if np.isnan(et[i]):
                continue
            entry_high = h[i - ep:i].max()
            entry_low  = l[i - ep:i].min()
            exit_high  = h[i - xp:i].max()  if i >= xp else h[:i].max()
            exit_low   = l[i - xp:i].min()  if i >= xp else l[:i].min()

            if c[i] > entry_high and c[i] > et[i]:
                sigs[i] = "buy"
            elif c[i] < entry_low and c[i] < et[i]:
                sigs[i] = "sell"
            # exit signals (flip)
            elif c[i] < exit_low:
                sigs[i] = "sell"
            elif c[i] > exit_high:
                sigs[i] = "buy"

        return sigs

    def __repr__(self):
        return f"Turtle(en={self.entry_period},ex={self.exit_period},et={self.ema_trend})"


# ════════════════════════════════════════════════════════════════════════════
# 3. MultiEMA Swing + ADX
# ════════════════════════════════════════════════════════════════════════════

class MultiEMASwingStrategy(_Base):
    """
    3 EMAs lentas + ADX confirma tendencia fuerte:
    - Alineadas bull (e1 > e2 > e3) y precio retrocede a e1 → BUY si ADX > thresh
    - Alineadas bear (e1 < e2 < e3) y precio rebota en e1  → SELL si ADX > thresh
    """

    def __init__(self, e1=21, e2=55, e3=100, adx_thresh=25, adx_period=14):
        self.e1         = e1
        self.e2         = e2
        self.e3         = e3
        self.adx_thresh = adx_thresh
        self.adx_period = adx_period
        self._min_bars  = e3 + adx_period * 2 + 5
        self.swing_window = e1

    def generate_signals_batch(self, data):
        h  = data["high"].values
        l  = data["low"].values
        c  = data["close"].values
        n  = len(c)

        ema1 = _ema(c, self.e1)
        ema2 = _ema(c, self.e2)
        ema3 = _ema(c, self.e3)
        adx, pdi, ndi = _adx(h, l, c, self.adx_period)

        sigs = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan(x) for x in [ema1[i], ema2[i], ema3[i], adx[i]]):
                continue

            strong_trend = adx[i] > self.adx_thresh

            bull_aligned = ema1[i] > ema2[i] > ema3[i]
            bear_aligned = ema1[i] < ema2[i] < ema3[i]

            # pullback to e1 in uptrend
            near_e1 = abs(c[i] - ema1[i]) / (ema1[i] + 1e-10) < 0.005
            prev_above = c[i-1] > ema1[i-1]

            if bull_aligned and strong_trend and pdi[i] > ndi[i]:
                if (near_e1 or (c[i-1] < ema1[i-1] and c[i] > ema1[i])):
                    sigs[i] = "buy"
            elif bear_aligned and strong_trend and ndi[i] > pdi[i]:
                if (near_e1 or (c[i-1] > ema1[i-1] and c[i] < ema1[i])):
                    sigs[i] = "sell"

        return sigs

    def __repr__(self):
        return f"MultiEMASwing(e1={self.e1},e2={self.e2},e3={self.e3},adx>{self.adx_thresh})"
