"""
strategy_zoo.py

10 estrategias adicionales con generación de señales en batch (O(n)).
Cada clase implementa:
  - generate_signals_batch(data) -> list[str]   ← rápida, toda la serie
  - generate_signal(data)        -> dict        ← última vela (compatible router)

Estrategias:
  1.  EMACrossStrategy          — cruce de dos EMAs
  2.  TripleEMAStrategy         — alineación de 3 EMAs + pullback
  3.  RSIDivergenceStrategy     — divergencia RSI vs precio
  4.  StochasticStrategy        — cruce K/D en zonas extremas
  5.  CCIStrategy               — CCI cruza ±100
  6.  DonchianBreakoutStrategy  — ruptura del canal de Donchian
  7.  PinBarStrategy            — mecha larga (pin bar) en extremo
  8.  EngulfingStrategy         — vela envolvente alcista/bajista
  9.  SupertrendStrategy        — supertrend indicator
  10. ADXDIStrategy             — DI+/DI- cruce con filtro ADX
"""

import numpy as np
import pandas as pd


# ── Indicadores compartidos ──────────────────────────────────────────────────

def _ema_v(arr: np.ndarray, period: int) -> np.ndarray:
    alpha  = 2.0 / (period + 1)
    out    = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    out[period - 1] = arr[:period].mean()
    for i in range(period, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _atr_v(high, low, close, period: int) -> np.ndarray:
    n  = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _rsi_v(close, period: int) -> np.ndarray:
    n      = len(close)
    out    = np.full(n, np.nan)
    if n < period + 1:
        return out
    d    = np.diff(close)
    g    = np.where(d > 0, d, 0.0)
    l    = np.where(d < 0, -d, 0.0)
    ag   = g[:period].mean()
    al   = l[:period].mean()
    for i in range(period, n - 1):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + l[i]) / period
        out[i + 1] = 100 - 100 / (1 + ag / (al + 1e-10))
    return out


def _sma_v(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        out[i] = arr[i - period + 1: i + 1].mean()
    return out


def _stoch_v(high, low, close, k_period=14, d_period=3):
    n  = len(close)
    k  = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        hh = high[i - k_period + 1: i + 1].max()
        ll = low[i  - k_period + 1: i + 1].min()
        k[i] = 100 * (close[i] - ll) / (hh - ll + 1e-10)
    d = _sma_v(k, d_period)
    return k, d


def _cci_v(high, low, close, period=20):
    n   = len(close)
    tp  = (high + low + close) / 3.0
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        sl  = tp[i - period + 1: i + 1]
        md  = np.mean(sl)
        mad = np.mean(np.abs(sl - md))
        out[i] = (tp[i] - md) / (0.015 * mad + 1e-10)
    return out


def _donchian_v(high, low, period=20):
    n  = len(high)
    ub = np.full(n, np.nan)
    lb = np.full(n, np.nan)
    for i in range(period - 1, n):
        ub[i] = high[i - period + 1: i + 1].max()
        lb[i] = low[i  - period + 1: i + 1].min()
    return ub, lb


def _wilder_di(high, low, close, period=14):
    """DI+ y DI- usando suavizado Wilder."""
    n   = len(close)
    dm_plus  = np.zeros(n)
    dm_minus = np.zeros(n)
    tr_arr   = np.zeros(n)
    for i in range(1, n):
        up   = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        dm_plus[i]  = up   if up > down and up > 0   else 0
        dm_minus[i] = down if down > up and down > 0 else 0
        tr_arr[i]   = max(high[i] - low[i],
                          abs(high[i] - close[i - 1]),
                          abs(low[i]  - close[i - 1]))
    atr  = np.full(n, np.nan)
    sdp  = np.full(n, np.nan)
    sdm  = np.full(n, np.nan)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    atr[period - 1] = tr_arr[:period].mean()
    sdp[period - 1] = dm_plus[:period].mean()
    sdm[period - 1] = dm_minus[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr_arr[i]) / period
        sdp[i] = (sdp[i-1] * (period-1) + dm_plus[i]) / period
        sdm[i] = (sdm[i-1] * (period-1) + dm_minus[i]) / period
    di_plus  = 100 * sdp / (atr + 1e-10)
    di_minus = 100 * sdm / (atr + 1e-10)
    return di_plus, di_minus


# ── Base ─────────────────────────────────────────────────────────────────────

class _BaseStrategy:
    swing_window      = 5
    require_fvg       = False
    use_choch_filter  = False

    def generate_signal(self, data) -> dict:
        if data is None or len(data) < self._min_bars:
            return {"signal": "hold", "reason": "insufficient_data"}
        sigs = self.generate_signals_batch(data)
        sig  = sigs[-1] if sigs else "hold"
        return {"signal": sig}

    def generate_signals_batch(self, data) -> list:
        raise NotImplementedError


# ════════════════════════════════════════════════════════════════════════════
# 1. EMA Cross
# ════════════════════════════════════════════════════════════════════════════

class EMACrossStrategy(_BaseStrategy):
    """Cruce de EMA rápida sobre EMA lenta con confirmación de tendencia (EMA trend)."""

    def __init__(self, fast=9, slow=21, trend=50, require_trend=True):
        self.fast          = fast
        self.slow          = slow
        self.trend         = trend
        self.require_trend = require_trend
        self._min_bars     = trend + 5
        self.swing_window  = slow

    def generate_signals_batch(self, data) -> list:
        c    = data["close"].values
        ef   = _ema_v(c, self.fast)
        es   = _ema_v(c, self.slow)
        et   = _ema_v(c, self.trend)
        n    = len(c)
        sigs = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan([ef[i], ef[i-1], es[i], es[i-1], et[i]])):
                continue
            cross_up   = ef[i-1] <= es[i-1] and ef[i] > es[i]
            cross_down = ef[i-1] >= es[i-1] and ef[i] < es[i]
            bull_trend = c[i] > et[i]
            bear_trend = c[i] < et[i]
            if cross_up   and (not self.require_trend or bull_trend):
                sigs[i] = "buy"
            elif cross_down and (not self.require_trend or bear_trend):
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"EMACross(fast={self.fast},slow={self.slow},trend={self.trend})"


# ════════════════════════════════════════════════════════════════════════════
# 2. Triple EMA
# ════════════════════════════════════════════════════════════════════════════

class TripleEMAStrategy(_BaseStrategy):
    """3 EMAs alineadas (e1<e2<e3 bull, e1>e2>e3 bear) + pullback a EMA2."""

    def __init__(self, e1=8, e2=21, e3=55, pullback_pct=0.003):
        self.e1           = e1
        self.e2           = e2
        self.e3           = e3
        self.pullback_pct = pullback_pct
        self._min_bars    = e3 + e1 + 5
        self.swing_window = e2

    def generate_signals_batch(self, data) -> list:
        c    = data["close"].values
        e1   = _ema_v(c, self.e1)
        e2   = _ema_v(c, self.e2)
        e3   = _ema_v(c, self.e3)
        n    = len(c)
        sigs = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan([e1[i], e2[i], e3[i]])):
                continue
            bull = e1[i] > e2[i] > e3[i]
            bear = e1[i] < e2[i] < e3[i]
            near_e2_bull = abs(c[i] - e2[i]) / e2[i] < self.pullback_pct and c[i] > e3[i]
            near_e2_bear = abs(c[i] - e2[i]) / e2[i] < self.pullback_pct and c[i] < e3[i]
            if bull and near_e2_bull:
                sigs[i] = "buy"
            elif bear and near_e2_bear:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"TripleEMA(e1={self.e1},e2={self.e2},e3={self.e3})"


# ════════════════════════════════════════════════════════════════════════════
# 3. RSI Divergence
# ════════════════════════════════════════════════════════════════════════════

class RSIDivergenceStrategy(_BaseStrategy):
    """Divergencia precio vs RSI en lookback velas."""

    def __init__(self, rsi_period=14, lookback=8, oversold=35, overbought=65):
        self.rsi_period  = rsi_period
        self.lookback    = lookback
        self.oversold    = oversold
        self.overbought  = overbought
        self._min_bars   = rsi_period + lookback + 5
        self.swing_window = rsi_period

    def generate_signals_batch(self, data) -> list:
        c    = data["close"].values
        rsi  = _rsi_v(c, self.rsi_period)
        n    = len(c)
        lb   = self.lookback
        sigs = ["hold"] * n
        for i in range(lb, n):
            if np.any(np.isnan(rsi[i-lb:i+1])):
                continue
            pw = c[i-lb:i+1]
            rw = rsi[i-lb:i+1]
            # Bullish div: precio mínimo más bajo, RSI mínimo más alto
            if (c[i] == pw.min() and rsi[i] > rw.min() and rsi[i] < self.oversold + 15):
                sigs[i] = "buy"
            # Bearish div: precio máximo más alto, RSI máximo más bajo
            elif (c[i] == pw.max() and rsi[i] < rw.max() and rsi[i] > self.overbought - 15):
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"RSIDiv(period={self.rsi_period},lb={self.lookback})"


# ════════════════════════════════════════════════════════════════════════════
# 4. Stochastic
# ════════════════════════════════════════════════════════════════════════════

class StochasticStrategy(_BaseStrategy):
    """Cruce K sobre D en zona sobrecomprada/sobrevendida."""

    def __init__(self, k_period=14, d_period=3, oversold=25, overbought=75):
        self.k_period   = k_period
        self.d_period   = d_period
        self.oversold   = oversold
        self.overbought = overbought
        self._min_bars  = k_period + d_period + 5
        self.swing_window = k_period

    def generate_signals_batch(self, data) -> list:
        h    = data["high"].values
        l    = data["low"].values
        c    = data["close"].values
        k, d = _stoch_v(h, l, c, self.k_period, self.d_period)
        n    = len(c)
        sigs = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan([k[i], k[i-1], d[i], d[i-1]])):
                continue
            cross_up   = k[i-1] <= d[i-1] and k[i] > d[i]
            cross_down = k[i-1] >= d[i-1] and k[i] < d[i]
            if cross_up   and k[i] < self.oversold:
                sigs[i] = "buy"
            elif cross_down and k[i] > self.overbought:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"Stoch(k={self.k_period},d={self.d_period},os={self.oversold})"


# ════════════════════════════════════════════════════════════════════════════
# 5. CCI
# ════════════════════════════════════════════════════════════════════════════

class CCIStrategy(_BaseStrategy):
    """CCI cruza el nivel ±threshold (entrada) con filtro de tendencia EMA."""

    def __init__(self, period=20, threshold=100, ema_trend=50):
        self.period    = period
        self.threshold = threshold
        self.ema_trend = ema_trend
        self._min_bars = max(period, ema_trend) + 5
        self.swing_window = period

    def generate_signals_batch(self, data) -> list:
        h    = data["high"].values
        l    = data["low"].values
        c    = data["close"].values
        cci  = _cci_v(h, l, c, self.period)
        et   = _ema_v(c, self.ema_trend)
        n    = len(c)
        sigs = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan([cci[i], cci[i-1], et[i]])):
                continue
            cross_above = cci[i-1] <= -self.threshold and cci[i] > -self.threshold
            cross_below = cci[i-1] >= self.threshold  and cci[i] < self.threshold
            if cross_above and c[i] > et[i]:
                sigs[i] = "buy"
            elif cross_below and c[i] < et[i]:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"CCI(period={self.period},thr={self.threshold})"


# ════════════════════════════════════════════════════════════════════════════
# 6. Donchian Breakout
# ════════════════════════════════════════════════════════════════════════════

class DonchianBreakoutStrategy(_BaseStrategy):
    """Precio cierra fuera del canal de Donchian (ruptura)."""

    def __init__(self, period=20, exit_period=10, atr_filter=True, atr_mult=1.0):
        self.period      = period
        self.exit_period = exit_period
        self.atr_filter  = atr_filter
        self.atr_mult    = atr_mult
        self._min_bars   = period + 10
        self.swing_window = period

    def generate_signals_batch(self, data) -> list:
        h    = data["high"].values
        l    = data["low"].values
        c    = data["close"].values
        ub, lb = _donchian_v(h, l, self.period)
        atr  = _atr_v(h, l, c, 14) if self.atr_filter else None
        # Mid channel for exit
        ub_e, lb_e = _donchian_v(h, l, self.exit_period)
        n    = len(c)
        sigs = ["hold"] * n
        pos  = None
        for i in range(self.period, n):
            if any(np.isnan([ub[i], lb[i], c[i]])):
                continue
            atr_ok = True
            if self.atr_filter and atr is not None and not np.isnan(atr[i]):
                # Solo entrar si la ruptura supera ATR * mult respecto al canal
                atr_ok = (c[i] - ub[i-1] > self.atr_mult * atr[i] or
                          lb[i-1] - c[i] > self.atr_mult * atr[i]) if pos is None else True

            if pos is None:
                if c[i] > ub[i-1] and atr_ok:
                    sigs[i] = "buy";  pos = "buy"
                elif c[i] < lb[i-1] and atr_ok:
                    sigs[i] = "sell"; pos = "sell"
            else:
                # Salida: precio vuelve al canal de exit
                if pos == "buy"  and not np.isnan(lb_e[i]) and c[i] < lb_e[i]:
                    sigs[i] = "sell"; pos = None
                elif pos == "sell" and not np.isnan(ub_e[i]) and c[i] > ub_e[i]:
                    sigs[i] = "buy";  pos = None
        return sigs

    def __repr__(self):
        return f"Donchian(period={self.period},exit={self.exit_period})"


# ════════════════════════════════════════════════════════════════════════════
# 7. Pin Bar
# ════════════════════════════════════════════════════════════════════════════

class PinBarStrategy(_BaseStrategy):
    """Pin bar (mecha larga) en zona de sobrecompra/sobreventa."""

    def __init__(self, wick_ratio=2.5, body_pct=0.35, rsi_period=14,
                 rsi_os=40, rsi_ob=60, ema_trend=50):
        self.wick_ratio = wick_ratio
        self.body_pct   = body_pct
        self.rsi_period = rsi_period
        self.rsi_os     = rsi_os
        self.rsi_ob     = rsi_ob
        self.ema_trend  = ema_trend
        self._min_bars  = max(rsi_period, ema_trend) + 5
        self.swing_window = ema_trend

    def generate_signals_batch(self, data) -> list:
        o    = data["open"].values
        h    = data["high"].values
        l    = data["low"].values
        c    = data["close"].values
        rsi  = _rsi_v(c, self.rsi_period)
        et   = _ema_v(c, self.ema_trend)
        n    = len(c)
        sigs = ["hold"] * n
        for i in range(self._min_bars, n):
            if np.isnan(rsi[i]) or np.isnan(et[i]):
                continue
            body       = abs(c[i] - o[i])
            candle_rng = h[i] - l[i]
            if candle_rng < 1e-10:
                continue
            body_ratio = body / candle_rng
            lower_wick = min(o[i], c[i]) - l[i]
            upper_wick = h[i] - max(o[i], c[i])
            # Bullish pin: mecha inferior larga, cuerpo pequeño, por debajo EMA
            bull_pin = (lower_wick >= self.wick_ratio * body and
                        body_ratio < self.body_pct and
                        c[i] < et[i] and rsi[i] < self.rsi_os + 15)
            # Bearish pin: mecha superior larga
            bear_pin = (upper_wick >= self.wick_ratio * body and
                        body_ratio < self.body_pct and
                        c[i] > et[i] and rsi[i] > self.rsi_ob - 15)
            if bull_pin:
                sigs[i] = "buy"
            elif bear_pin:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"PinBar(wick={self.wick_ratio},body={self.body_pct})"


# ════════════════════════════════════════════════════════════════════════════
# 8. Engulfing
# ════════════════════════════════════════════════════════════════════════════

class EngulfingStrategy(_BaseStrategy):
    """Vela envolvente alcista/bajista con filtro de tendencia."""

    def __init__(self, ema_trend=50, body_mult=1.2, rsi_period=14,
                 rsi_os=45, rsi_ob=55):
        self.ema_trend  = ema_trend
        self.body_mult  = body_mult
        self.rsi_period = rsi_period
        self.rsi_os     = rsi_os
        self.rsi_ob     = rsi_ob
        self._min_bars  = max(ema_trend, rsi_period) + 5
        self.swing_window = ema_trend

    def generate_signals_batch(self, data) -> list:
        o    = data["open"].values
        c    = data["close"].values
        rsi  = _rsi_v(c, self.rsi_period)
        et   = _ema_v(c, self.ema_trend)
        n    = len(c)
        sigs = ["hold"] * n
        for i in range(1, n):
            if np.isnan(et[i]) or np.isnan(rsi[i]):
                continue
            prev_bull = c[i-1] > o[i-1]
            prev_bear = c[i-1] < o[i-1]
            prev_body = abs(c[i-1] - o[i-1])
            curr_body = abs(c[i]   - o[i])
            # Bullish engulfing: prev bajista, actual alcista y envuelve
            bull_eng = (prev_bear and c[i] > o[i] and
                        c[i] > o[i-1] and o[i] < c[i-1] and
                        curr_body > self.body_mult * prev_body)
            # Bearish engulfing
            bear_eng = (prev_bull and c[i] < o[i] and
                        c[i] < o[i-1] and o[i] > c[i-1] and
                        curr_body > self.body_mult * prev_body)
            if bull_eng and rsi[i] < self.rsi_ob:
                sigs[i] = "buy"
            elif bear_eng and rsi[i] > self.rsi_os:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"Engulfing(ema={self.ema_trend},mult={self.body_mult})"


# ════════════════════════════════════════════════════════════════════════════
# 9. Supertrend
# ════════════════════════════════════════════════════════════════════════════

class SupertrendStrategy(_BaseStrategy):
    """Supertrend: flip de banda sobre/bajo precio."""

    def __init__(self, atr_period=10, multiplier=3.0):
        self.atr_period  = atr_period
        self.multiplier  = multiplier
        self._min_bars   = atr_period + 5
        self.swing_window = atr_period

    def _supertrend(self, high, low, close):
        n    = len(close)
        atr  = _atr_v(high, low, close, self.atr_period)
        hl2  = (high + low) / 2.0
        ub   = hl2 + self.multiplier * atr   # upper band
        lb   = hl2 - self.multiplier * atr   # lower band
        final_ub = np.copy(ub)
        final_lb = np.copy(lb)
        trend    = np.ones(n, dtype=int)  # 1=bull, -1=bear

        for i in range(1, n):
            if np.isnan(atr[i]):
                continue
            final_ub[i] = ub[i] if (ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]) else final_ub[i-1]
            final_lb[i] = lb[i] if (lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]) else final_lb[i-1]
            if trend[i-1] == -1 and close[i] > final_ub[i]:
                trend[i] = 1
            elif trend[i-1] == 1 and close[i] < final_lb[i]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
        return trend

    def generate_signals_batch(self, data) -> list:
        h    = data["high"].values
        l    = data["low"].values
        c    = data["close"].values
        trend = self._supertrend(h, l, c)
        n    = len(c)
        sigs = ["hold"] * n
        for i in range(1, n):
            if trend[i] == 1 and trend[i-1] == -1:
                sigs[i] = "buy"
            elif trend[i] == -1 and trend[i-1] == 1:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"Supertrend(atr={self.atr_period},mult={self.multiplier})"


# ════════════════════════════════════════════════════════════════════════════
# 10. ADX + DI Cross
# ════════════════════════════════════════════════════════════════════════════

class ADXDIStrategy(_BaseStrategy):
    """DI+ cruza sobre DI- con ADX por encima del umbral (tendencia activa)."""

    def __init__(self, period=14, adx_threshold=20, adx_strong=30):
        self.period       = period
        self.adx_threshold = adx_threshold
        self.adx_strong   = adx_strong
        self._min_bars    = period * 2 + 5
        self.swing_window = period

    def generate_signals_batch(self, data) -> list:
        h    = data["high"].values
        l    = data["low"].values
        c    = data["close"].values
        dip, dim = _wilder_di(h, l, c, self.period)
        dx   = 100 * np.abs(dip - dim) / (dip + dim + 1e-10)
        adx  = _ema_v(dx, self.period)   # aproximación rápida
        n    = len(c)
        sigs = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan([dip[i], dim[i], dip[i-1], dim[i-1], adx[i]])):
                continue
            if adx[i] < self.adx_threshold:
                continue
            cross_up   = dip[i-1] <= dim[i-1] and dip[i] > dim[i]
            cross_down = dip[i-1] >= dim[i-1] and dip[i] < dim[i]
            if cross_up:
                sigs[i] = "buy"
            elif cross_down:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"ADXDI(period={self.period},adx_thr={self.adx_threshold})"


# ── Registro de todas las estrategias ────────────────────────────────────────

ALL_STRATEGY_CLASSES = {
    "ema_cross":         EMACrossStrategy,
    "triple_ema":        TripleEMAStrategy,
    "rsi_divergence":    RSIDivergenceStrategy,
    "stochastic":        StochasticStrategy,
    "cci":               CCIStrategy,
    "donchian":          DonchianBreakoutStrategy,
    "pin_bar":           PinBarStrategy,
    "engulfing":         EngulfingStrategy,
    "supertrend":        SupertrendStrategy,
    "adx_di":            ADXDIStrategy,
}
