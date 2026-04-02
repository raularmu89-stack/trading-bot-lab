"""
scalp_strategies.py

10 estrategias de scalping para timeframes cortos (15min, 5min).
Alta frecuencia de señales, períodos cortos, entradas rápidas.

  1.  EMAScalpStrategy       — cruce 3/8 EMA ultrarrápido
  2.  RSIScalpStrategy       — RSI con bandas ajustadas (40/60) + EMA corta
  3.  StochScalpStrategy     — Stochastic 5/3 con zonas extremas 10/90
  4.  BBScalpStrategy        — Rebote de Bollinger con std=1.5 y período corto
  5.  CCIScalpStrategy       — CCI ±50 con confirmación de vela
  6.  MACDZeroStrategy       — MACD cruza línea cero
  7.  VolumeBreakStrategy    — Spike de volumen + vela direccional grande
  8.  PriceActionScalpStrategy — Inside bar + outside bar breakout
  9.  MomentumScalpStrategy  — N velas consecutivas en la misma dirección
  10. DualThrustStrategy     — Rango diario para niveles de entrada (adaptado)
"""

import numpy as np
import pandas as pd


# ── Indicadores ──────────────────────────────────────────────────────────────

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
        out[i] = arr[i - p + 1:i + 1].mean()
    return out


def _std(arr, p):
    out = np.full(len(arr), np.nan)
    for i in range(p - 1, len(arr)):
        out[i] = arr[i - p + 1:i + 1].std(ddof=1)
    return out


def _atr(h, l, c, p):
    n = len(c)
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


def _rsi(c, p=14):
    n = len(c)
    out = np.full(n, np.nan)
    if n < p + 1:
        return out
    d = np.diff(c)
    g = np.where(d > 0, d, 0.)
    ll = np.where(d < 0, -d, 0.)
    ag, al = g[:p].mean(), ll[:p].mean()
    for i in range(p, n - 1):
        ag = (ag * (p-1) + g[i]) / p
        al = (al * (p-1) + ll[i]) / p
        out[i+1] = 100 - 100 / (1 + ag / (al + 1e-10))
    return out


def _stoch(h, l, c, k=5, d=3):
    n = len(c)
    K = np.full(n, np.nan)
    for i in range(k - 1, n):
        hh = h[i - k + 1:i + 1].max()
        ll = l[i - k + 1:i + 1].min()
        K[i] = 100 * (c[i] - ll) / (hh - ll + 1e-10)
    D = _sma(K, d)
    return K, D


# ── Base ─────────────────────────────────────────────────────────────────────

class _Base:
    swing_window = 3
    require_fvg = False
    use_choch_filter = False
    _min_bars = 30

    def generate_signal(self, data):
        if data is None or len(data) < self._min_bars:
            return {"signal": "hold", "reason": "insufficient_data"}
        s = self.generate_signals_batch(data)
        return {"signal": s[-1] if s else "hold"}

    def generate_signals_batch(self, data):
        raise NotImplementedError


# ════════════════════════════════════════════════════════════════════════════
# 1. EMA Scalp — cruce ultrarrápido 3/8 con filtro 21
# ════════════════════════════════════════════════════════════════════════════

class EMAScalpStrategy(_Base):
    """Cruce EMA rápida/lenta con filtro de tendencia EMA media."""

    def __init__(self, fast=3, slow=8, trend=21, require_trend=True):
        self.fast = fast; self.slow = slow; self.trend = trend
        self.require_trend = require_trend
        self._min_bars = trend + 5
        self.swing_window = slow

    def generate_signals_batch(self, data):
        c  = data["close"].values
        ef = _ema(c, self.fast)
        es = _ema(c, self.slow)
        et = _ema(c, self.trend)
        n  = len(c)
        sg = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan([ef[i], ef[i-1], es[i], es[i-1], et[i]])):
                continue
            cup  = ef[i-1] <= es[i-1] and ef[i] > es[i]
            cdn  = ef[i-1] >= es[i-1] and ef[i] < es[i]
            bull = c[i] > et[i]
            bear = c[i] < et[i]
            if cup  and (not self.require_trend or bull): sg[i] = "buy"
            elif cdn and (not self.require_trend or bear): sg[i] = "sell"
        return sg

    def __repr__(self):
        return f"EMAScalp(f{self.fast},s{self.slow},t{self.trend})"


# ════════════════════════════════════════════════════════════════════════════
# 2. RSI Scalp — zonas ajustadas 40/60 con período corto
# ════════════════════════════════════════════════════════════════════════════

class RSIScalpStrategy(_Base):
    """RSI con bandas ajustadas para scalping. Más señales que el RSI clásico."""

    def __init__(self, rsi_period=7, os=42, ob=58, ema_fast=8, ema_slow=21):
        self.rsi_period = rsi_period
        self.os = os; self.ob = ob
        self.ema_fast = ema_fast; self.ema_slow = ema_slow
        self._min_bars = max(rsi_period, ema_slow) + 5
        self.swing_window = ema_slow

    def generate_signals_batch(self, data):
        c   = data["close"].values
        r   = _rsi(c, self.rsi_period)
        ef  = _ema(c, self.ema_fast)
        es  = _ema(c, self.ema_slow)
        n   = len(c)
        sg  = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan([r[i], r[i-1], ef[i], es[i]])):
                continue
            cross_up  = r[i-1] <= self.os and r[i] > self.os
            cross_dn  = r[i-1] >= self.ob and r[i] < self.ob
            bull = ef[i] > es[i]
            bear = ef[i] < es[i]
            if cross_up and bull: sg[i] = "buy"
            elif cross_dn and bear: sg[i] = "sell"
        return sg

    def __repr__(self):
        return f"RSIScalp(p{self.rsi_period},os{self.os},ob{self.ob})"


# ════════════════════════════════════════════════════════════════════════════
# 3. Stochastic Scalp — K=5, D=3, zonas 10/90
# ════════════════════════════════════════════════════════════════════════════

class StochScalpStrategy(_Base):
    """Stochastic ultra-rápido con cruces en zonas extremas ajustadas."""

    def __init__(self, k=5, d=3, os=15, ob=85, ema_trend=21):
        self.k = k; self.d = d
        self.os = os; self.ob = ob
        self.ema_trend = ema_trend
        self._min_bars = max(k + d, ema_trend) + 5
        self.swing_window = k

    def generate_signals_batch(self, data):
        h  = data["high"].values
        l  = data["low"].values
        c  = data["close"].values
        K, D = _stoch(h, l, c, self.k, self.d)
        et = _ema(c, self.ema_trend)
        n  = len(c)
        sg = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan([K[i], K[i-1], D[i], D[i-1], et[i]])):
                continue
            cup = K[i-1] <= D[i-1] and K[i] > D[i] and K[i] < self.ob
            cdn = K[i-1] >= D[i-1] and K[i] < D[i] and K[i] > self.os
            if cup and c[i] > et[i]: sg[i] = "buy"
            elif cdn and c[i] < et[i]: sg[i] = "sell"
        return sg

    def __repr__(self):
        return f"StochScalp(k{self.k},d{self.d},os{self.os})"


# ════════════════════════════════════════════════════════════════════════════
# 4. Bollinger Scalp — std=1.5, período=10, rebote rápido
# ════════════════════════════════════════════════════════════════════════════

class BBScalpStrategy(_Base):
    """Bollinger estrecho (std=1.5) para capturar rebotes frecuentes."""

    def __init__(self, period=10, std=1.5, rsi_period=7, os=35, ob=65):
        self.period = period; self.std = std
        self.rsi_period = rsi_period; self.os = os; self.ob = ob
        self._min_bars = max(period, rsi_period) + 5
        self.swing_window = period

    def generate_signals_batch(self, data):
        c   = data["close"].values
        h   = data["high"].values
        l   = data["low"].values
        mid = _sma(c, self.period)
        sd  = _std(c, self.period)
        ub  = mid + self.std * sd
        lb  = mid - self.std * sd
        r   = _rsi(c, self.rsi_period)
        n   = len(c)
        sg  = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan([ub[i], lb[i], mid[i], r[i]])):
                continue
            touch_lb = l[i] <= lb[i] and c[i] > lb[i]
            touch_ub = h[i] >= ub[i] and c[i] < ub[i]
            if touch_lb and r[i] < self.ob: sg[i] = "buy"
            elif touch_ub and r[i] > self.os: sg[i] = "sell"
        return sg

    def __repr__(self):
        return f"BBScalp(p{self.period},std{self.std},rsi{self.rsi_period})"


# ════════════════════════════════════════════════════════════════════════════
# 5. CCI Scalp — ±50 con confirmación rápida
# ════════════════════════════════════════════════════════════════════════════

class CCIScalpStrategy(_Base):
    """CCI con umbral ±50 (más señales que ±100)."""

    def __init__(self, period=10, thr=50, ema_trend=21):
        self.period = period; self.thr = thr; self.ema_trend = ema_trend
        self._min_bars = max(period, ema_trend) + 5
        self.swing_window = period

    def _cci(self, h, l, c):
        tp  = (h + l + c) / 3
        out = np.full(len(c), np.nan)
        for i in range(self.period - 1, len(c)):
            sl  = tp[i - self.period + 1:i + 1]
            md  = sl.mean()
            mad = np.mean(np.abs(sl - md))
            out[i] = (tp[i] - md) / (0.015 * mad + 1e-10)
        return out

    def generate_signals_batch(self, data):
        h  = data["high"].values
        l  = data["low"].values
        c  = data["close"].values
        cci = self._cci(h, l, c)
        et  = _ema(c, self.ema_trend)
        n   = len(c)
        sg  = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan([cci[i], cci[i-1], et[i]])):
                continue
            cup = cci[i-1] <= -self.thr and cci[i] > -self.thr
            cdn = cci[i-1] >= self.thr  and cci[i] < self.thr
            if cup and c[i] > et[i]: sg[i] = "buy"
            elif cdn and c[i] < et[i]: sg[i] = "sell"
        return sg

    def __repr__(self):
        return f"CCIScalp(p{self.period},thr{self.thr})"


# ════════════════════════════════════════════════════════════════════════════
# 6. MACD Zero Cross — MACD cruza la línea cero
# ════════════════════════════════════════════════════════════════════════════

class MACDZeroStrategy(_Base):
    """MACD cruza el cero: captura cambios de momentum temprano."""

    def __init__(self, fast=5, slow=13, signal=3, ema_trend=21):
        self.fast = fast; self.slow = slow
        self.signal = signal; self.ema_trend = ema_trend
        self._min_bars = slow + signal + ema_trend + 5
        self.swing_window = slow

    def generate_signals_batch(self, data):
        c    = data["close"].values
        ef   = _ema(c, self.fast)
        es   = _ema(c, self.slow)
        macd = ef - es
        sig  = _ema(np.where(np.isnan(macd), 0, macd), self.signal)
        et   = _ema(c, self.ema_trend)
        n    = len(c)
        sg   = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan([macd[i], macd[i-1], sig[i], et[i]])):
                continue
            # Cruce sobre cero del histograma (MACD - signal)
            hist_prev = macd[i-1] - sig[i-1]
            hist_curr = macd[i]   - sig[i]
            cup = hist_prev <= 0 and hist_curr > 0
            cdn = hist_prev >= 0 and hist_curr < 0
            bull = c[i] > et[i]
            bear = c[i] < et[i]
            if cup and bull: sg[i] = "buy"
            elif cdn and bear: sg[i] = "sell"
        return sg

    def __repr__(self):
        return f"MACDZero(f{self.fast},s{self.slow},t{self.ema_trend})"


# ════════════════════════════════════════════════════════════════════════════
# 7. Volume Break — spike de volumen + vela con cuerpo grande
# ════════════════════════════════════════════════════════════════════════════

class VolumeBreakStrategy(_Base):
    """Volumen > N×media y vela con cuerpo > M×ATR → entrada direccional."""

    def __init__(self, vol_period=20, vol_mult=1.8, atr_period=10,
                 body_mult=0.6, ema_trend=21):
        self.vol_period = vol_period
        self.vol_mult   = vol_mult
        self.atr_period = atr_period
        self.body_mult  = body_mult
        self.ema_trend  = ema_trend
        self._min_bars  = max(vol_period, atr_period, ema_trend) + 5
        self.swing_window = vol_period // 4

    def generate_signals_batch(self, data):
        o   = data["open"].values
        h   = data["high"].values
        l   = data["low"].values
        c   = data["close"].values
        v   = data["volume"].values if "volume" in data.columns else np.ones(len(c))
        at  = _atr(h, l, c, self.atr_period)
        et  = _ema(c, self.ema_trend)
        vm  = _sma(v, self.vol_period)
        n   = len(c)
        sg  = ["hold"] * n
        for i in range(self._min_bars, n):
            if any(np.isnan([at[i], et[i], vm[i]])):
                continue
            vol_spike  = v[i] > self.vol_mult * vm[i]
            body       = abs(c[i] - o[i])
            big_candle = body > self.body_mult * at[i]
            bull_candle = c[i] > o[i]
            bear_candle = c[i] < o[i]
            if vol_spike and big_candle and bull_candle and c[i] > et[i]:
                sg[i] = "buy"
            elif vol_spike and big_candle and bear_candle and c[i] < et[i]:
                sg[i] = "sell"
        return sg

    def __repr__(self):
        return f"VolBreak(vm{self.vol_mult},bm{self.body_mult})"


# ════════════════════════════════════════════════════════════════════════════
# 8. Price Action Scalp — Inside/Outside bar breakout
# ════════════════════════════════════════════════════════════════════════════

class PriceActionScalpStrategy(_Base):
    """
    Inside bar: vela contenida dentro de la anterior.
    Señal cuando la siguiente vela rompe el rango del inside bar.
    """
    def __init__(self, ema_trend=21, atr_period=10, min_range_atr=0.3):
        self.ema_trend    = ema_trend
        self.atr_period   = atr_period
        self.min_range_atr = min_range_atr
        self._min_bars    = max(ema_trend, atr_period) + 5
        self.swing_window = ema_trend // 3

    def generate_signals_batch(self, data):
        h   = data["high"].values
        l   = data["low"].values
        c   = data["close"].values
        et  = _ema(c, self.ema_trend)
        at  = _atr(h, l, c, self.atr_period)
        n   = len(c)
        sg  = ["hold"] * n
        inside_high = None
        inside_low  = None
        for i in range(2, n):
            if any(np.isnan([et[i], at[i]])):
                continue
            # Detectar inside bar en i-1
            is_inside = h[i-1] <= h[i-2] and l[i-1] >= l[i-2]
            if is_inside and at[i-1] > 0:
                rng = h[i-2] - l[i-2]
                if rng >= self.min_range_atr * at[i-1]:
                    inside_high = h[i-2]
                    inside_low  = l[i-2]
            # Romper el rango del inside bar
            if inside_high is not None:
                if c[i] > inside_high and c[i] > et[i]:
                    sg[i] = "buy"
                    inside_high = inside_low = None
                elif c[i] < inside_low and c[i] < et[i]:
                    sg[i] = "sell"
                    inside_high = inside_low = None
                elif h[i] > h[i-2]:  # superado rango materno, reset
                    inside_high = inside_low = None
        return sg

    def __repr__(self):
        return f"PAScalp(et{self.ema_trend},mr{self.min_range_atr})"


# ════════════════════════════════════════════════════════════════════════════
# 9. Momentum Scalp — N velas consecutivas misma dirección
# ════════════════════════════════════════════════════════════════════════════

class MomentumScalpStrategy(_Base):
    """
    N velas consecutivas alcistas/bajistas indican momentum fuerte.
    Entra en la dirección si el volumen también acompaña.
    """
    def __init__(self, streak=3, vol_confirm=True, vol_mult=1.2,
                 ema_trend=21, atr_period=10, min_body_atr=0.2):
        self.streak       = streak
        self.vol_confirm  = vol_confirm
        self.vol_mult     = vol_mult
        self.ema_trend    = ema_trend
        self.atr_period   = atr_period
        self.min_body_atr = min_body_atr
        self._min_bars    = max(ema_trend, atr_period) + streak + 5
        self.swing_window = streak

    def generate_signals_batch(self, data):
        o   = data["open"].values
        h   = data["high"].values
        l   = data["low"].values
        c   = data["close"].values
        v   = data["volume"].values if "volume" in data.columns else np.ones(len(c))
        et  = _ema(c, self.ema_trend)
        at  = _atr(h, l, c, self.atr_period)
        vm  = _sma(v, 20)
        n   = len(c)
        s   = self.streak
        sg  = ["hold"] * n
        for i in range(s, n):
            if np.isnan(et[i]) or np.isnan(at[i]):
                continue
            w_bull = all(c[i-j] > o[i-j] for j in range(s))
            w_bear = all(c[i-j] < o[i-j] for j in range(s))
            # Cuerpos mínimos
            bodies_ok = all(abs(c[i-j] - o[i-j]) > self.min_body_atr * at[i]
                            for j in range(s) if not np.isnan(at[i]))
            vol_ok = True
            if self.vol_confirm and not np.isnan(vm[i]):
                vol_ok = v[i] > self.vol_mult * vm[i]
            if w_bull and bodies_ok and vol_ok and c[i] > et[i]:
                sg[i] = "buy"
            elif w_bear and bodies_ok and vol_ok and c[i] < et[i]:
                sg[i] = "sell"
        return sg

    def __repr__(self):
        return f"MomScalp(streak{self.streak},vol{self.vol_mult})"


# ════════════════════════════════════════════════════════════════════════════
# 10. Dual Thrust Scalp — rango N velas para niveles dinámicos
# ════════════════════════════════════════════════════════════════════════════

class DualThrustStrategy(_Base):
    """
    Dual Thrust adaptado: calcula un rango de N velas y usa multiplicador
    para determinar niveles de ruptura. Muy popular en scalping.

    Upper = open + k * range
    Lower = open - k * range
    range = max(HH-LC, HC-LL)
    """
    def __init__(self, n=20, k=0.5, ema_trend=21, atr_filter=True):
        self.n          = n
        self.k          = k
        self.ema_trend  = ema_trend
        self.atr_filter = atr_filter
        self._min_bars  = max(n, ema_trend) + 5
        self.swing_window = n // 4

    def generate_signals_batch(self, data):
        h   = data["high"].values
        l   = data["low"].values
        c   = data["close"].values
        o   = data["open"].values
        et  = _ema(c, self.ema_trend)
        at  = _atr(h, l, c, 14) if self.atr_filter else None
        n_  = len(c)
        sg  = ["hold"] * n_
        for i in range(self.n, n_):
            if np.isnan(et[i]):
                continue
            # Rango de las últimas N velas
            HH = h[i - self.n:i].max()
            LC = l[i - self.n:i].min()
            HC = h[i - self.n:i].min()  # lowest high
            LL = l[i - self.n:i].max()  # highest low
            rng = max(HH - LC, HC - LL) if HH - LC > 0 else HH - LC
            if rng <= 0:
                continue
            upper = o[i] + self.k * rng
            lower = o[i] - self.k * rng
            # Filtro ATR opcional: solo entrar si ruptura > 0.3×ATR
            if self.atr_filter and at is not None and not np.isnan(at[i]):
                if abs(c[i] - upper) > at[i] and abs(c[i] - lower) > at[i]:
                    continue
            if c[i] > upper and c[i] > et[i]:
                sg[i] = "buy"
            elif c[i] < lower and c[i] < et[i]:
                sg[i] = "sell"
        return sg

    def __repr__(self):
        return f"DualThrust(n{self.n},k{self.k})"


# ── Registro ──────────────────────────────────────────────────────────────────

ALL_SCALP_STRATEGIES = {
    "ema_scalp":        EMAScalpStrategy,
    "rsi_scalp":        RSIScalpStrategy,
    "stoch_scalp":      StochScalpStrategy,
    "bb_scalp":         BBScalpStrategy,
    "cci_scalp":        CCIScalpStrategy,
    "macd_zero":        MACDZeroStrategy,
    "volume_break":     VolumeBreakStrategy,
    "pa_scalp":         PriceActionScalpStrategy,
    "momentum_scalp":   MomentumScalpStrategy,
    "dual_thrust":      DualThrustStrategy,
}
