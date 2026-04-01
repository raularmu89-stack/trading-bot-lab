"""
strategy_zoo2.py

10 estrategias nuevas para el torneo de la ronda 2.
Todas con generate_signals_batch(data) -> list para O(n).

  1.  VWAPStrategy           — rebote/cruce sobre VWAP diario
  2.  ParabolicSARStrategy   — flip del SAR parabólico
  3.  WilliamsRStrategy      — Williams %R en zonas extremas
  4.  KeltnerBreakoutStrategy— ruptura de canal de Keltner
  5.  MACDHistStrategy       — cambio de pendiente del histograma MACD
  6.  BollingerBandStrategy  — precio toca banda + rebote confirmado
  7.  ROCStrategy            — Rate of Change con filtro de tendencia
  8.  MFIStrategy            — Money Flow Index (RSI ponderado por volumen)
  9.  HeikinAshiStrategy     — señales sobre velas Heikin-Ashi
  10. ZScoreStrategy         — precio fuera de Z bandas (mean reversion)
"""

import numpy as np
import pandas as pd


# ── Indicadores compartidos ──────────────────────────────────────────────────

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


def _std(arr, p):
    out = np.full(len(arr), np.nan)
    for i in range(p - 1, len(arr)):
        out[i] = arr[i - p + 1:i + 1].std(ddof=1)
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


# ── Base ─────────────────────────────────────────────────────────────────────

class _Base:
    swing_window = 5
    require_fvg = False
    use_choch_filter = False
    _min_bars = 60

    def generate_signal(self, data):
        if data is None or len(data) < self._min_bars:
            return {"signal": "hold", "reason": "insufficient_data"}
        s = self.generate_signals_batch(data)
        return {"signal": s[-1] if s else "hold"}

    def generate_signals_batch(self, data):
        raise NotImplementedError


# ════════════════════════════════════════════════════════════════════════════
# 1. VWAP Bounce
# ════════════════════════════════════════════════════════════════════════════

class VWAPStrategy(_Base):
    """
    VWAP rodante (ventana fija): precio cruza/rebota desde VWAP con
    confirmación de RSI.
    """
    def __init__(self, vwap_window=96, rsi_period=14, rsi_os=45, rsi_ob=55,
                 bounce_pct=0.002):
        self.vwap_window = vwap_window
        self.rsi_period  = rsi_period
        self.rsi_os      = rsi_os
        self.rsi_ob      = rsi_ob
        self.bounce_pct  = bounce_pct
        self._min_bars   = vwap_window + rsi_period + 5
        self.swing_window = vwap_window // 10

    def generate_signals_batch(self, data):
        c   = data["close"].values
        h   = data["high"].values
        l   = data["low"].values
        v   = data["volume"].values if "volume" in data.columns else np.ones(len(c))
        tp  = (h + l + c) / 3.0
        n   = len(c)
        w   = self.vwap_window
        vwap = np.full(n, np.nan)
        for i in range(w - 1, n):
            vs = v[i - w + 1:i + 1]
            ts = tp[i - w + 1:i + 1]
            vwap[i] = (ts * vs).sum() / (vs.sum() + 1e-10)
        rsi = _rsi(c, self.rsi_period)
        sigs = ["hold"] * n
        for i in range(1, n):
            if np.isnan(vwap[i]) or np.isnan(rsi[i]):
                continue
            near_vwap = abs(c[i] - vwap[i]) / (vwap[i] + 1e-10) < self.bounce_pct
            cross_up   = c[i-1] < vwap[i-1] and c[i] > vwap[i]
            cross_down = c[i-1] > vwap[i-1] and c[i] < vwap[i]
            if (near_vwap or cross_up)   and rsi[i] < self.rsi_ob and c[i] > vwap[i]:
                sigs[i] = "buy"
            elif (near_vwap or cross_down) and rsi[i] > self.rsi_os and c[i] < vwap[i]:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"VWAP(w={self.vwap_window},bounce={self.bounce_pct})"


# ════════════════════════════════════════════════════════════════════════════
# 2. Parabolic SAR
# ════════════════════════════════════════════════════════════════════════════

class ParabolicSARStrategy(_Base):
    """Flip del SAR parabólico, filtrado por EMA de tendencia."""

    def __init__(self, af_start=0.02, af_step=0.02, af_max=0.2, ema_trend=50):
        self.af_start  = af_start
        self.af_step   = af_step
        self.af_max    = af_max
        self.ema_trend = ema_trend
        self._min_bars = ema_trend + 10
        self.swing_window = ema_trend // 5

    def _sar(self, h, l):
        n   = len(h)
        sar = np.full(n, np.nan)
        trend = 1   # 1 alcista, -1 bajista
        ep    = h[0]
        af    = self.af_start
        sar[0] = l[0]
        for i in range(1, n):
            if trend == 1:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                sar[i] = min(sar[i], l[i-1], l[max(0,i-2)])
                if l[i] < sar[i]:
                    trend = -1; sar[i] = ep; ep = l[i]; af = self.af_start
                else:
                    if h[i] > ep:
                        ep = h[i]
                        af = min(af + self.af_step, self.af_max)
            else:
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                sar[i] = max(sar[i], h[i-1], h[max(0,i-2)])
                if h[i] > sar[i]:
                    trend = 1; sar[i] = ep; ep = h[i]; af = self.af_start
                else:
                    if l[i] < ep:
                        ep = l[i]
                        af = min(af + self.af_step, self.af_max)
        return sar

    def generate_signals_batch(self, data):
        h   = data["high"].values
        l   = data["low"].values
        c   = data["close"].values
        sar = self._sar(h, l)
        et  = _ema(c, self.ema_trend)
        n   = len(c)
        sigs = ["hold"] * n
        for i in range(1, n):
            if np.isnan(sar[i]) or np.isnan(sar[i-1]) or np.isnan(et[i]):
                continue
            flip_bull = sar[i-1] > c[i-1] and sar[i] < c[i]
            flip_bear = sar[i-1] < c[i-1] and sar[i] > c[i]
            if flip_bull and c[i] > et[i]:
                sigs[i] = "buy"
            elif flip_bear and c[i] < et[i]:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"SAR(af={self.af_start}/{self.af_max},et={self.ema_trend})"


# ════════════════════════════════════════════════════════════════════════════
# 3. Williams %R
# ════════════════════════════════════════════════════════════════════════════

class WilliamsRStrategy(_Base):
    """Williams %R cruza desde zona extrema + confirmación EMA."""

    def __init__(self, period=14, oversold=-80, overbought=-20, ema_trend=50):
        self.period     = period
        self.oversold   = oversold
        self.overbought = overbought
        self.ema_trend  = ema_trend
        self._min_bars  = max(period, ema_trend) + 5
        self.swing_window = period

    def generate_signals_batch(self, data):
        h   = data["high"].values
        l   = data["low"].values
        c   = data["close"].values
        n   = len(c)
        wr  = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            hh = h[i - self.period + 1:i + 1].max()
            ll = l[i - self.period + 1:i + 1].min()
            wr[i] = -100 * (hh - c[i]) / (hh - ll + 1e-10)
        et   = _ema(c, self.ema_trend)
        sigs = ["hold"] * n
        for i in range(1, n):
            if np.isnan(wr[i]) or np.isnan(wr[i-1]) or np.isnan(et[i]):
                continue
            cross_up   = wr[i-1] <= self.oversold   and wr[i] > self.oversold
            cross_down = wr[i-1] >= self.overbought  and wr[i] < self.overbought
            if cross_up   and c[i] > et[i]:
                sigs[i] = "buy"
            elif cross_down and c[i] < et[i]:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"WilliamsR(p={self.period},os={self.oversold})"


# ════════════════════════════════════════════════════════════════════════════
# 4. Keltner Channel Breakout
# ════════════════════════════════════════════════════════════════════════════

class KeltnerBreakoutStrategy(_Base):
    """Precio cierra fuera del canal de Keltner → tendencia iniciada."""

    def __init__(self, ema_period=20, atr_period=14, mult=2.0, confirm_bars=1):
        self.ema_period   = ema_period
        self.atr_period   = atr_period
        self.mult         = mult
        self.confirm_bars = confirm_bars
        self._min_bars    = max(ema_period, atr_period) + confirm_bars + 5
        self.swing_window = ema_period

    def generate_signals_batch(self, data):
        h   = data["high"].values
        l   = data["low"].values
        c   = data["close"].values
        mid = _ema(c, self.ema_period)
        a   = _atr(h, l, c, self.atr_period)
        ub  = mid + self.mult * a
        lb  = mid - self.mult * a
        n   = len(c)
        sigs = ["hold"] * n
        cb   = self.confirm_bars
        for i in range(cb, n):
            if np.isnan(ub[i]) or np.isnan(lb[i]):
                continue
            # Confirmar cb velas consecutivas fuera
            above = all(c[i - j] > ub[i - j] for j in range(cb)
                        if not np.isnan(ub[i - j]))
            below = all(c[i - j] < lb[i - j] for j in range(cb)
                        if not np.isnan(lb[i - j]))
            # Solo señal en la primera vela del breakout
            was_inside = not (c[i - cb] > ub[i - cb]) if above else (not (c[i - cb] < lb[i - cb]))
            if above:
                sigs[i] = "buy"
            elif below:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"Keltner(ema={self.ema_period},mult={self.mult},cb={self.confirm_bars})"


# ════════════════════════════════════════════════════════════════════════════
# 5. MACD Histogram Slope
# ════════════════════════════════════════════════════════════════════════════

class MACDHistStrategy(_Base):
    """
    Histograma MACD cambia de pendiente (de negativo creciente → compra)
    sin necesidad de cruzar cero — captura momentum temprano.
    """
    def __init__(self, fast=12, slow=26, signal=9, slope_bars=3, ema_trend=50):
        self.fast       = fast
        self.slow       = slow
        self.signal     = signal
        self.slope_bars = slope_bars
        self.ema_trend  = ema_trend
        self._min_bars  = slow + signal + slope_bars + ema_trend + 5
        self.swing_window = slow

    def generate_signals_batch(self, data):
        c    = data["close"].values
        ef   = _ema(c, self.fast)
        es   = _ema(c, self.slow)
        macd = ef - es
        sig  = _ema(np.where(np.isnan(macd), 0, macd), self.signal)
        hist = macd - sig
        et   = _ema(c, self.ema_trend)
        n    = len(c)
        sb   = self.slope_bars
        sigs = ["hold"] * n
        for i in range(sb, n):
            if any(np.isnan(hist[i-sb:i+1])) or np.isnan(et[i]):
                continue
            w = hist[i-sb:i+1]
            # Bullish: histograma negativo pero subiendo consistentemente
            rising   = bool(np.all(np.diff(w) > 0) and hist[i] < 0)
            # Bearish: histograma positivo pero bajando consistentemente
            falling  = bool(np.all(np.diff(w) < 0) and hist[i] > 0)
            if rising  and c[i] > et[i] * 0.99:
                sigs[i] = "buy"
            elif falling and c[i] < et[i] * 1.01:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"MACDHist(f={self.fast},s={self.slow},sb={self.slope_bars})"


# ════════════════════════════════════════════════════════════════════════════
# 6. Bollinger Band Touch
# ════════════════════════════════════════════════════════════════════════════

class BollingerBandStrategy(_Base):
    """
    Precio toca/cruza banda BB inferior (compra) o superior (venta)
    y rebota hacia la media. Confirmación: vela siguiente cierra en dirección.
    """
    def __init__(self, period=20, std_mult=2.0, confirm_close=True, rsi_period=14,
                 rsi_os=40, rsi_ob=60):
        self.period        = period
        self.std_mult      = std_mult
        self.confirm_close = confirm_close
        self.rsi_period    = rsi_period
        self.rsi_os        = rsi_os
        self.rsi_ob        = rsi_ob
        self._min_bars     = period + rsi_period + 5
        self.swing_window  = period

    def generate_signals_batch(self, data):
        c    = data["close"].values
        h    = data["high"].values
        l    = data["low"].values
        mid  = _sma(c, self.period)
        sd   = _std(c, self.period)
        ub   = mid + self.std_mult * sd
        lb   = mid - self.std_mult * sd
        rsi  = _rsi(c, self.rsi_period)
        n    = len(c)
        sigs = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan([ub[i], lb[i], mid[i], rsi[i]])):
                continue
            # Touch lower band: low <= lb → potential buy
            touch_lb = l[i] <= lb[i]
            touch_ub = h[i] >= ub[i]
            # Confirm: close above lb (rebote) + RSI sobrevendido
            if touch_lb and c[i] > lb[i] and rsi[i] < self.rsi_os + 15:
                sigs[i] = "buy"
            elif touch_ub and c[i] < ub[i] and rsi[i] > self.rsi_ob - 15:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"BB(p={self.period},std={self.std_mult})"


# ════════════════════════════════════════════════════════════════════════════
# 7. Rate of Change (ROC)
# ════════════════════════════════════════════════════════════════════════════

class ROCStrategy(_Base):
    """
    ROC cruza umbral positivo/negativo con EMA de tendencia.
    Captura momentum de precio.
    """
    def __init__(self, roc_period=12, threshold=1.5, ema_trend=50, smooth=3):
        self.roc_period = roc_period
        self.threshold  = threshold
        self.ema_trend  = ema_trend
        self.smooth     = smooth
        self._min_bars  = max(roc_period, ema_trend) + smooth + 5
        self.swing_window = roc_period

    def generate_signals_batch(self, data):
        c   = data["close"].values
        n   = len(c)
        roc = np.full(n, np.nan)
        for i in range(self.roc_period, n):
            roc[i] = (c[i] - c[i - self.roc_period]) / (c[i - self.roc_period] + 1e-10) * 100
        roc_s = _sma(roc, self.smooth)   # suavizado
        et    = _ema(c, self.ema_trend)
        sigs  = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan([roc_s[i], roc_s[i-1], et[i]])):
                continue
            cross_pos = roc_s[i-1] <= self.threshold  and roc_s[i] > self.threshold
            cross_neg = roc_s[i-1] >= -self.threshold and roc_s[i] < -self.threshold
            if cross_pos and c[i] > et[i]:
                sigs[i] = "buy"
            elif cross_neg and c[i] < et[i]:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"ROC(p={self.roc_period},thr={self.threshold})"


# ════════════════════════════════════════════════════════════════════════════
# 8. Money Flow Index (MFI)
# ════════════════════════════════════════════════════════════════════════════

class MFIStrategy(_Base):
    """MFI (RSI ponderado por volumen) en zonas extremas."""

    def __init__(self, period=14, oversold=25, overbought=75, ema_trend=50):
        self.period     = period
        self.oversold   = oversold
        self.overbought = overbought
        self.ema_trend  = ema_trend
        self._min_bars  = max(period, ema_trend) + 5
        self.swing_window = period

    def _mfi(self, h, l, c, v, p):
        tp  = (h + l + c) / 3.0
        mf  = tp * v
        n   = len(c)
        out = np.full(n, np.nan)
        for i in range(p, n):
            pos = sum(mf[j] for j in range(i-p+1, i+1) if tp[j] > tp[j-1])
            neg = sum(mf[j] for j in range(i-p+1, i+1) if tp[j] < tp[j-1])
            out[i] = 100 - 100 / (1 + pos / (neg + 1e-10))
        return out

    def generate_signals_batch(self, data):
        h   = data["high"].values
        l   = data["low"].values
        c   = data["close"].values
        v   = data["volume"].values if "volume" in data.columns else np.ones(len(c))
        mfi = self._mfi(h, l, c, v, self.period)
        et  = _ema(c, self.ema_trend)
        n   = len(c)
        sigs = ["hold"] * n
        for i in range(1, n):
            if any(np.isnan([mfi[i], mfi[i-1], et[i]])):
                continue
            cross_up   = mfi[i-1] <= self.oversold   and mfi[i] > self.oversold
            cross_down = mfi[i-1] >= self.overbought  and mfi[i] < self.overbought
            if cross_up   and c[i] > et[i]:
                sigs[i] = "buy"
            elif cross_down and c[i] < et[i]:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"MFI(p={self.period},os={self.oversold})"


# ════════════════════════════════════════════════════════════════════════════
# 9. Heikin-Ashi
# ════════════════════════════════════════════════════════════════════════════

class HeikinAshiStrategy(_Base):
    """
    Velas Heikin-Ashi: señal cuando se produce un cambio de color
    con confirmación de N velas del mismo color + EMA.
    """
    def __init__(self, confirm_bars=2, ema_trend=50, rsi_period=14,
                 rsi_os=45, rsi_ob=55):
        self.confirm_bars = confirm_bars
        self.ema_trend    = ema_trend
        self.rsi_period   = rsi_period
        self.rsi_os       = rsi_os
        self.rsi_ob       = rsi_ob
        self._min_bars    = max(ema_trend, rsi_period) + confirm_bars + 5
        self.swing_window = ema_trend // 5

    def _heikin_ashi(self, o, h, l, c):
        n     = len(c)
        ha_c  = (o + h + l + c) / 4.0
        ha_o  = np.zeros(n)
        ha_o[0] = (o[0] + c[0]) / 2.0
        for i in range(1, n):
            ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2.0
        ha_bull = ha_c > ha_o   # True = vela alcista
        return ha_bull

    def generate_signals_batch(self, data):
        o   = data["open"].values
        h   = data["high"].values
        l   = data["low"].values
        c   = data["close"].values
        bull = self._heikin_ashi(o, h, l, c)
        et   = _ema(c, self.ema_trend)
        rsi  = _rsi(c, self.rsi_period)
        n    = len(c)
        cb   = self.confirm_bars
        sigs = ["hold"] * n
        for i in range(cb, n):
            if np.isnan(et[i]) or np.isnan(rsi[i]):
                continue
            # N velas alcistas consecutivas tras una bajista
            was_bear = not bull[i - cb]
            all_bull  = all(bull[i - j] for j in range(cb))
            was_bull  = bull[i - cb]
            all_bear  = all(not bull[i - j] for j in range(cb))
            if was_bear and all_bull and c[i] > et[i] and rsi[i] < self.rsi_ob:
                sigs[i] = "buy"
            elif was_bull and all_bear and c[i] < et[i] and rsi[i] > self.rsi_os:
                sigs[i] = "sell"
        return sigs

    def __repr__(self):
        return f"HeikinAshi(cb={self.confirm_bars},et={self.ema_trend})"


# ════════════════════════════════════════════════════════════════════════════
# 10. Z-Score Mean Reversion
# ════════════════════════════════════════════════════════════════════════════

class ZScoreStrategy(_Base):
    """
    Z-score del precio: entra cuando precio está N desviaciones de la media
    (mean reversion) o sale cuando cruza de vuelta a la media.
    """
    def __init__(self, period=30, z_entry=2.0, z_exit=0.5, ema_trend=100,
                 require_trend_align=False):
        self.period             = period
        self.z_entry            = z_entry
        self.z_exit             = z_exit
        self.ema_trend          = ema_trend
        self.require_trend_align = require_trend_align
        self._min_bars          = max(period, ema_trend) + 5
        self.swing_window       = period

    def generate_signals_batch(self, data):
        c   = data["close"].values
        n   = len(c)
        z   = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            w = c[i - self.period + 1:i + 1]
            m, s = w.mean(), w.std(ddof=1)
            z[i] = (c[i] - m) / (s + 1e-10)
        et   = _ema(c, self.ema_trend)
        sigs = ["hold"] * n
        pos  = None
        for i in range(1, n):
            if np.isnan(z[i]) or np.isnan(et[i]):
                continue
            if pos is None:
                if z[i] <= -self.z_entry:
                    if not self.require_trend_align or c[i] > et[i]:
                        sigs[i] = "buy"; pos = "buy"
                elif z[i] >= self.z_entry:
                    if not self.require_trend_align or c[i] < et[i]:
                        sigs[i] = "sell"; pos = "sell"
            else:
                if pos == "buy"  and z[i] >= -self.z_exit:
                    sigs[i] = "sell"; pos = None
                elif pos == "sell" and z[i] <= self.z_exit:
                    sigs[i] = "buy";  pos = None
        return sigs

    def __repr__(self):
        return f"ZScore(p={self.period},z={self.z_entry})"


# ── Registro ──────────────────────────────────────────────────────────────────

ALL_STRATEGY_CLASSES_2 = {
    "vwap":            VWAPStrategy,
    "parabolic_sar":   ParabolicSARStrategy,
    "williams_r":      WilliamsRStrategy,
    "keltner":         KeltnerBreakoutStrategy,
    "macd_hist":       MACDHistStrategy,
    "bollinger_band":  BollingerBandStrategy,
    "roc":             ROCStrategy,
    "mfi":             MFIStrategy,
    "heikin_ashi":     HeikinAshiStrategy,
    "zscore":          ZScoreStrategy,
}
