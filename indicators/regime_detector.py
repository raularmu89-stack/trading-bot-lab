"""
regime_detector.py

Detecta el régimen de mercado actual para seleccionar la estrategia óptima.

Regímenes:
  - "strong_trend_bull"  : tendencia alcista fuerte  (ADX>25, EMA fan alcista)
  - "strong_trend_bear"  : tendencia bajista fuerte  (ADX>25, EMA fan bajista)
  - "weak_trend_bull"    : tendencia alcista débil   (ADX 18-25, precio > EMA50)
  - "weak_trend_bear"    : tendencia bajista débil   (ADX 18-25, precio < EMA50)
  - "ranging"            : mercado lateral           (ADX<18, ATR bajo)
  - "breakout"           : ruptura de rango          (ATR spike + BOS reciente)
  - "high_volatility"    : volatilidad extrema       (ATR > 2.5× media)
  - "mean_reversion"     : sobre-extensión + P/D     (RSI extremo + zona P/D)

Uso:
    from indicators.regime_detector import detect_regime, RegimeDetector

    regime = detect_regime(df)
    # → {"regime": "strong_trend_bull", "adx": 32.1, "atr_ratio": 1.1, ...}
"""

import numpy as np
import pandas as pd


# ── Indicadores base ──────────────────────────────────────────────────────────

def _atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - np.roll(close, 1)).abs() if hasattr(high, 'index') else \
          np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = (low  - np.roll(close, 1)).abs() if hasattr(low,  'index') else \
          np.abs(low  - np.concatenate([[close[0]], close[:-1]]))

    if isinstance(high, pd.Series):
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()
    else:
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.zeros(len(tr))
        atr[0] = tr[0]
        alpha = 1.0 / period
        for i in range(1, len(tr)):
            atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
        return atr


def _ema(series, period):
    if isinstance(series, pd.Series):
        return series.ewm(span=period, adjust=False).mean()
    result = np.zeros(len(series))
    alpha  = 2.0 / (period + 1)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = result[i-1] * (1 - alpha) + series[i] * alpha
    return result


def _adx(high, low, close, period=14):
    """
    Average Directional Index (ADX).
    Retorna dict con arrays: adx, plus_di, minus_di.
    """
    if isinstance(high, pd.Series):
        h, l, c = high.values, low.values, close.values
    else:
        h, l, c = high, low, close

    n = len(c)
    atr_arr = _atr(h, l, c, period)

    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up   = h[i] - h[i-1]
        down = l[i-1] - l[i]
        plus_dm[i]  = up   if up > down and up > 0   else 0.0
        minus_dm[i] = down if down > up and down > 0 else 0.0

    alpha = 1.0 / period
    sm_plus = sm_minus = sm_atr = 0.0
    plus_di_arr  = np.zeros(n)
    minus_di_arr = np.zeros(n)
    dx_arr       = np.zeros(n)

    for i in range(n):
        sm_plus  = sm_plus  * (1 - alpha) + plus_dm[i]  * alpha
        sm_minus = sm_minus * (1 - alpha) + minus_dm[i] * alpha
        sm_atr   = sm_atr   * (1 - alpha) + atr_arr[i]  * alpha

        if sm_atr > 0:
            pdi = 100 * sm_plus  / sm_atr
            mdi = 100 * sm_minus / sm_atr
        else:
            pdi = mdi = 0.0
        plus_di_arr[i]  = pdi
        minus_di_arr[i] = mdi

        if pdi + mdi > 0:
            dx_arr[i] = 100 * abs(pdi - mdi) / (pdi + mdi)

    adx_arr = np.zeros(n)
    adx_arr[0] = dx_arr[0]
    for i in range(1, n):
        adx_arr[i] = adx_arr[i-1] * (1 - alpha) + dx_arr[i] * alpha

    return {"adx": adx_arr, "plus_di": plus_di_arr, "minus_di": minus_di_arr}


def _rsi(close, period=14):
    if isinstance(close, pd.Series):
        c = close.values
    else:
        c = close
    n = len(c)
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        d = c[i] - c[i-1]
        gains[i]  = d if d > 0 else 0.0
        losses[i] = -d if d < 0 else 0.0
    alpha = 1.0 / period
    avg_gain = avg_loss = 0.0
    rsi_arr = np.full(n, 50.0)
    for i in range(n):
        avg_gain  = avg_gain  * (1 - alpha) + gains[i]  * alpha
        avg_loss  = avg_loss  * (1 - alpha) + losses[i] * alpha
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi_arr[i] = 100 - 100 / (1 + rs)
        else:
            rsi_arr[i] = 100.0 if avg_gain > 0 else 50.0
    return rsi_arr


def _range_width(high, low, window=50):
    """Ancho del rango como % del precio medio en la ventana."""
    if isinstance(high, pd.Series):
        h, l = high.values, low.values
    else:
        h, l = high, low
    n = len(h)
    width = np.zeros(n)
    for i in range(window, n):
        top    = h[i-window:i].max()
        bottom = l[i-window:i].min()
        mid    = (top + bottom) / 2
        width[i] = (top - bottom) / mid if mid > 0 else 0.0
    return width


# ── Detector principal ────────────────────────────────────────────────────────

# Umbrales configurables
ADX_STRONG     = 25.0   # ADX > 25 → tendencia fuerte
ADX_WEAK       = 18.0   # ADX 18-25 → tendencia débil
ATR_HV_MULT    = 2.5    # ATR > 2.5× media → alta volatilidad
ATR_LV_MULT    = 0.6    # ATR < 0.6× media → baja volatilidad / ranging
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD   = 30.0
RANGE_WINDOW   = 50
ATR_WINDOW     = 50     # ventana para calcular ATR medio


def detect_regime(data: pd.DataFrame,
                  adx_period: int = 14,
                  atr_period: int = 14,
                  rsi_period: int = 14,
                  ema_fast: int = 20,
                  ema_slow: int = 50) -> dict:
    """
    Detecta el régimen de la última vela del DataFrame.

    Parámetros
    ----------
    data        : DataFrame OHLCV
    adx_period  : período del ADX
    atr_period  : período del ATR
    rsi_period  : período del RSI
    ema_fast    : EMA rápida (para tendencia)
    ema_slow    : EMA lenta (para tendencia)

    Retorna dict:
    {
        "regime"     : str,
        "adx"        : float,
        "plus_di"    : float,
        "minus_di"   : float,
        "atr"        : float,
        "atr_ratio"  : float,   # atr / atr_mean(50 velas)
        "rsi"        : float,
        "ema_fast"   : float,
        "ema_slow"   : float,
        "price"      : float,
        "range_width": float,
        "bos_recent" : bool,
    }
    """
    if data is None or len(data) < max(adx_period, ema_slow, RANGE_WINDOW) + 5:
        return {"regime": "insufficient_data"}

    h = data["high"].values
    l = data["low"].values
    c = data["close"].values
    n = len(c)

    # Indicadores
    adx_data = _adx(h, l, c, adx_period)
    atr_arr  = _atr(h, l, c, atr_period)
    rsi_arr  = _rsi(c, rsi_period)
    ema_f    = _ema(c, ema_fast)
    ema_s    = _ema(c, ema_slow)
    rw       = _range_width(h, l, RANGE_WINDOW)

    # Valores actuales (última vela)
    adx_val   = float(adx_data["adx"][-1])
    plus_di   = float(adx_data["plus_di"][-1])
    minus_di  = float(adx_data["minus_di"][-1])
    atr_val   = float(atr_arr[-1])
    atr_mean  = float(np.mean(atr_arr[max(0, n - ATR_WINDOW):n]))
    atr_ratio = atr_val / atr_mean if atr_mean > 0 else 1.0
    rsi_val   = float(rsi_arr[-1])
    ema_f_val = float(ema_f[-1])
    ema_s_val = float(ema_s[-1])
    price     = float(c[-1])
    rw_val    = float(rw[-1])

    # BOS reciente: precio rompió el swing high/low de las últimas 20 velas
    lookback  = min(20, n - 1)
    bos_recent = (
        price > h[n - 1 - lookback: n - 1].max() or
        price < l[n - 1 - lookback: n - 1].min()
    )

    # ── Lógica de clasificación ───────────────────────────────────────────────
    regime = _classify(
        adx_val, plus_di, minus_di,
        atr_ratio, rsi_val,
        ema_f_val, ema_s_val, price,
        bos_recent, rw_val
    )

    return {
        "regime":      regime,
        "adx":         round(adx_val,  2),
        "plus_di":     round(plus_di,  2),
        "minus_di":    round(minus_di, 2),
        "atr":         round(atr_val,  6),
        "atr_ratio":   round(atr_ratio, 3),
        "rsi":         round(rsi_val,  2),
        "ema_fast":    round(ema_f_val, 6),
        "ema_slow":    round(ema_s_val, 6),
        "price":       round(price,    6),
        "range_width": round(rw_val,   4),
        "bos_recent":  bos_recent,
    }


def _classify(adx, plus_di, minus_di, atr_ratio, rsi,
              ema_fast, ema_slow, price, bos_recent, range_width):
    """Lógica de clasificación del régimen."""

    # 1. Alta volatilidad — prioridad máxima (evitar ruido extremo)
    if atr_ratio > ATR_HV_MULT:
        return "high_volatility"

    # 2. Mean reversion: RSI extremo + precio fuera de equilibrio
    if rsi >= RSI_OVERBOUGHT and price > ema_fast > ema_slow:
        return "mean_reversion_bear"
    if rsi <= RSI_OVERSOLD and price < ema_fast < ema_slow:
        return "mean_reversion_bull"

    # 3. Breakout: ATR spike moderado + BOS reciente + rango previo estrecho
    if bos_recent and 1.3 < atr_ratio <= ATR_HV_MULT and range_width < 0.04:
        return "breakout"

    # 4. Tendencia fuerte
    if adx >= ADX_STRONG:
        if plus_di > minus_di:
            return "strong_trend_bull"
        else:
            return "strong_trend_bear"

    # 5. Tendencia débil
    if adx >= ADX_WEAK:
        if plus_di > minus_di and price > ema_slow:
            return "weak_trend_bull"
        if minus_di > plus_di and price < ema_slow:
            return "weak_trend_bear"

    # 6. Ranging
    return "ranging"


# ── Detector con ventana histórica (para backtests) ───────────────────────────

def detect_regime_series(data: pd.DataFrame,
                         adx_period: int = 14,
                         atr_period: int = 14,
                         rsi_period: int = 14,
                         ema_fast: int = 20,
                         ema_slow: int = 50) -> list:
    """
    Calcula el régimen para CADA vela del DataFrame (para backtests).
    Retorna una lista de strings, una por vela.
    O(n) al precomputar todos los arrays.
    """
    if data is None or len(data) < max(adx_period, ema_slow, RANGE_WINDOW) + 5:
        return ["insufficient_data"] * (len(data) if data is not None else 0)

    h = data["high"].values
    l = data["low"].values
    c = data["close"].values
    n = len(c)

    adx_data = _adx(h, l, c, adx_period)
    atr_arr  = _atr(h, l, c, atr_period)
    rsi_arr  = _rsi(c, rsi_period)
    ema_f    = _ema(c, ema_fast)
    ema_s    = _ema(c, ema_slow)
    rw       = _range_width(h, l, RANGE_WINDOW)

    # ATR rolling mean (ventana ATR_WINDOW)
    atr_mean_arr = np.zeros(n)
    for i in range(n):
        start = max(0, i - ATR_WINDOW)
        atr_mean_arr[i] = np.mean(atr_arr[start: i + 1])

    regimes = []
    for i in range(n):
        atr_ratio  = atr_arr[i] / atr_mean_arr[i] if atr_mean_arr[i] > 0 else 1.0
        lookback   = min(20, i)
        bos_recent = i > 0 and (
            c[i] > h[max(0, i - lookback): i].max() or
            c[i] < l[max(0, i - lookback): i].min()
        )
        regime = _classify(
            adx_data["adx"][i], adx_data["plus_di"][i], adx_data["minus_di"][i],
            atr_ratio, rsi_arr[i],
            ema_f[i], ema_s[i], c[i],
            bos_recent, rw[i]
        )
        regimes.append(regime)

    return regimes


# ── Clase OOP opcional ────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Interfaz OOP para detección de régimen.

    Ejemplo:
        rd = RegimeDetector()
        result = rd.detect(df)          # última vela
        regimes = rd.detect_all(df)     # todas las velas (backtest)
    """

    def __init__(self, adx_period=14, atr_period=14, rsi_period=14,
                 ema_fast=20, ema_slow=50):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.ema_fast   = ema_fast
        self.ema_slow   = ema_slow

    def detect(self, data: pd.DataFrame) -> dict:
        return detect_regime(data, self.adx_period, self.atr_period,
                             self.rsi_period, self.ema_fast, self.ema_slow)

    def detect_all(self, data: pd.DataFrame) -> list:
        return detect_regime_series(data, self.adx_period, self.atr_period,
                                    self.rsi_period, self.ema_fast, self.ema_slow)

    @staticmethod
    def regime_label(regime: str) -> str:
        labels = {
            "strong_trend_bull":  "Tendencia Fuerte ↑",
            "strong_trend_bear":  "Tendencia Fuerte ↓",
            "weak_trend_bull":    "Tendencia Débil ↑",
            "weak_trend_bear":    "Tendencia Débil ↓",
            "ranging":            "Lateral / Rango",
            "breakout":           "Breakout",
            "high_volatility":    "Alta Volatilidad",
            "mean_reversion_bull":"Mean Reversion ↑",
            "mean_reversion_bear":"Mean Reversion ↓",
            "insufficient_data":  "Datos insuficientes",
        }
        return labels.get(regime, regime)

    @staticmethod
    def is_trending(regime: str) -> bool:
        return regime in ("strong_trend_bull", "strong_trend_bear",
                          "weak_trend_bull", "weak_trend_bear")

    @staticmethod
    def is_bullish(regime: str) -> bool:
        return regime in ("strong_trend_bull", "weak_trend_bull",
                          "mean_reversion_bull")

    @staticmethod
    def is_bearish(regime: str) -> bool:
        return regime in ("strong_trend_bear", "weak_trend_bear",
                          "mean_reversion_bear")
