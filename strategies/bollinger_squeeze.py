"""
bollinger_squeeze.py

Estrategia Bollinger Band Squeeze (TTM Squeeze simplificado).

Lógica:
  - Squeeze: las Bandas de Bollinger (BB) están DENTRO del canal de Keltner (KC)
    → mercado comprimido, acumulando energía antes de una ruptura
  - Cuando BB sale del KC (squeeze se libera), se espera la dirección:
    - Histograma de momentum positivo creciente → BUY
    - Histograma de momentum negativo decreciente → SELL

Indicadores:
  - BB: media ± bb_mult * std(close, bb_period)
  - KC: media ± kc_mult * ATR(kc_period)
  - Momentum: close - media(close[n-momentum_period:n])  (proxy de TTM momentum)
"""

import numpy as np
import pandas as pd


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = arr[i - period + 1: i + 1].mean()
    return result


def _std(arr: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = arr[i - period + 1: i + 1].std(ddof=1)
    return result


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n  = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i]  - close[i - 1]))
    # Wilder smoothing
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class BollingerSqueezeStrategy:
    """
    Estrategia de squeeze de Bollinger Bands / Canal de Keltner.

    Parámetros
    ----------
    bb_period      : período para las Bandas de Bollinger (20)
    bb_mult        : multiplicador para BB (2.0)
    kc_period      : período para el Canal de Keltner (20)
    kc_mult        : multiplicador ATR para KC (1.5)
    momentum_period: período para el oscilador de momentum (12)
    confirm_bars   : velas consecutivas con momentum direccional para confirmar (2)
    swing_window   : alias compatibilidad FastBacktester
    require_fvg    : alias (False)
    use_choch_filter: alias (False)
    """

    def __init__(self,
                 bb_period: int       = 20,
                 bb_mult: float       = 2.0,
                 kc_period: int       = 20,
                 kc_mult: float       = 1.5,
                 momentum_period: int = 12,
                 confirm_bars: int    = 2):
        self.bb_period       = bb_period
        self.bb_mult         = bb_mult
        self.kc_period       = kc_period
        self.kc_mult         = kc_mult
        self.momentum_period = momentum_period
        self.confirm_bars    = confirm_bars

        # Compatibilidad
        self.swing_window       = bb_period
        self.require_fvg        = False
        self.use_choch_filter   = False

        self._min_bars = max(bb_period, kc_period) + momentum_period + confirm_bars + 2

    def _is_squeeze(self, bb_upper: float, bb_lower: float,
                    kc_upper: float, kc_lower: float) -> bool:
        """True cuando BB está completamente dentro del Canal de Keltner."""
        return bb_upper <= kc_upper and bb_lower >= kc_lower

    def generate_signal(self, data) -> dict:
        """
        Genera señal cuando el squeeze se libera con momentum confirmado.
        """
        if data is None or len(data) < self._min_bars:
            return {"signal": "hold", "reason": "insufficient_data"}

        closes = data["close"].values
        highs  = data["high"].values
        lows   = data["low"].values
        n      = len(closes)

        # ── Bollinger Bands ───────────────────────────────────────────
        bb_mid   = _sma(closes, self.bb_period)
        bb_std   = _std(closes, self.bb_period)
        bb_upper = bb_mid + self.bb_mult * bb_std
        bb_lower = bb_mid - self.bb_mult * bb_std

        # ── Canal de Keltner ──────────────────────────────────────────
        atr_vals = _atr(highs, lows, closes, self.kc_period)
        kc_mid   = _sma(closes, self.kc_period)
        kc_upper = kc_mid + self.kc_mult * atr_vals
        kc_lower = kc_mid - self.kc_mult * atr_vals

        # ── Momentum (proxy TTM) ──────────────────────────────────────
        momentum = np.full(n, np.nan)
        for i in range(self.momentum_period, n):
            mid_highest = (highs[i - self.momentum_period: i + 1].max() +
                           lows[i - self.momentum_period: i + 1].min()) / 2
            delta_close = closes[i - self.momentum_period: i + 1].mean()
            momentum[i] = closes[i] - (mid_highest + delta_close) / 2

        i = n - 1

        # Verificar datos válidos
        if any(np.isnan(v) for v in [bb_upper[i], bb_lower[i],
                                      kc_upper[i], kc_lower[i], momentum[i]]):
            return {"signal": "hold", "reason": "nan_in_indicators"}

        # ── Detectar squeeze previo y liberación actual ───────────────
        # La vela actual NO está en squeeze (se liberó)
        in_squeeze_now = self._is_squeeze(bb_upper[i], bb_lower[i],
                                          kc_upper[i], kc_lower[i])
        if in_squeeze_now:
            return {"signal": "hold", "reason": "still_in_squeeze",
                    "momentum": round(float(momentum[i]), 4)}

        # Comprobar que hubo squeeze en alguna de las últimas `confirm_bars+1` velas
        prev_had_squeeze = False
        for j in range(1, self.confirm_bars + 3):
            k = i - j
            if k < 0:
                break
            if any(np.isnan(v) for v in [bb_upper[k], bb_lower[k],
                                          kc_upper[k], kc_lower[k]]):
                break
            if self._is_squeeze(bb_upper[k], bb_lower[k], kc_upper[k], kc_lower[k]):
                prev_had_squeeze = True
                break

        if not prev_had_squeeze:
            return {"signal": "hold", "reason": "no_prior_squeeze"}

        # ── Confirmar momentum con `confirm_bars` barras consecutivas ─
        cb = self.confirm_bars
        if i < cb:
            return {"signal": "hold", "reason": "not_enough_bars"}

        mom_window = momentum[i - cb + 1: i + 1]
        if np.any(np.isnan(mom_window)):
            return {"signal": "hold", "reason": "nan_momentum"}

        all_positive = bool(np.all(mom_window > 0))
        all_negative = bool(np.all(mom_window < 0))
        # Momentum creciente para mayor convicción
        rising  = bool(all_positive and mom_window[-1] >= mom_window[-2])
        falling = bool(all_negative and mom_window[-1] <= mom_window[-2])

        if rising:
            signal = "buy"
        elif falling:
            signal = "sell"
        else:
            return {"signal": "hold", "reason": "momentum_not_directional",
                    "momentum": round(float(momentum[i]), 4)}

        return {
            "signal":        signal,
            "momentum":      round(float(momentum[i]), 4),
            "squeeze_fired": True,
            "bb_width":      round(float(bb_upper[i] - bb_lower[i]), 4),
            "kc_width":      round(float(kc_upper[i] - kc_lower[i]), 4),
        }

    def __repr__(self):
        return (f"BollingerSqueezeStrategy(bb={self.bb_period}x{self.bb_mult}, "
                f"kc={self.kc_period}x{self.kc_mult}, mom={self.momentum_period})")
