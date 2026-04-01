"""
macd_divergence.py

Estrategia basada en divergencias MACD.

Lógica:
  - MACD = EMA(fast) - EMA(slow)
  - Signal = EMA(signal_period) del MACD
  - Histograma = MACD - Signal

Señales:
  BUY:  precio hace mínimo más bajo (bearish price swing) pero histograma
        hace mínimo más alto (divergencia alcista) → posible reversión alcista
  SELL: precio hace máximo más alto (bullish price swing) pero histograma
        hace máximo más bajo (divergencia bajista) → posible reversión bajista

Confirmación adicional (opcional):
  - Cruce MACD sobre la línea de señal (cruce_confirm=True)
  - RSI no en zona extrema opuesta (rsi_guard=True)
"""

import numpy as np
import pandas as pd


def _ema(series: np.ndarray, period: int) -> np.ndarray:
    """EMA con warmup completo usando Wilder/pandas-like smoothing."""
    alpha  = 2.0 / (period + 1)
    result = np.empty(len(series))
    result[:] = np.nan
    # Warmup: primera EMA = media de los primeros `period` valores
    if len(series) < period:
        return result
    result[period - 1] = series[:period].mean()
    for i in range(period, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI estándar."""
    n      = len(closes)
    result = np.full(n, np.nan)
    if n < period + 1:
        return result
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / (avg_loss + 1e-10)
        result[i + 1] = 100 - 100 / (1 + rs)
    return result


class MACDDivergenceStrategy:
    """
    Estrategia de divergencia MACD.

    Parámetros
    ----------
    fast_period   : período EMA rápida (12)
    slow_period   : período EMA lenta (26)
    signal_period : período EMA de la señal MACD (9)
    lookback      : velas hacia atrás para detectar divergencia (5)
    cruce_confirm : exigir cruce MACD/señal para confirmar (False)
    rsi_guard     : filtrar con RSI para evitar señales en zona extrema (True)
    rsi_period    : período RSI para el filtro
    swing_window  : alias para compatibilidad con FastBacktester
    require_fvg   : alias para compatibilidad (siempre False)
    use_choch_filter: alias (siempre False)
    """

    def __init__(self,
                 fast_period: int   = 12,
                 slow_period: int   = 26,
                 signal_period: int = 9,
                 lookback: int      = 5,
                 cruce_confirm: bool = False,
                 rsi_guard: bool    = True,
                 rsi_period: int    = 14):
        self.fast_period   = fast_period
        self.slow_period   = slow_period
        self.signal_period = signal_period
        self.lookback      = lookback
        self.cruce_confirm = cruce_confirm
        self.rsi_guard     = rsi_guard
        self.rsi_period    = rsi_period

        # Compatibilidad FastBacktester
        self.swing_window      = slow_period
        self.require_fvg       = False
        self.use_choch_filter  = False

        self._min_bars = slow_period + signal_period + lookback + 2

    def _compute(self, closes: np.ndarray):
        ema_fast   = _ema(closes, self.fast_period)
        ema_slow   = _ema(closes, self.slow_period)
        macd_line  = ema_fast - ema_slow
        signal_line = _ema(np.where(np.isnan(macd_line), 0.0, macd_line), self.signal_period)
        histogram  = macd_line - signal_line
        return macd_line, signal_line, histogram

    def generate_signal(self, data) -> dict:
        """
        Genera señal con divergencia MACD.

        Retorna dict con: signal, macd, signal_line, histogram, divergence_type
        """
        if data is None or len(data) < self._min_bars:
            return {"signal": "hold", "reason": "insufficient_data"}

        closes = data["close"].values
        macd_line, signal_line, histogram = self._compute(closes)

        n  = len(closes)
        lb = self.lookback
        i  = n - 1  # vela actual

        # Necesitamos histograma válido en las últimas `lookback` velas
        hist_window  = histogram[i - lb: i + 1]
        price_window = closes[i - lb: i + 1]

        if np.any(np.isnan(hist_window)):
            return {"signal": "hold", "reason": "nan_in_window"}

        # ── Divergencia alcista ───────────────────────────────────────
        # Precio: nuevo mínimo; histograma: mínimo más alto
        price_low  = price_window[-1] == price_window.min()
        hist_low   = hist_window[-1] > hist_window.min()   # "más alto" = menos negativo

        bullish_div = (price_low and hist_low and histogram[i] < 0)

        # ── Divergencia bajista ───────────────────────────────────────
        # Precio: nuevo máximo; histograma: máximo más bajo
        price_high = price_window[-1] == price_window.max()
        hist_high  = hist_window[-1] < hist_window.max()   # "más bajo" = menos positivo

        bearish_div = (price_high and hist_high and histogram[i] > 0)

        if not bullish_div and not bearish_div:
            return {"signal": "hold", "reason": "no_divergence",
                    "histogram": float(histogram[i])}

        # ── Confirmaciones opcionales ─────────────────────────────────
        if self.cruce_confirm:
            # Cruce MACD sobre señal en las últimas 2 velas
            if bullish_div:
                crossed = (macd_line[i] > signal_line[i] and
                           macd_line[i - 1] <= signal_line[i - 1])
                if not crossed:
                    return {"signal": "hold", "reason": "no_macd_cross"}
            else:
                crossed = (macd_line[i] < signal_line[i] and
                           macd_line[i - 1] >= signal_line[i - 1])
                if not crossed:
                    return {"signal": "hold", "reason": "no_macd_cross"}

        if self.rsi_guard:
            rsi_vals = _rsi(closes, self.rsi_period)
            rsi_now  = rsi_vals[i]
            if not np.isnan(rsi_now):
                if bullish_div and rsi_now > 70:  # sobrecomprado → no comprar
                    return {"signal": "hold", "reason": "rsi_overbought"}
                if bearish_div and rsi_now < 30:   # sobrevendido → no vender
                    return {"signal": "hold", "reason": "rsi_oversold"}

        signal = "buy" if bullish_div else "sell"
        return {
            "signal":         signal,
            "macd":           round(float(macd_line[i]), 6),
            "signal_line":    round(float(signal_line[i]), 6),
            "histogram":      round(float(histogram[i]), 6),
            "divergence_type": "bullish" if bullish_div else "bearish",
        }

    def __repr__(self):
        return (f"MACDDivergenceStrategy(fast={self.fast_period}, "
                f"slow={self.slow_period}, signal={self.signal_period}, "
                f"lookback={self.lookback})")
