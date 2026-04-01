"""
momentum_burst.py

Estrategia Momentum Burst (ruptura explosiva de momentum).

Lógica:
  Un "burst" de momentum ocurre cuando:
  1. El precio lleva N velas en rango comprimido (ATR bajo respecto a media)
  2. Se produce una vela de expansión significativa (cuerpo > umbral × ATR)
  3. El volumen acompaña la ruptura (vol > vol_mult × media_vol)
  4. El RSI confirma la dirección (> 55 para buy, < 45 para sell)

  El burst se clasifica alcista si la vela de expansión cierra alcista,
  bajista si cierra bajista.

Idea: capturar las primeras velas de un movimiento impulsivo tras un período
de consolidación, antes de que la tendencia se establezca.
"""

import numpy as np
import pandas as pd


def _atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
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


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
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


class MomentumBurstStrategy:
    """
    Estrategia de ruptura explosiva por momentum.

    Parámetros
    ----------
    atr_period       : período ATR (14)
    compression_bars : velas previas para evaluar compresión (10)
    compression_ratio: ATR actual / ATR_media < ratio → comprimido (0.7)
    burst_mult       : cuerpo_vela > burst_mult × ATR → "burst" (1.5)
    vol_mult         : volumen > vol_mult × media_vol para confirmar (1.5)
    vol_period       : período para la media de volumen (20)
    rsi_period       : período RSI para confirmación (14)
    rsi_bull_min     : RSI mínimo para señal de compra (55)
    rsi_bear_max     : RSI máximo para señal de venta (45)
    swing_window     : alias compatibilidad FastBacktester
    require_fvg      : alias (False)
    use_choch_filter : alias (False)
    """

    def __init__(self,
                 atr_period: int        = 14,
                 compression_bars: int  = 10,
                 compression_ratio: float = 0.7,
                 burst_mult: float      = 1.5,
                 vol_mult: float        = 1.5,
                 vol_period: int        = 20,
                 rsi_period: int        = 14,
                 rsi_bull_min: float    = 55.0,
                 rsi_bear_max: float    = 45.0):
        self.atr_period        = atr_period
        self.compression_bars  = compression_bars
        self.compression_ratio = compression_ratio
        self.burst_mult        = burst_mult
        self.vol_mult          = vol_mult
        self.vol_period        = vol_period
        self.rsi_period        = rsi_period
        self.rsi_bull_min      = rsi_bull_min
        self.rsi_bear_max      = rsi_bear_max

        # Compatibilidad
        self.swing_window       = atr_period
        self.require_fvg        = False
        self.use_choch_filter   = False

        self._min_bars = max(atr_period, vol_period, rsi_period) + compression_bars + 2

    def generate_signal(self, data) -> dict:
        """
        Genera señal cuando detecta un burst de momentum tras compresión.
        """
        if data is None or len(data) < self._min_bars:
            return {"signal": "hold", "reason": "insufficient_data"}

        closes  = data["close"].values
        opens   = data["open"].values
        highs   = data["high"].values
        lows    = data["low"].values
        n       = len(closes)
        i       = n - 1

        # ── ATR ───────────────────────────────────────────────────────
        atr = _atr_series(highs, lows, closes, self.atr_period)
        if np.isnan(atr[i]):
            return {"signal": "hold", "reason": "atr_nan"}

        # ── Compresión: ATR actual vs media de compression_bars anteriores
        cb     = self.compression_bars
        if i < cb + 1:
            return {"signal": "hold", "reason": "not_enough_bars"}

        atr_hist = atr[i - cb: i]   # excluir vela actual
        if np.any(np.isnan(atr_hist)):
            return {"signal": "hold", "reason": "atr_hist_nan"}

        atr_mean      = float(atr_hist.mean())
        current_atr   = float(atr[i])
        in_compression = current_atr <= self.compression_ratio * atr_mean

        # También aceptar si alguna de las últimas 3 velas estaba comprimida
        recently_compressed = False
        for j in range(1, min(4, cb)):
            if not np.isnan(atr[i - j]) and atr[i - j] <= self.compression_ratio * atr_mean:
                recently_compressed = True
                break

        if not in_compression and not recently_compressed:
            return {"signal": "hold", "reason": "no_compression",
                    "atr_ratio": round(current_atr / (atr_mean + 1e-10), 3)}

        # ── Burst: vela actual tiene cuerpo grande ────────────────────
        candle_body = abs(closes[i] - opens[i])
        burst_threshold = self.burst_mult * atr_mean

        if candle_body < burst_threshold:
            return {"signal": "hold", "reason": "no_burst",
                    "body_ratio": round(candle_body / (atr_mean + 1e-10), 3)}

        bullish_candle = closes[i] > opens[i]
        bearish_candle = closes[i] < opens[i]

        # ── Volumen ───────────────────────────────────────────────────
        if "volume" in data.columns:
            vol     = data["volume"].values
            vp      = self.vol_period
            if i >= vp and not np.any(np.isnan(vol[i - vp: i])):
                vol_mean = float(vol[i - vp: i].mean())
                vol_ok   = float(vol[i]) >= self.vol_mult * vol_mean
            else:
                vol_ok = True   # si no hay datos de volumen, no filtrar
        else:
            vol_ok = True

        if not vol_ok:
            return {"signal": "hold", "reason": "low_volume",
                    "burst_ratio": round(candle_body / (atr_mean + 1e-10), 3)}

        # ── RSI ───────────────────────────────────────────────────────
        rsi_vals = _rsi(closes, self.rsi_period)
        rsi_now  = float(rsi_vals[i]) if not np.isnan(rsi_vals[i]) else 50.0

        if bullish_candle:
            if rsi_now < self.rsi_bull_min:
                return {"signal": "hold", "reason": "rsi_too_low",
                        "rsi": round(rsi_now, 1)}
            signal = "buy"
        elif bearish_candle:
            if rsi_now > self.rsi_bear_max:
                return {"signal": "hold", "reason": "rsi_too_high",
                        "rsi": round(rsi_now, 1)}
            signal = "sell"
        else:
            return {"signal": "hold", "reason": "doji_candle"}

        return {
            "signal":      signal,
            "burst_ratio": round(candle_body / (atr_mean + 1e-10), 3),
            "atr_ratio":   round(current_atr / (atr_mean + 1e-10), 3),
            "rsi":         round(rsi_now, 1),
            "vol_confirm": vol_ok,
        }

    def __repr__(self):
        return (f"MomentumBurstStrategy(atr={self.atr_period}, "
                f"compression={self.compression_ratio}, "
                f"burst={self.burst_mult}x, vol={self.vol_mult}x)")
