"""
volatility_filter.py

Estrategia para ALTA VOLATILIDAD (ATR > 2.5× media, eventos de noticias).
En modo normal: devuelve hold para proteger capital.
Modo exhaustion: detecta spike + vela de reversión → entrada contraria.

Escenario ideal: high_volatility
"""

import numpy as np
from indicators.regime_detector import _atr, _ema


class VolatilityFilterStrategy:
    """
    Gestiona periodos de alta volatilidad.

    Parámetros
    ----------
    swing_window            : compat FastBacktester
    atr_high_mult           : ATR > media × este valor → alto riesgo (2.5)
    atr_extreme_mult        : ATR > media × este → evento extremo, siempre hold (4.0)
    allow_exhaustion_entry  : permitir entrada en agotamiento del spike
    exhaustion_lookback     : velas previas para detectar spike de agotamiento
    size_reduction_factor   : factor para reducir sizing (leído por ScenarioRouter)
    """

    def __init__(self,
                 swing_window: int = 5,
                 atr_high_mult: float = 2.5,
                 atr_extreme_mult: float = 4.0,
                 allow_exhaustion_entry: bool = False,
                 exhaustion_lookback: int = 3,
                 size_reduction_factor: float = 0.5):
        self.swing_window           = swing_window
        self.atr_high_mult          = atr_high_mult
        self.atr_extreme_mult       = atr_extreme_mult
        self.allow_exhaustion_entry = allow_exhaustion_entry
        self.exhaustion_lookback    = exhaustion_lookback
        self.size_reduction_factor  = size_reduction_factor

        # FastBacktester compat
        self.require_fvg      = False
        self.use_choch_filter = False

    def generate_signal(self, data) -> dict:
        if data is None or len(data) < 30:
            return {"signal": "hold", "reason": "datos insuficientes"}

        h = data["high"].values
        l = data["low"].values
        c = data["close"].values
        n = len(c)

        atr_arr  = _atr(h, l, c, 14)
        atr_now  = atr_arr[-1]
        atr_mean = np.mean(atr_arr[max(0, n - 50): n])
        atr_ratio = atr_now / atr_mean if atr_mean > 0 else 1.0

        # Evento extremo — nunca operar
        if atr_ratio >= self.atr_extreme_mult:
            return {
                "signal": "hold",
                "reason": f"Evento extremo — ATR {atr_ratio:.1f}× media",
                "atr_ratio": atr_ratio,
                "size_reduction": 0.0,
            }

        # Alta volatilidad sin exhaustion
        if not self.allow_exhaustion_entry:
            return {
                "signal": "hold",
                "reason": f"Alta volatilidad — ATR {atr_ratio:.1f}× media",
                "atr_ratio": atr_ratio,
                "size_reduction": self.size_reduction_factor,
            }

        # Modo exhaustion: spike + vela doji/envolvente contraria
        lb = self.exhaustion_lookback
        if n < lb + 2:
            return {"signal": "hold", "reason": "datos insuficientes para exhaustion"}

        # Spike: una de las últimas lb velas tuvo un rango > 2× ATR_mean
        spike_bars = h[n-lb:n] - l[n-lb:n]
        had_spike  = any(rb > atr_mean * 2.0 for rb in spike_bars)
        if not had_spike:
            return {"signal": "hold", "reason": "sin spike previo detectable"}

        # Dirección del spike (última vela grande)
        spike_idx = int(np.argmax(spike_bars)) + (n - lb)
        spike_bull = c[spike_idx] > c[spike_idx - 1]

        # Vela de agotamiento actual: cuerpo pequeño (< 25% del rango)
        body  = abs(c[-1] - data["open"].values[-1])
        range_ = h[-1] - l[-1]
        is_doji = (range_ > 0) and (body / range_ < 0.25)

        if is_doji:
            # Contraría al spike
            if spike_bull:
                return {
                    "signal": "sell",
                    "reason": f"Exhaustion SELL — spike alcista + doji, ATR:{atr_ratio:.1f}×",
                    "atr_ratio": atr_ratio,
                    "size_reduction": self.size_reduction_factor,
                }
            else:
                return {
                    "signal": "buy",
                    "reason": f"Exhaustion BUY — spike bajista + doji, ATR:{atr_ratio:.1f}×",
                    "atr_ratio": atr_ratio,
                    "size_reduction": self.size_reduction_factor,
                }

        return {
            "signal": "hold",
            "reason": f"Alta vol sin agotamiento (ATR {atr_ratio:.1f}×)",
            "atr_ratio": atr_ratio,
            "size_reduction": self.size_reduction_factor,
        }
