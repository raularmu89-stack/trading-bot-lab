"""
trend_rider.py

Estrategia para TENDENCIAS FUERTES (ADX > 25).
Lógica: EMA fan + BOS en dirección de tendencia + pullback a EMA20.

Escenario ideal: strong_trend_bull / strong_trend_bear
Entrada: precio retrocede a EMA20, rebota y confirma BOS alcista/bajista
Filtros: ADX > umbral, EMA20 > EMA50 > EMA200 (fan alcista) o inverso
"""

import numpy as np
from indicators.regime_detector import _ema, _adx, _atr


class TrendRiderStrategy:
    """
    Ride the trend — entra en pullbacks a EMA20 dentro de tendencia fuerte.

    Parámetros
    ----------
    swing_window      : ventana pivot para BOS (FastBacktester compat)
    adx_threshold     : ADX mínimo para considerar tendencia (default 22)
    ema_fast          : EMA rápida (20)
    ema_mid           : EMA media (50)
    ema_slow          : EMA lenta (200)
    pullback_pct      : max distancia precio/EMA20 para considerar pullback (0.5%)
    require_ema_fan   : exige EMA20>EMA50>EMA200 (bull) o inverso (bear)
    require_bos       : exige BOS en dirección de tendencia
    """

    def __init__(self,
                 swing_window: int = 5,
                 adx_threshold: float = 22.0,
                 ema_fast: int = 20,
                 ema_mid: int = 50,
                 ema_slow: int = 200,
                 pullback_pct: float = 0.008,
                 require_ema_fan: bool = True,
                 require_bos: bool = True):
        self.swing_window   = swing_window
        self.adx_threshold  = adx_threshold
        self.ema_fast       = ema_fast
        self.ema_mid        = ema_mid
        self.ema_slow       = ema_slow
        self.pullback_pct   = pullback_pct
        self.require_ema_fan = require_ema_fan
        self.require_bos    = require_bos

        # FastBacktester compat
        self.require_fvg    = False
        self.use_choch_filter = False

    def generate_signal(self, data) -> dict:
        if data is None or len(data) < max(self.ema_slow, 30):
            return {"signal": "hold", "reason": "datos insuficientes"}

        h = data["high"].values
        l = data["low"].values
        c = data["close"].values
        n = len(c)

        ema_f = _ema(c, self.ema_fast)
        ema_m = _ema(c, self.ema_mid)
        ema_s = _ema(c, self.ema_slow)
        adx_d = _adx(h, l, c, 14)

        price    = c[-1]
        adx_val  = adx_d["adx"][-1]
        plus_di  = adx_d["plus_di"][-1]
        minus_di = adx_d["minus_di"][-1]
        ef, em, es = ema_f[-1], ema_m[-1], ema_s[-1]

        # ── Filtro ADX ────────────────────────────────────────────────
        if adx_val < self.adx_threshold:
            return {"signal": "hold", "reason": f"ADX {adx_val:.1f} < umbral",
                    "adx": adx_val}

        bullish_trend = plus_di > minus_di
        bearish_trend = minus_di > plus_di

        # ── Filtro EMA fan ────────────────────────────────────────────
        if self.require_ema_fan:
            bull_fan = ef > em > es
            bear_fan = ef < em < es
            if bullish_trend and not bull_fan:
                return {"signal": "hold", "reason": "Sin EMA fan alcista", "adx": adx_val}
            if bearish_trend and not bear_fan:
                return {"signal": "hold", "reason": "Sin EMA fan bajista", "adx": adx_val}

        # ── Pullback a EMA fast ───────────────────────────────────────
        dist_pct = abs(price - ef) / ef if ef > 0 else 1.0
        near_ema = dist_pct <= self.pullback_pct

        # ── BOS en dirección de tendencia ─────────────────────────────
        bos_bull = bos_bear = False
        if self.require_bos:
            w = self.swing_window
            if n > w * 2:
                recent_high = h[max(0, n-1-w*2): n-1].max()
                recent_low  = l[max(0, n-1-w*2): n-1].min()
                bos_bull = c[-1] > recent_high
                bos_bear = c[-1] < recent_low
        else:
            bos_bull = bullish_trend
            bos_bear = bearish_trend

        # ── Señal final ───────────────────────────────────────────────
        if bullish_trend and near_ema and bos_bull:
            return {
                "signal": "buy",
                "reason": f"TrendRider BUY — ADX:{adx_val:.1f} pullback EMA{self.ema_fast}",
                "adx": adx_val, "ema_fast": ef, "dist_pct": dist_pct,
            }
        if bearish_trend and near_ema and bos_bear:
            return {
                "signal": "sell",
                "reason": f"TrendRider SELL — ADX:{adx_val:.1f} pullback EMA{self.ema_fast}",
                "adx": adx_val, "ema_fast": ef, "dist_pct": dist_pct,
            }

        return {
            "signal": "hold",
            "reason": f"Sin pullback (dist={dist_pct*100:.2f}%) o sin BOS",
            "adx": adx_val,
        }
