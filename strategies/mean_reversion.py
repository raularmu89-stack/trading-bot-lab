"""
mean_reversion.py

Estrategia para SOBRE-EXTENSIÓN del precio (mean reversion).
Lógica: RSI extremo + precio en zona premium/discount + FVG de retroceso.

Escenario ideal: mean_reversion_bull / mean_reversion_bear
Entrada bull: RSI < 30 + precio en zona discount + FVG alcista activo
Entrada bear: RSI > 70 + precio en zona premium + FVG bajista activo
"""

import numpy as np
from indicators.premium_discount import detect_premium_discount
from indicators.fvg import detect_fvg
from indicators.regime_detector import _rsi, _atr, _ema


class MeanReversionStrategy:
    """
    Mean reversion — entra cuando el precio está sobre-extendido.

    Parámetros
    ----------
    swing_window      : ventana swing para P/D zones (FastBacktester compat)
    pd_window         : ventana para premium/discount
    rsi_period        : período RSI
    rsi_oversold      : umbral RSI sobreventa (compra)
    rsi_overbought    : umbral RSI sobrecompra (venta)
    require_fvg       : exige FVG activo en zona
    require_pd_zone   : exige que precio esté en zona premium/discount
    ema_period        : EMA para confirmar dirección de reversión
    max_adx           : ADX máximo (si hay tendencia fuerte, no hacer mean rev)
    """

    def __init__(self,
                 swing_window: int = 5,
                 pd_window: int = 50,
                 rsi_period: int = 14,
                 rsi_oversold: float = 32.0,
                 rsi_overbought: float = 68.0,
                 require_fvg: bool = False,
                 require_pd_zone: bool = True,
                 ema_period: int = 50,
                 max_adx: float = 28.0):
        self.swing_window    = swing_window
        self.pd_window       = pd_window
        self.rsi_period      = rsi_period
        self.rsi_oversold    = rsi_oversold
        self.rsi_overbought  = rsi_overbought
        self.require_fvg     = require_fvg
        self.require_pd_zone = require_pd_zone
        self.ema_period      = ema_period
        self.max_adx         = max_adx

        # FastBacktester compat
        self.use_choch_filter = False

    def generate_signal(self, data) -> dict:
        min_len = max(self.pd_window, self.ema_period, 30) + 5
        if data is None or len(data) < min_len:
            return {"signal": "hold", "reason": "datos insuficientes"}

        h = data["high"].values
        l = data["low"].values
        c = data["close"].values
        n = len(c)

        rsi_arr = _rsi(c, self.rsi_period)
        ema_arr = _ema(c, self.ema_period)
        atr_arr = _atr(h, l, c, 14)

        price   = c[-1]
        rsi_val = rsi_arr[-1]
        ema_val = ema_arr[-1]
        atr_val = atr_arr[-1]

        # ── Premium/Discount zones ────────────────────────────────────
        pd_zone = "unknown"
        if self.require_pd_zone:
            try:
                pd = detect_premium_discount(data, swing_window=self.pd_window)
                pd_zone = pd.get("zone", "unknown")
            except Exception:
                pd_zone = "unknown"

        in_discount = pd_zone in ("discount", "deep_discount") or not self.require_pd_zone
        in_premium  = pd_zone in ("premium",  "deep_premium")  or not self.require_pd_zone

        # ── FVG activo ─────────────────────────────────────────────────
        fvg_bull = fvg_bear = False
        if self.require_fvg:
            try:
                fvgs = detect_fvg(data)
                fvg_bull = any(not f.get("mitigated", True) and f.get("type") == "bullish"
                               for f in fvgs)
                fvg_bear = any(not f.get("mitigated", True) and f.get("type") == "bearish"
                               for f in fvgs)
            except Exception:
                pass
        else:
            fvg_bull = fvg_bear = True

        # ── RSI divergencia: RSI sube mientras precio baja (bull div) ─
        rsi_recovering = len(rsi_arr) > 3 and rsi_arr[-1] > rsi_arr[-2]
        rsi_declining  = len(rsi_arr) > 3 and rsi_arr[-1] < rsi_arr[-2]

        # ── Señal buy (reversal alcista) ──────────────────────────────
        if rsi_val <= self.rsi_oversold and in_discount and fvg_bull and rsi_recovering:
            return {
                "signal": "buy",
                "reason": f"MeanRev BUY — RSI:{rsi_val:.1f} en discount, FVG activo",
                "rsi": rsi_val, "pd_zone": pd_zone, "ema": ema_val,
            }

        # ── Señal sell (reversal bajista) ─────────────────────────────
        if rsi_val >= self.rsi_overbought and in_premium and fvg_bear and rsi_declining:
            return {
                "signal": "sell",
                "reason": f"MeanRev SELL — RSI:{rsi_val:.1f} en premium, FVG activo",
                "rsi": rsi_val, "pd_zone": pd_zone, "ema": ema_val,
            }

        return {
            "signal": "hold",
            "reason": f"Sin condiciones (RSI:{rsi_val:.1f}, zona:{pd_zone})",
            "rsi": rsi_val, "pd_zone": pd_zone,
        }
