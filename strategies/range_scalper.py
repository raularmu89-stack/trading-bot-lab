"""
range_scalper.py

Estrategia para MERCADOS LATERALES (ADX < 18, ATR bajo).
Lógica: detecta rango (high/low de N velas), compra en soporte, vende en resistencia.

Escenario ideal: ranging
Entrada bull: precio toca zona soporte (< 15% inferior del rango) + RSI < 45
Entrada bear: precio toca zona resistencia (> 85% superior del rango) + RSI > 55
"""

import numpy as np
from indicators.regime_detector import _rsi, _atr, _ema


class RangeScalperStrategy:
    """
    Scalper de rango — compra en soporte, vende en resistencia.

    Parámetros
    ----------
    swing_window    : ventana para detectar rango (pivot highs/lows)
    range_window    : velas para calcular el rango actual
    entry_zone_pct  : zona de entrada desde extremo (0.15 = 15% del rango)
    rsi_period      : período RSI
    rsi_buy_max     : RSI máximo para compra (45)
    rsi_sell_min    : RSI mínimo para venta (55)
    min_range_pct   : rango mínimo como % del precio (evita rangos demasiado estrechos)
    max_range_pct   : rango máximo (si > esto no es rango, es tendencia)
    """

    def __init__(self,
                 swing_window: int = 5,
                 range_window: int = 50,
                 entry_zone_pct: float = 0.18,
                 rsi_period: int = 14,
                 rsi_buy_max: float = 48.0,
                 rsi_sell_min: float = 52.0,
                 min_range_pct: float = 0.01,
                 max_range_pct: float = 0.12):
        self.swing_window   = swing_window
        self.range_window   = range_window
        self.entry_zone_pct = entry_zone_pct
        self.rsi_period     = rsi_period
        self.rsi_buy_max    = rsi_buy_max
        self.rsi_sell_min   = rsi_sell_min
        self.min_range_pct  = min_range_pct
        self.max_range_pct  = max_range_pct

        # FastBacktester compat
        self.require_fvg      = False
        self.use_choch_filter = False

    def generate_signal(self, data) -> dict:
        if data is None or len(data) < self.range_window + self.rsi_period + 5:
            return {"signal": "hold", "reason": "datos insuficientes"}

        h = data["high"].values
        l = data["low"].values
        c = data["close"].values
        n = len(c)

        rsi_arr = _rsi(c, self.rsi_period)
        price   = c[-1]
        rsi_val = rsi_arr[-1]

        # ── Detectar rango actual ──────────────────────────────────────
        w   = self.range_window
        top = h[n - w: n].max()
        bot = l[n - w: n].min()
        mid = (top + bot) / 2.0

        if mid <= 0:
            return {"signal": "hold", "reason": "precio inválido"}

        range_pct = (top - bot) / mid

        if range_pct < self.min_range_pct:
            return {"signal": "hold", "reason": f"Rango demasiado estrecho ({range_pct*100:.2f}%)"}
        if range_pct > self.max_range_pct:
            return {"signal": "hold", "reason": f"Rango demasiado amplio (posible tendencia)"}

        # ── Zona de soporte y resistencia ─────────────────────────────
        zone_size   = (top - bot) * self.entry_zone_pct
        support_top = bot + zone_size       # zona inferior del rango
        resist_bot  = top - zone_size       # zona superior del rango

        in_support    = price <= support_top
        in_resistance = price >= resist_bot

        # ── Señal ─────────────────────────────────────────────────────
        if in_support and rsi_val <= self.rsi_buy_max:
            return {
                "signal": "buy",
                "reason": f"RangeScalper BUY — soporte {bot:.4f}–{support_top:.4f}, RSI:{rsi_val:.1f}",
                "range_top": top, "range_bot": bot, "range_pct": range_pct, "rsi": rsi_val,
            }
        if in_resistance and rsi_val >= self.rsi_sell_min:
            return {
                "signal": "sell",
                "reason": f"RangeScalper SELL — resistencia {resist_bot:.4f}–{top:.4f}, RSI:{rsi_val:.1f}",
                "range_top": top, "range_bot": bot, "range_pct": range_pct, "rsi": rsi_val,
            }

        return {
            "signal": "hold",
            "reason": f"Precio en zona media del rango (RSI:{rsi_val:.1f})",
            "range_top": top, "range_bot": bot,
        }
