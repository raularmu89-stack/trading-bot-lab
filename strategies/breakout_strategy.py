"""
breakout_strategy.py

Estrategia para BREAKOUTS (ruptura de rango con volumen + ATR spike).
Lógica: precio supera el rango consolidado + volumen > media + confirmación BOS.

Escenario ideal: breakout (ATR spike 1.3-2.5× + BOS reciente)
Entrada: cierre > max del rango (N velas) + vol > vol_mean * multiplier
Filtro falso breakout: exige 2 velas de cierre fuera del rango
"""

import numpy as np
from indicators.regime_detector import _atr, _ema


class BreakoutStrategy:
    """
    Breakout de rango consolidado con confirmación de volumen.

    Parámetros
    ----------
    swing_window       : compat FastBacktester
    consolidation_bars : velas de consolidación previa (rango)
    vol_multiplier     : volumen debe ser > media × este multiplicador
    confirm_bars       : velas de cierre fuera del rango para confirmar
    atr_min_mult       : ATR actual debe ser > ATR_media × este mínimo
    min_consolidation  : ancho mínimo del rango (% del precio)
    """

    def __init__(self,
                 swing_window: int = 5,
                 consolidation_bars: int = 30,
                 vol_multiplier: float = 1.4,
                 confirm_bars: int = 1,
                 atr_min_mult: float = 1.1,
                 min_consolidation_pct: float = 0.008):
        self.swing_window          = swing_window
        self.consolidation_bars    = consolidation_bars
        self.vol_multiplier        = vol_multiplier
        self.confirm_bars          = confirm_bars
        self.atr_min_mult          = atr_min_mult
        self.min_consolidation_pct = min_consolidation_pct

        # FastBacktester compat
        self.require_fvg      = False
        self.use_choch_filter = False

    def generate_signal(self, data) -> dict:
        n_needed = self.consolidation_bars + self.confirm_bars + 20
        if data is None or len(data) < n_needed:
            return {"signal": "hold", "reason": "datos insuficientes"}

        h   = data["high"].values
        l   = data["low"].values
        c   = data["close"].values
        vol = data["volume"].values
        n   = len(c)

        atr_arr = _atr(h, l, c, 14)

        price   = c[-1]
        cb      = self.consolidation_bars

        # ── Rango de consolidación (excluyendo velas recientes de confirm_bars) ──
        offset       = self.confirm_bars
        consol_h     = h[n - 1 - cb - offset: n - 1 - offset]
        consol_l     = l[n - 1 - cb - offset: n - 1 - offset]
        range_top    = consol_h.max()
        range_bot    = consol_l.min()
        mid          = (range_top + range_bot) / 2

        if mid <= 0:
            return {"signal": "hold", "reason": "precio inválido"}

        range_pct = (range_top - range_bot) / mid
        if range_pct < self.min_consolidation_pct:
            return {"signal": "hold", "reason": f"Consolidación demasiado estrecha ({range_pct*100:.2f}%)"}

        # ── ATR spike ─────────────────────────────────────────────────
        atr_mean = np.mean(atr_arr[max(0, n - 50): n])
        atr_now  = atr_arr[-1]
        atr_ratio = atr_now / atr_mean if atr_mean > 0 else 1.0
        if atr_ratio < self.atr_min_mult:
            return {"signal": "hold", "reason": f"Sin ATR spike ({atr_ratio:.2f}×)"}

        # ── Volumen ────────────────────────────────────────────────────
        vol_mean = np.mean(vol[max(0, n - 30): n - 1])
        vol_now  = vol[-1]
        high_vol = vol_now >= vol_mean * self.vol_multiplier

        if not high_vol:
            return {"signal": "hold", "reason": f"Volumen insuficiente ({vol_now:.0f} vs {vol_mean:.0f})"}

        # ── Confirmación: N velas cerradas fuera del rango ────────────
        confirm_candles = c[n - self.confirm_bars: n]
        bull_confirm = all(cc > range_top for cc in confirm_candles)
        bear_confirm = all(cc < range_bot for cc in confirm_candles)

        if bull_confirm:
            return {
                "signal": "buy",
                "reason": f"Breakout BUY — rompió {range_top:.4f}, vol:{vol_now:.0f} ({atr_ratio:.1f}×ATR)",
                "range_top": range_top, "range_bot": range_bot,
                "atr_ratio": atr_ratio, "vol_ratio": vol_now / vol_mean,
            }
        if bear_confirm:
            return {
                "signal": "sell",
                "reason": f"Breakout SELL — rompió {range_bot:.4f}, vol:{vol_now:.0f} ({atr_ratio:.1f}×ATR)",
                "range_top": range_top, "range_bot": range_bot,
                "atr_ratio": atr_ratio, "vol_ratio": vol_now / vol_mean,
            }

        return {
            "signal": "hold",
            "reason": "Sin confirmación de cierre fuera del rango",
            "range_top": range_top, "range_bot": range_bot,
        }
