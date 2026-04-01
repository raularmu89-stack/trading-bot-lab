"""
ob_rejection.py

Estrategia para PULLBACKS a ORDER BLOCKS en tendencia (SMC puro).
Lógica: detecta OB con Pine ATR filter, espera precio retroceda al OB,
        confirma rechazo con vela inversora + FVG o BOS local.

Escenario ideal: weak_trend_bull / weak_trend_bear
Entrada: precio en zona OB + vela de rechazo + BOS/FVG local
"""

import numpy as np
from indicators.order_blocks import detect_dual_order_blocks
from indicators.fvg import detect_fvg
from indicators.regime_detector import _atr, _ema, _rsi


class OBRejectionStrategy:
    """
    Pullback a Order Block con confirmación de rechazo.

    Parámetros
    ----------
    swing_window      : ventana para detección de OB (FastBacktester compat)
    ob_window         : ventana interna para OB (puede diferir)
    fvg_confirm       : exige FVG de confirmación en la zona OB
    bos_confirm       : exige BOS local tras el rechazo
    rejection_pct     : la vela de rechazo debe cerrar > X% hacia afuera del OB
    rsi_period        : RSI para filtro de sobrextensión
    rsi_bull_max      : RSI máximo en compra (evitar entrar en sobrecompra)
    rsi_bear_min      : RSI mínimo en venta (evitar entrar en sobreventa)
    max_ob_age        : máximo de velas desde que se formó el OB (frescura)
    """

    def __init__(self,
                 swing_window: int = 5,
                 ob_window: int = 5,
                 fvg_confirm: bool = False,
                 bos_confirm: bool = True,
                 rejection_pct: float = 0.003,
                 rsi_period: int = 14,
                 rsi_bull_max: float = 65.0,
                 rsi_bear_min: float = 35.0,
                 max_ob_age: int = 100):
        self.swing_window  = swing_window
        self.ob_window     = ob_window
        self.fvg_confirm   = fvg_confirm
        self.bos_confirm   = bos_confirm
        self.rejection_pct = rejection_pct
        self.rsi_period    = rsi_period
        self.rsi_bull_max  = rsi_bull_max
        self.rsi_bear_min  = rsi_bear_min
        self.max_ob_age    = max_ob_age

        # FastBacktester compat
        self.require_fvg      = fvg_confirm
        self.use_choch_filter = False

    def generate_signal(self, data) -> dict:
        min_len = max(self.ob_window * 4, 50)
        if data is None or len(data) < min_len:
            return {"signal": "hold", "reason": "datos insuficientes"}

        h = data["high"].values
        l = data["low"].values
        c = data["close"].values
        o = data["open"].values
        n = len(c)

        rsi_arr  = _rsi(c, self.rsi_period)
        atr_arr  = _atr(h, l, c, 14)
        price    = c[-1]
        rsi_val  = rsi_arr[-1]
        atr_val  = atr_arr[-1]

        # ── Obtener OBs activos ───────────────────────────────────────
        try:
            obs = detect_dual_order_blocks(data, internal_window=self.ob_window,
                                           swing_window=self.ob_window * 5)
            bullish_obs = [ob for ob in obs.get("bullish", [])
                           if not ob.get("mitigated", False)
                           and (n - 1 - ob.get("ob_index", 0)) <= self.max_ob_age]
            bearish_obs = [ob for ob in obs.get("bearish", [])
                           if not ob.get("mitigated", False)
                           and (n - 1 - ob.get("ob_index", 0)) <= self.max_ob_age]
        except Exception:
            return {"signal": "hold", "reason": "Error en detección de OB"}

        # ── FVG activos ────────────────────────────────────────────────
        fvg_bull_active = fvg_bear_active = False
        if self.fvg_confirm:
            try:
                fvgs = detect_fvg(data)
                fvg_bull_active = any(
                    not f.get("mitigated", True)
                    for f in fvgs if f.get("type") == "bullish"
                )
                fvg_bear_active = any(
                    not f.get("mitigated", True)
                    for f in fvgs if f.get("type") == "bearish"
                )
            except Exception:
                pass

        # ── BOS local (últimas swing_window velas) ────────────────────
        w = self.swing_window
        bos_bull = bos_bear = False
        if self.bos_confirm and n > w * 2:
            bos_bull = c[-1] > h[max(0, n-1-w*2): n-1].max()
            bos_bear = c[-1] < l[max(0, n-1-w*2): n-1].min()

        # ── Vela de rechazo ───────────────────────────────────────────
        # Bull rejection: vela anterior bajista + cierre actual por encima
        prev_bearish = o[-2] > c[-2]
        prev_bullish = o[-2] < c[-2]
        rejection_up   = prev_bearish and (c[-1] - c[-2]) / c[-2] >= self.rejection_pct
        rejection_down = prev_bullish and (c[-2] - c[-1]) / c[-2] >= self.rejection_pct

        # ── Precio en zona OB ─────────────────────────────────────────
        in_bull_ob = any(ob["low"] <= price <= ob["high"] for ob in bullish_obs)
        in_bear_ob = any(ob["low"] <= price <= ob["high"] for ob in bearish_obs)

        # ── Señal buy ─────────────────────────────────────────────────
        if in_bull_ob and rejection_up and rsi_val <= self.rsi_bull_max:
            bos_ok = (bos_bull or not self.bos_confirm)
            fvg_ok = (fvg_bull_active or not self.fvg_confirm)
            if bos_ok and fvg_ok:
                return {
                    "signal": "buy",
                    "reason": f"OBRejection BUY — en OB alcista, rechazo, RSI:{rsi_val:.1f}",
                    "ob_count": len(bullish_obs), "rsi": rsi_val, "atr": atr_val,
                }

        # ── Señal sell ────────────────────────────────────────────────
        if in_bear_ob and rejection_down and rsi_val >= self.rsi_bear_min:
            bos_ok = (bos_bear or not self.bos_confirm)
            fvg_ok = (fvg_bear_active or not self.fvg_confirm)
            if bos_ok and fvg_ok:
                return {
                    "signal": "sell",
                    "reason": f"OBRejection SELL — en OB bajista, rechazo, RSI:{rsi_val:.1f}",
                    "ob_count": len(bearish_obs), "rsi": rsi_val, "atr": atr_val,
                }

        reason_parts = []
        if not in_bull_ob and not in_bear_ob:
            reason_parts.append("precio fuera de OBs")
        if not rejection_up and not rejection_down:
            reason_parts.append("sin vela de rechazo")
        return {
            "signal": "hold",
            "reason": "; ".join(reason_parts) or "condiciones no cumplidas",
            "bull_obs": len(bullish_obs), "bear_obs": len(bearish_obs),
        }
