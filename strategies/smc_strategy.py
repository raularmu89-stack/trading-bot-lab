"""
smc_strategy.py

Estrategia SMC completa — conversión del indicador Pine 'raul smc'.

Combina todos los indicadores:
  - Estructura dual (Internal + Swing BOS/CHoCH)
  - Order Blocks (internal + swing, con filtro ATR de volatilidad)
  - Fair Value Gaps (con filtro de umbral automático)
  - Equal Highs/Lows (zonas de liquidez)
  - EMA + RSI (con presets Scalping / Intraday / Swing)
  - Premium/Discount zones

Jerarquía de señal (de mayor a menor prioridad):
  1. Estructura de swing + CHoCH filter
  2. Filtro EMA trend (precio vs EMA lenta)
  3. Filtro Premium/Discount (entrar en Discount para long, Premium para short)
  4. Confirmación: FVG activo del mismo tipo
  5. Confirmación: precio en Order Block activo
  6. Confirmación: Liquidity Sweep reciente
"""

from indicators.market_structure import detect_market_structure, detect_dual_structure
from indicators.fvg              import detect_fvg, get_active_fvgs, price_in_fvg
from indicators.order_blocks     import get_active_order_blocks, price_in_order_block, detect_dual_order_blocks
from indicators.liquidity        import last_sweep
from indicators.ema_rsi          import compute_ema_rsi, ema_trend_filter, TRADING_STYLES
from indicators.premium_discount import detect_premium_discount, price_zone
from indicators.equal_highs_lows import detect_equal_highs_lows


class SMCStrategy:
    """
    Estrategia SMC configurable con todos los filtros del indicador Pine.

    Parametros
    ----------
    swing_window      : ventana swing principal (Pine: swingsLengthInput=50)
    internal_window   : ventana estructura interna (Pine: 5, fijo)
    require_fvg       : exige FVG activo del mismo tipo (Pine: showFairValueGapsInput)
    use_choch_filter  : espera confirmación tras CHoCH (Pine: comportamiento implícito)
    require_ob        : exige que el precio esté en OB activo
    require_sweep     : exige Liquidity Sweep reciente
    use_dual_structure: usa estructura dual (swing + internal) si True
    trading_style     : "scalping" | "intraday" | "swing" (presets EMA/RSI)
    use_ema_filter    : filtra señales según EMA trend (precio vs EMA lenta)
    use_pd_filter     : filtra según zona Premium/Discount
    use_fvg_threshold : auto-threshold en FVG (Pine: fairValueGapsThresholdInput)
    ob_mitigation     : "close" | "highlow" para OBs (Pine: orderBlockMitigationInput)
    """

    def __init__(
        self,
        swing_window=5,            # legacy default 5; Pine usa 50
        internal_window=5,
        require_fvg=False,
        use_choch_filter=True,
        require_ob=False,
        require_sweep=False,
        use_dual_structure=False,
        trading_style="scalping",
        use_ema_filter=False,
        use_pd_filter=False,
        use_fvg_threshold=False,
        ob_mitigation="highlow",
    ):
        self.swing_window      = swing_window
        self.internal_window   = internal_window
        self.require_fvg       = require_fvg
        self.use_choch_filter  = use_choch_filter
        self.require_ob        = require_ob
        self.require_sweep     = require_sweep
        self.use_dual_structure = use_dual_structure
        self.trading_style     = trading_style
        self.use_ema_filter    = use_ema_filter
        self.use_pd_filter     = use_pd_filter
        self.use_fvg_threshold = use_fvg_threshold
        self.ob_mitigation     = ob_mitigation

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _get_structure(self, data):
        """Devuelve (trend, bos, choch) según el modo de estructura."""
        if self.use_dual_structure:
            dual = detect_dual_structure(
                data,
                internal_window=self.internal_window,
                swing_window=self.swing_window,
            )
            if dual is None:
                return None, False, False
            # Prioridad: nivel swing
            swing = dual["swing"]
            if swing is None:
                return None, False, False
            return swing["trend"], swing["bos"], swing["choch"]
        else:
            structure = detect_market_structure(data, window=self.swing_window)
            if structure is None:
                return None, False, False
            return structure["trend"], structure["bos"], structure["choch"]

    def _check_ema_filter(self, data, direction):
        """Retorna False si el filtro EMA rechaza la señal."""
        if not self.use_ema_filter:
            return True
        ema_trend = ema_trend_filter(data, style=self.trading_style)
        if direction == "buy"  and ema_trend == "bullish":
            return True
        if direction == "sell" and ema_trend == "bearish":
            return True
        return False

    def _check_pd_filter(self, data, direction):
        """Retorna False si el precio está en zona desfavorable."""
        if not self.use_pd_filter:
            return True
        pd = detect_premium_discount(data, swing_window=self.swing_window)
        if pd is None:
            return True
        zone = pd["zone"]
        if direction == "buy"  and zone in ("discount", "equilibrium"):
            return True
        if direction == "sell" and zone in ("premium", "equilibrium"):
            return True
        return False

    # ─── Main signal ─────────────────────────────────────────────────────────

    def generate_signal(self, data):
        """
        Genera una señal de trading basada en la estructura SMC.

        Devuelve dict:
          signal    : "buy" | "sell" | "hold"
          reason    : texto explicativo
          structure : dict de estructura (si disponible)
          fvg_count : número de FVGs detectados
          [ema]     : resultado EMA/RSI si use_ema_filter=True
          [pd_zone] : zona Premium/Discount si use_pd_filter=True
        """
        if data is None or len(data) < 5:
            return {"signal": "hold", "reason": "Not enough data"}

        trend, bos, choch = self._get_structure(data)

        if trend is None:
            return {"signal": "hold", "reason": "No market structure detected"}

        if self.use_choch_filter and choch:
            return {
                "signal": "hold",
                "reason": "CHoCH detected, waiting confirmation",
            }

        current_price = float(data["close"].iloc[-1])
        fvgs = detect_fvg(data, auto_threshold=self.use_fvg_threshold)

        extra = {
            "structure": {"trend": trend, "bos": bos, "choch": choch},
            "fvg_count": len(fvgs),
        }

        # ── EMA/RSI info (siempre calculado si use_ema_filter, o para info)
        if self.use_ema_filter:
            ema_result = compute_ema_rsi(data, style=self.trading_style)
            extra["ema"] = {
                "signal":       ema_result["signal"],
                "cross_up":     ema_result["cross_up"],
                "cross_down":   ema_result["cross_down"],
                "rsi_ob":       ema_result["rsi_overbought"],
                "rsi_os":       ema_result["rsi_oversold"],
            }

        if self.use_pd_filter:
            pd = detect_premium_discount(data, swing_window=self.swing_window)
            extra["pd_zone"] = pd["zone"] if pd else "unknown"

        # ── Determinar dirección candidata ───────────────────────────────────
        if trend == "bullish" and bos:
            direction = "buy"
        elif trend == "bearish" and bos:
            direction = "sell"
        else:
            return {**extra, "signal": "hold", "reason": "No clear BOS"}

        # ── Filtros opcionales ───────────────────────────────────────────────
        if not self._check_ema_filter(data, direction):
            return {**extra, "signal": "hold",
                    "reason": f"EMA filter: trend not {direction}ish"}

        if not self._check_pd_filter(data, direction):
            return {**extra, "signal": "hold",
                    "reason": f"PD filter: bad zone for {direction}"}

        if self.require_fvg:
            fvg_type = "bullish" if direction == "buy" else "bearish"
            active_fvgs = get_active_fvgs(data, self.use_fvg_threshold)
            if not any(f["type"] == fvg_type for f in active_fvgs):
                return {**extra, "signal": "hold",
                        "reason": f"No active {fvg_type} FVG"}

        if self.require_ob:
            ob_type     = "bullish" if direction == "buy" else "bearish"
            active_obs  = get_active_order_blocks(
                data, self.swing_window, self.ob_mitigation
            )
            if not price_in_order_block(current_price, active_obs, ob_type):
                return {**extra, "signal": "hold",
                        "reason": f"Price not in {ob_type} OB"}

        if self.require_sweep:
            sweep = last_sweep(data, self.swing_window)
            sweep_type = "bullish" if direction == "buy" else "bearish"
            if sweep is None or sweep["type"] != sweep_type:
                return {**extra, "signal": "hold",
                        "reason": f"No {sweep_type} liquidity sweep"}

        # ── Señal confirmada ─────────────────────────────────────────────────
        reason_parts = [f"{'Bullish' if direction=='buy' else 'Bearish'} BOS"]
        if self.require_ob:      reason_parts.append("OB")
        if self.require_fvg:     reason_parts.append("FVG")
        if self.require_sweep:   reason_parts.append("Sweep")
        if self.use_ema_filter:  reason_parts.append("EMA")
        if self.use_pd_filter:   reason_parts.append("P/D")

        return {**extra, "signal": direction, "reason": " + ".join(reason_parts)}
