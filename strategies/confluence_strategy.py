"""
confluence_strategy.py

Estrategia de alta confluencia SMC — los setups de mayor probabilidad.

Lógica (orden de prioridad):
  1. Liquidity Sweep → precio barre EQH/EQL (stop hunt)
  2. Retorno a zona FVG activo O Order Block activo
  3. BOS interno que confirma la dirección
  4. (Opcional) EMA trend alignment
  5. (Opcional) Precio en zona P/D correcta

Esta combinación es la que usa el trader SMC profesional:
  "espera que el precio barra stops, entre en un OB/FVG y confirme con BOS"

Cuanto más confluencia, mayor WR y mayor RR — pero menos trades.
Se puede configurar con más o menos filtros para ajustar frecuencia vs calidad.
"""

from indicators.market_structure import _detect_structure_level
from indicators.liquidity import detect_liquidity_sweeps, last_sweep
from indicators.fvg import get_active_fvgs, price_in_fvg
from indicators.order_blocks import get_active_order_blocks, price_in_order_block
from indicators.equal_highs_lows import detect_equal_highs_lows
from indicators.premium_discount import detect_premium_discount, price_zone
from indicators.ema_rsi import ema_trend_filter


class ConfluenceStrategy:
    """
    Estrategia de confluencia SMC de alta probabilidad.

    Parámetros
    ----------
    swing_window    : ventana de swing para estructura y EQH/EQL (por defecto 10)
    require_sweep   : exige liquidity sweep reciente (stop hunt) — recomendado True
    require_fvg_or_ob: exige FVG activo O OB activo en zona de entrada
    require_bos     : exige BOS que confirme el retorno — siempre True en SMC
    use_pd_filter   : solo entrar en Discount (longs) / Premium (shorts)
    use_ema_filter  : filtro EMA trend
    trading_style   : preset EMA/RSI
    eql_threshold   : sensibilidad EQH/EQL (fracción de ATR)
    min_sweep_bars  : bars mínimas tras el sweep antes de entrar (cooldown)
    """

    def __init__(
        self,
        swing_window=10,
        require_sweep=True,
        require_fvg_or_ob=True,
        require_bos=True,
        use_pd_filter=False,
        use_ema_filter=False,
        trading_style="swing",
        eql_threshold=0.5,
        min_sweep_bars=1,
    ):
        self.swing_window     = swing_window
        self.require_sweep    = require_sweep
        self.require_fvg_or_ob = require_fvg_or_ob
        self.require_bos      = require_bos
        self.use_pd_filter    = use_pd_filter
        self.use_ema_filter   = use_ema_filter
        self.trading_style    = trading_style
        self.eql_threshold    = eql_threshold
        self.min_sweep_bars   = min_sweep_bars

        # Compatibilidad FastBacktester
        self.require_fvg      = False
        self.use_choch_filter = False

    def generate_signal(self, data):
        """
        Genera señal de confluencia SMC.

        Devuelve dict: signal, reason, sweep, structure, fvg_hit, ob_hit
        """
        if data is None or len(data) < self.swing_window * 3:
            return {"signal": "hold", "reason": "Datos insuficientes"}

        current_price = float(data["close"].iloc[-1])
        n             = len(data)
        extra         = {}

        # ── 1. Estructura de mercado base ────────────────────────────────────
        struct = _detect_structure_level(data, self.swing_window)
        if struct is None:
            return {"signal": "hold", "reason": "Sin estructura detectada"}

        extra["structure"] = struct

        # ── 2. Liquidity Sweep ───────────────────────────────────────────────
        sweep = last_sweep(data, self.swing_window)

        if self.require_sweep:
            if sweep is None:
                return {**extra, "signal": "hold", "reason": "Sin liquidity sweep"}
            # El sweep debe ser reciente (dentro de min_sweep_bars)
            sweep_age = n - 1 - sweep.get("index", 0)
            if sweep_age > max(self.min_sweep_bars * 3, 15):
                return {**extra, "signal": "hold",
                        "reason": f"Sweep demasiado antiguo ({sweep_age} velas)"}

        sweep_type = sweep["type"] if sweep else None
        extra["sweep"] = sweep_type

        # ── 3. Determinar dirección candidata ────────────────────────────────
        # Bullish: sweep bajista (barrió lows) → rebote alcista
        # Bearish: sweep alcista (barrió highs) → caída bajista
        if sweep_type == "bullish":
            direction = "buy"
        elif sweep_type == "bearish":
            direction = "sell"
        else:
            # Sin sweep: usar dirección del BOS si está activo
            if struct["bos"] and struct["bos_direction"]:
                direction = "buy" if struct["bos_direction"] == "bullish" else "sell"
            else:
                return {**extra, "signal": "hold", "reason": "Sin dirección clara"}

        # ── 4. FVG o OB como zona de precio ─────────────────────────────────
        fvg_hit = False
        ob_hit  = False

        if self.require_fvg_or_ob:
            fvg_type   = "bullish" if direction == "buy" else "bearish"
            ob_type    = "bullish" if direction == "buy" else "bearish"
            active_fvgs = get_active_fvgs(data)
            active_obs  = get_active_order_blocks(data, self.swing_window)

            fvg_hit = price_in_fvg(current_price, active_fvgs, fvg_type)
            ob_hit  = price_in_order_block(current_price, active_obs, ob_type)

            if not fvg_hit and not ob_hit:
                return {**extra, "signal": "hold",
                        "reason": f"Precio fuera de FVG/OB {fvg_type}",
                        "fvg_hit": False, "ob_hit": False}

        extra["fvg_hit"] = fvg_hit
        extra["ob_hit"]  = ob_hit

        # ── 5. BOS de confirmación ───────────────────────────────────────────
        if self.require_bos:
            bos_dir = struct.get("bos_direction")
            expected = "bullish" if direction == "buy" else "bearish"
            if not struct["bos"] or bos_dir != expected:
                return {**extra, "signal": "hold",
                        "reason": f"Sin BOS {expected} de confirmación"}

        # ── 6. Filtro P/D ────────────────────────────────────────────────────
        if self.use_pd_filter:
            pd = detect_premium_discount(data, swing_window=self.swing_window)
            if pd:
                zone = price_zone(current_price, pd)
                extra["pd_zone"] = zone
                if direction == "buy"  and zone == "premium":
                    return {**extra, "signal": "hold",
                            "reason": "Long en zona Premium — esperar Discount"}
                if direction == "sell" and zone == "discount":
                    return {**extra, "signal": "hold",
                            "reason": "Short en zona Discount — esperar Premium"}

        # ── 7. Filtro EMA ────────────────────────────────────────────────────
        if self.use_ema_filter:
            ema_t = ema_trend_filter(data, style=self.trading_style)
            if direction == "buy"  and ema_t != "bullish":
                return {**extra, "signal": "hold", "reason": "EMA no confirma long"}
            if direction == "sell" and ema_t != "bearish":
                return {**extra, "signal": "hold", "reason": "EMA no confirma short"}

        # ── Señal confirmada ─────────────────────────────────────────────────
        parts = ["Sweep" if self.require_sweep else ""]
        if fvg_hit: parts.append("FVG")
        if ob_hit:  parts.append("OB")
        if self.require_bos: parts.append("BOS")
        if self.use_pd_filter: parts.append("P/D")
        reason = " + ".join(p for p in parts if p)

        return {**extra, "signal": direction, "reason": reason}
