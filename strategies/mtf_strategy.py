"""
mtf_strategy.py

Estrategia Multi-TimeFrame (MTF) SMC profesional.

Lógica:
  1. Timeframe ALTO (4h / 1d):  detecta la tendencia y el BOS de swing principal
  2. Timeframe BAJO (1h / 15m): espera CHoCH en la dirección opuesta (pullback)
     seguido de BOS interno que confirma retorno a la tendencia del TF alto

  Solo se opera cuando ambos TF están alineados:
    - TF alto: tendencia bullish  → TF bajo: BOS alcista tras pullback (CHoCH bajista)
    - TF alto: tendencia bearish  → TF bajo: BOS bajista tras pullback (CHoCH alcista)

Filtros adicionales opcionales:
  - EMA slow del TF alto como filtro de tendencia
  - Precio en zona Discount (longs) o Premium (shorts) del TF alto
  - FVG o OB del TF bajo como zona de entrada precisa

Pine equivalent:
  request.security(syminfo.tickerid, higherTF, ...) → resample
  Internal structure (5)  → low TF BOS/CHoCH
  Swing structure (50)    → high TF trend

Uso:
    from strategies.mtf_strategy import MTFStrategy
    strategy = MTFStrategy(high_tf="4h", low_tf_window=5, high_tf_window=50)
    signal = strategy.generate_signal(df_1h)
"""

import numpy as np
from indicators.market_structure import _detect_structure_level, _find_pivots
from indicators.resampler import resample
from indicators.fvg import get_active_fvgs
from indicators.order_blocks import get_active_order_blocks, price_in_order_block
from indicators.premium_discount import detect_premium_discount, price_zone
from indicators.ema_rsi import compute_ema_rsi, ema_trend_filter


class MTFStrategy:
    """
    Estrategia SMC multi-timeframe.

    Parámetros
    ----------
    high_tf          : timeframe alto para la tendencia ("4h", "1d")
    high_tf_window   : swing_window en el TF alto (por defecto 10)
    low_tf_window    : swing_window en el TF bajo (por defecto 5)
    require_pullback : exige CHoCH en TF bajo antes del BOS de entrada
    use_pd_filter    : solo entrar en Discount (longs) o Premium (shorts) del TF alto
    use_fvg_entry    : refinar entrada con FVG activo del TF bajo
    use_ob_entry     : refinar entrada con OB activo del TF bajo
    trading_style    : preset EMA/RSI ("scalping"|"intraday"|"swing")
    use_ema_filter   : filtro EMA del TF bajo
    """

    def __init__(
        self,
        high_tf="4h",
        high_tf_window=10,
        low_tf_window=5,
        require_pullback=True,
        use_pd_filter=True,
        use_fvg_entry=False,
        use_ob_entry=False,
        trading_style="swing",
        use_ema_filter=False,
    ):
        self.high_tf         = high_tf
        self.high_tf_window  = high_tf_window
        self.low_tf_window   = low_tf_window
        self.require_pullback = require_pullback
        self.use_pd_filter   = use_pd_filter
        self.use_fvg_entry   = use_fvg_entry
        self.use_ob_entry    = use_ob_entry
        self.trading_style   = trading_style
        self.use_ema_filter  = use_ema_filter

        # Para compatibilidad con FastBacktester (lee estos atributos)
        self.swing_window    = low_tf_window
        self.require_fvg     = use_fvg_entry
        self.use_choch_filter = False   # MTF gestiona CHoCH internamente

    def generate_signal(self, data):
        """
        Genera señal MTF.

        data : DataFrame 1h (o el TF bajo configurado)

        Devuelve dict: signal, reason, high_tf_trend, low_tf_signal, pd_zone
        """
        if data is None or len(data) < self.high_tf_window * 4:
            return {"signal": "hold", "reason": "Datos insuficientes para MTF"}

        # ── 1. Tendencia del TF alto ────────────────────────────────────────
        try:
            df_high = resample(data, self.high_tf)
        except Exception:
            # Sin timestamps reales, simular resample por bloques
            factor   = _tf_factor(self.high_tf)
            df_high  = _block_resample(data, factor)

        high_struct = _detect_structure_level(df_high, self.high_tf_window)
        if high_struct is None or high_struct["trend"] == "neutral":
            return {"signal": "hold", "reason": "TF alto sin tendencia clara",
                    "high_tf_trend": "neutral"}

        high_trend = high_struct["trend"]   # "bullish" o "bearish"

        # ── 2. Filtro Premium / Discount en TF alto ─────────────────────────
        pd_zone = "unknown"
        if self.use_pd_filter:
            pd = detect_premium_discount(df_high, swing_window=self.high_tf_window)
            if pd:
                pd_zone = pd["zone"]
                current_price = float(data["close"].iloc[-1])
                zone = price_zone(current_price, pd)
                if high_trend == "bullish" and zone == "premium":
                    return {"signal": "hold",
                            "reason": "Precio en Premium — esperar Discount para long",
                            "high_tf_trend": high_trend, "pd_zone": pd_zone}
                if high_trend == "bearish" and zone == "discount":
                    return {"signal": "hold",
                            "reason": "Precio en Discount — esperar Premium para short",
                            "high_tf_trend": high_trend, "pd_zone": pd_zone}

        # ── 3. Estructura interna del TF bajo ───────────────────────────────
        low_struct = _detect_structure_level(data, self.low_tf_window)
        if low_struct is None:
            return {"signal": "hold", "reason": "TF bajo sin estructura",
                    "high_tf_trend": high_trend, "pd_zone": pd_zone}

        low_trend  = low_struct["trend"]
        low_bos    = low_struct["bos"]
        low_choch  = low_struct["choch"]
        low_dir    = low_struct["bos_direction"]

        # ── 4. Lógica de entrada MTF ─────────────────────────────────────────
        # Bullish setup: TF alto bullish + TF bajo BOS alcista (tras pullback bajista)
        if high_trend == "bullish":
            if self.require_pullback:
                # Exige que el TF bajo haya tenido un CHoCH bajista (pullback)
                # y ahora el BOS sea alcista (reanuda la tendencia)
                if not (low_bos and low_dir == "bullish"):
                    return {"signal": "hold",
                            "reason": "Esperando BOS alcista en TF bajo (pullback pendiente)",
                            "high_tf_trend": high_trend, "pd_zone": pd_zone}
            else:
                if not (low_bos and low_dir == "bullish"):
                    return {"signal": "hold",
                            "reason": "Sin BOS alcista en TF bajo",
                            "high_tf_trend": high_trend, "pd_zone": pd_zone}
            direction = "buy"

        # Bearish setup: TF alto bearish + TF bajo BOS bajista
        elif high_trend == "bearish":
            if self.require_pullback:
                if not (low_bos and low_dir == "bearish"):
                    return {"signal": "hold",
                            "reason": "Esperando BOS bajista en TF bajo",
                            "high_tf_trend": high_trend, "pd_zone": pd_zone}
            else:
                if not (low_bos and low_dir == "bearish"):
                    return {"signal": "hold",
                            "reason": "Sin BOS bajista en TF bajo",
                            "high_tf_trend": high_trend, "pd_zone": pd_zone}
            direction = "sell"
        else:
            return {"signal": "hold", "reason": "TF alto neutral",
                    "high_tf_trend": high_trend, "pd_zone": pd_zone}

        # ── 5. Filtro EMA ────────────────────────────────────────────────────
        if self.use_ema_filter:
            ema_trend = ema_trend_filter(data, style=self.trading_style)
            if direction == "buy"  and ema_trend != "bullish":
                return {"signal": "hold", "reason": "EMA no confirma long",
                        "high_tf_trend": high_trend, "pd_zone": pd_zone}
            if direction == "sell" and ema_trend != "bearish":
                return {"signal": "hold", "reason": "EMA no confirma short",
                        "high_tf_trend": high_trend, "pd_zone": pd_zone}

        # ── 6. Refinamiento: FVG / OB del TF bajo ───────────────────────────
        current_price = float(data["close"].iloc[-1])
        if self.use_fvg_entry:
            fvg_type    = "bullish" if direction == "buy" else "bearish"
            active_fvgs = get_active_fvgs(data)
            if not any(f["type"] == fvg_type for f in active_fvgs):
                return {"signal": "hold", "reason": f"Sin FVG {fvg_type} activo en TF bajo",
                        "high_tf_trend": high_trend, "pd_zone": pd_zone}

        if self.use_ob_entry:
            ob_type    = "bullish" if direction == "buy" else "bearish"
            active_obs = get_active_order_blocks(data, self.low_tf_window)
            if not price_in_order_block(current_price, active_obs, ob_type):
                return {"signal": "hold", "reason": f"Precio fuera de OB {ob_type}",
                        "high_tf_trend": high_trend, "pd_zone": pd_zone}

        # ── Señal confirmada ─────────────────────────────────────────────────
        parts = [f"MTF {high_trend.upper()} ({self.high_tf})",
                 f"BOS {'↑' if direction=='buy' else '↓'} ({self.low_tf_window}v)"]
        if self.use_pd_filter:  parts.append(f"PD:{pd_zone}")
        if self.use_fvg_entry:  parts.append("FVG")
        if self.use_ob_entry:   parts.append("OB")

        return {
            "signal":        direction,
            "reason":        " + ".join(parts),
            "high_tf_trend": high_trend,
            "low_tf_trend":  low_trend,
            "pd_zone":       pd_zone,
        }


# ─── Helpers internos ────────────────────────────────────────────────────────

def _tf_factor(tf):
    """Cuántas velas de 1h equivalen a un candle del TF alto."""
    return {"5m": 1, "15m": 1, "30m": 1, "1h": 1,
            "2h": 2, "4h": 4, "6h": 6, "8h": 8,
            "12h": 12, "1d": 24, "1w": 168}.get(tf, 4)


def _block_resample(df, factor):
    """Resamplea por bloques de N filas cuando no hay timestamps reales."""
    import pandas as pd
    n      = len(df)
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    vols   = df["volume"].values

    rows = []
    for i in range(0, n - factor + 1, factor):
        rows.append({
            "open":   opens[i],
            "high":   highs[i:i+factor].max(),
            "low":    lows[i:i+factor].min(),
            "close":  closes[i+factor-1],
            "volume": vols[i:i+factor].sum(),
        })
    return pd.DataFrame(rows) if rows else df.head(0)
