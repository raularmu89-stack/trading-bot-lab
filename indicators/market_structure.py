"""
market_structure.py

Conversión desde Pine Script 'raul smc' (TradingView).

Detecta estructura de mercado SMC en DOS niveles simultáneos:
  - Internal (internal_window=5) : estructura interna de corto plazo
  - Swing    (swing_window=50)   : estructura de swing de largo plazo

Para cada nivel detecta:
  BOS   (Break of Structure)   : rotura en dirección de la tendencia
  CHoCH (Change of Character)  : rotura contra la tendencia previa

Pine equivalents:
  leg(size)               → _find_pivots(window)
  startOfBullishLeg()     → pivot low confirmado
  startOfBearishLeg()     → pivot high confirmado
  displayStructure()      → _detect_structure_level()
  updateTrailingExtremes()→ trailing_high / trailing_low
  drawHighLowSwings()     → strong_high / weak_high / strong_low / weak_low
"""

import numpy as np


# ─── Pivot detection ─────────────────────────────────────────────────────────

def _find_pivots(highs, lows, window, require_left=True):
    """
    Detecta pivot highs y lows.

    Pine equivalent (right-side only):
        newLegHigh = high[size] > ta.highest(size)
        → h[i] > max(h[i+1 : i+size+1])

    Para ventanas grandes (swing_window >= 20) se desactiva el lado izquierdo
    (require_left=False) para no eliminar demasiados pivots, igualando el
    comportamiento del estado-máquina de Pine.

    Devuelve:
      pivot_highs : list of (index, float)
      pivot_lows  : list of (index, float)
    """
    n = len(highs)
    pivot_highs = []
    pivot_lows  = []

    for i in range(window, n - window):
        h = highs[i]
        l = lows[i]

        right_h = h > highs[i + 1:i + window + 1].max()
        left_h  = (not require_left) or h > highs[i - window:i].max()

        right_l = l < lows[i + 1:i + window + 1].min()
        left_l  = (not require_left) or l < lows[i - window:i].min()

        if right_h and left_h:
            pivot_highs.append((i, float(h)))
        if right_l and left_l:
            pivot_lows.append((i, float(l)))

    return pivot_highs, pivot_lows


# ─── Single-level structure ───────────────────────────────────────────────────

def _detect_structure_level(data, window):
    """
    Detecta BOS/CHoCH para un nivel de estructura dado.

    Devuelve dict con:
      trend         : "bullish" | "bearish" | "neutral"
      bos           : True si hay Break of Structure en la vela actual
      bos_direction : "bullish" | "bearish" (solo si bos=True)
      choch         : True si es Change of Character
      last_sh / last_sl      : precio del último swing high/low
      last_sh_idx / last_sl_idx : índice del último swing
      prev_sh / prev_sl      : penúltimo swing high/low
      pivot_highs / pivot_lows : listas completas (index, price)
      hh, hl, lh, ll         : secuencia de swings detectada
    """
    min_candles = window * 2 + 2
    if data is None or len(data) < min_candles:
        return None

    h = data["high"].values.astype(float)
    l = data["low"].values.astype(float)
    c = data["close"].values.astype(float)

    pivot_highs, pivot_lows = _find_pivots(h, l, window, require_left=(window < 20))

    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return None

    last_sh,  prev_sh  = pivot_highs[-1][1], pivot_highs[-2][1]
    last_sl,  prev_sl  = pivot_lows[-1][1],  pivot_lows[-2][1]
    last_sh_idx        = pivot_highs[-1][0]
    last_sl_idx        = pivot_lows[-1][0]

    hh = last_sh > prev_sh
    hl = last_sl > prev_sl
    lh = last_sh < prev_sh
    ll = last_sl < prev_sl

    if hh and hl:
        trend = "bullish"
    elif lh and ll:
        trend = "bearish"
    else:
        trend = "neutral"

    current_close = c[-1]
    bos           = False
    choch         = False
    bos_direction = None

    if current_close > last_sh:
        bos           = True
        bos_direction = "bullish"
        choch         = trend == "bearish"
    elif current_close < last_sl:
        bos           = True
        bos_direction = "bearish"
        choch         = trend == "bullish"

    return {
        "trend":         trend,
        "bos":           bos,
        "bos_direction": bos_direction,
        "choch":         choch,
        "last_sh":       last_sh,
        "last_sl":       last_sl,
        "last_sh_idx":   last_sh_idx,
        "last_sl_idx":   last_sl_idx,
        "prev_sh":       prev_sh,
        "prev_sl":       prev_sl,
        "pivot_highs":   pivot_highs,
        "pivot_lows":    pivot_lows,
        "hh": hh, "hl": hl, "lh": lh, "ll": ll,
    }


# ─── Dual-level structure (Pine main API) ────────────────────────────────────

def detect_dual_structure(data, internal_window=5, swing_window=50):
    """
    Detecta estructura en dos niveles simultáneamente.

    Pine equivalents:
        getCurrentStructure(swingsLengthInput)       → swing level
        getCurrentStructure(5, false, true)          → internal level
        updateTrailingExtremes() / drawHighLowSwings()→ trailing + strong/weak

    Parametros:
      internal_window : ventana interna  (Pine: 5)
      swing_window    : ventana de swing (Pine: 50, configurable)

    Devuelve dict:
      internal       : resultado _detect_structure_level(internal_window)
      swing          : resultado _detect_structure_level(swing_window)
      trailing_high  : máximo absoluto del rango
      trailing_low   : mínimo absoluto del rango
      strong_high    : resistencia fuerte (tendencia bajista → liquidez arriba)
      weak_high      : resistencia débil  (tendencia alcista → probablemente superada)
      strong_low     : soporte fuerte     (tendencia alcista → demanda abajo)
      weak_low       : soporte débil      (tendencia bajista → probablemente roto)
    """
    if data is None or len(data) < swing_window * 2 + 2:
        return None

    internal = _detect_structure_level(data, internal_window)
    swing    = _detect_structure_level(data, swing_window)

    if swing is None:
        return None

    h = data["high"].values.astype(float)
    l = data["low"].values.astype(float)
    trailing_high = float(h.max())
    trailing_low  = float(l.min())

    swing_trend = swing["trend"]

    return {
        "internal":      internal,
        "swing":         swing,
        "trailing_high": trailing_high,
        "trailing_low":  trailing_low,
        "strong_high": trailing_high if swing_trend == "bearish" else None,
        "weak_high":   trailing_high if swing_trend == "bullish" else None,
        "strong_low":  trailing_low  if swing_trend == "bullish" else None,
        "weak_low":    trailing_low  if swing_trend == "bearish" else None,
    }


# ─── Legacy API (backward compatibility) ─────────────────────────────────────

def detect_market_structure(data, window=5):
    """
    API de compatibilidad con el código existente.
    Detecta un solo nivel de estructura.

    Devuelve dict con claves originales:
      trend, bos, choch, last_swing_high, last_swing_low
    """
    result = _detect_structure_level(data, window)
    if result is None:
        return None
    return {
        "trend":           result["trend"],
        "bos":             result["bos"],
        "choch":           result["choch"],
        "last_swing_high": result["last_sh"],
        "last_swing_low":  result["last_sl"],
    }
