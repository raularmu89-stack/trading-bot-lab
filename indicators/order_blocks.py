"""
order_blocks.py

Conversión desde Pine Script 'raul smc' (TradingView).

Detecta Order Blocks (OB) en dos niveles:
  - Internal OBs : basados en estructura interna (window=5)
  - Swing OBs    : basados en estructura de swing (window=50)

Lógica Pine original:
  storeOrderBlock():
    Bearish OB → bar con mayor parsedHigh entre p_ivot.barIndex y BOS
    Bullish OB → bar con menor parsedLow  entre p_ivot.barIndex y BOS

  Filtro de volatilidad (parsedHigh/Low):
    Si (high - low) >= 2 * ATR(200) → barra de alta volatilidad
      parsedHigh = low  (ignorar el high extremo)
      parsedLow  = high (ignorar el low extremo)
    Si no → parsedHigh = high, parsedLow = low

  Mitigación:
    Bearish OB: mitigado si close (o high) > ob_high
    Bullish OB: mitigado si close (o low)  < ob_low
"""

import numpy as np
from indicators.market_structure import _find_pivots


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _atr(highs, lows, closes, period=200):
    """ATR simple sin librería externa."""
    n = len(closes)
    period = min(period, n - 1)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:]  - closes[:-1]),
        )
    )
    if len(tr) == 0:
        return np.zeros(n)
    atr_vals = np.full(n, np.nan)
    atr_vals[1] = tr[0]
    alpha = 1.0 / period
    for i in range(1, len(tr)):
        atr_vals[i + 1] = alpha * tr[i] + (1 - alpha) * atr_vals[i]
    # Rellenar NaN del inicio con el primer valor válido
    first_valid = np.nanargmin(np.isnan(atr_vals[::-1]))
    atr_vals[:] = np.where(np.isnan(atr_vals), np.nanmean(atr_vals), atr_vals)
    return atr_vals


def _parsed_highs_lows(highs, lows, closes, atr_period=200):
    """
    Aplica el filtro de volatilidad de Pine:
      Si barra de alta volatilidad → swap high y low para parseo.

    Pine equivalent:
        highVolatilityBar = (high - low) >= 2 * atrMeasure
        parsedHigh = highVolatilityBar ? low  : high
        parsedLow  = highVolatilityBar ? high : low
    """
    atr_vals = _atr(highs, lows, closes, atr_period)
    high_vol = (highs - lows) >= (2 * atr_vals)
    parsed_highs = np.where(high_vol, lows,  highs)
    parsed_lows  = np.where(high_vol, highs, lows)
    return parsed_highs, parsed_lows


# ─── Core OB detection ───────────────────────────────────────────────────────

def _detect_order_blocks(data, window, mitigation="highlow", atr_period=200):
    """
    Detecta order blocks para un nivel de estructura dado.

    Pine equivalent:
        storeOrderBlock(p_ivot, internal, bias)
        deleteOrderBlocks(internal)

    Parametros:
      window      : ventana de pivots (5 = internal, 50 = swing)
      mitigation  : "close" o "highlow" (cierre o high/low para mitigación)
      atr_period  : periodo del ATR para filtro de volatilidad

    Devuelve lista de dicts:
      type        : "bullish" | "bearish"
      ob_high     : límite superior del OB
      ob_low      : límite inferior del OB
      ob_index    : índice de la vela OB
      pivot_index : índice del pivot que lo generó
      mitigated   : True si el precio ha cruzado el OB
    """
    if data is None or len(data) < window * 2 + 3:
        return []

    h   = data["high"].values.astype(float)
    l   = data["low"].values.astype(float)
    c   = data["close"].values.astype(float)
    o   = data["open"].values.astype(float)
    n   = len(data)

    parsed_highs, parsed_lows = _parsed_highs_lows(h, l, c, atr_period)

    pivot_highs, pivot_lows = _find_pivots(h, l, window, require_left=(window < 20))
    order_blocks = []

    # ── Bullish OBs (antes de un BOS alcista / rotura de swing high) ─────────
    # Pine: Cuando close > swingHigh → buscar barra con menor parsedLow
    #       entre swingHigh.barIndex y la barra actual → ese es el Bullish OB
    for sh_idx, sh_price in pivot_highs:
        # Buscar si hay un BOS alcista después de este swing high
        bos_bar = None
        for j in range(sh_idx + 1, n):
            if c[j] > sh_price:
                bos_bar = j
                break
        if bos_bar is None:
            continue

        # Buscar barra con menor parsedLow en la ventana [sh_idx, bos_bar)
        window_slice = parsed_lows[sh_idx:bos_bar]
        if len(window_slice) == 0:
            continue
        local_idx   = int(np.argmin(window_slice))
        ob_idx      = sh_idx + local_idx
        ob_high     = float(h[ob_idx])
        ob_low      = float(l[ob_idx])

        # Comprobar mitigación
        if mitigation == "close":
            mitigated = bool(np.any(c[bos_bar:] < ob_low))
        else:  # highlow
            mitigated = bool(np.any(l[bos_bar:] < ob_low))

        order_blocks.append({
            "type":        "bullish",
            "ob_high":     ob_high,
            "ob_low":      ob_low,
            "ob_index":    ob_idx,
            "pivot_index": sh_idx,
            "mitigated":   mitigated,
        })

    # ── Bearish OBs (antes de un BOS bajista / rotura de swing low) ──────────
    for sl_idx, sl_price in pivot_lows:
        bos_bar = None
        for j in range(sl_idx + 1, n):
            if c[j] < sl_price:
                bos_bar = j
                break
        if bos_bar is None:
            continue

        window_slice = parsed_highs[sl_idx:bos_bar]
        if len(window_slice) == 0:
            continue
        local_idx   = int(np.argmax(window_slice))
        ob_idx      = sl_idx + local_idx
        ob_high     = float(h[ob_idx])
        ob_low      = float(l[ob_idx])

        if mitigation == "close":
            mitigated = bool(np.any(c[bos_bar:] > ob_high))
        else:  # highlow
            mitigated = bool(np.any(h[bos_bar:] > ob_high))

        order_blocks.append({
            "type":        "bearish",
            "ob_high":     ob_high,
            "ob_low":      ob_low,
            "ob_index":    ob_idx,
            "pivot_index": sl_idx,
            "mitigated":   mitigated,
        })

    return order_blocks


# ─── Public API ──────────────────────────────────────────────────────────────

def detect_order_blocks(data, swing_window=5, mitigation="highlow",
                        atr_period=200, min_move=0.0):
    """
    Detecta Order Blocks (legacy + Pine-faithful).

    Parámetros:
      swing_window : ventana de pivot (usa como nivel único)
      mitigation   : "close" | "highlow"
      atr_period   : periodo ATR para filtro de volatilidad
      min_move     : ignorado (compatibilidad backwards)

    Devuelve lista de dicts con claves:
      type, ob_high, ob_low, ob_index, pivot_index, mitigated
      (+ clave 'broken' = mitigated para compatibilidad con código anterior)
    """
    obs = _detect_order_blocks(data, swing_window, mitigation, atr_period)
    # Añadir clave 'broken' para backward compatibility
    for ob in obs:
        ob["broken"] = ob["mitigated"]
        ob["index"]  = ob["ob_index"]
    return obs


def detect_dual_order_blocks(data, internal_window=5, swing_window=50,
                              mitigation="highlow", atr_period=200):
    """
    Detecta OBs en dos niveles (internal + swing).

    Pine equivalent:
        showInternalOrderBlocksInput → internal OBs (window=5)
        showSwingOrderBlocksInput    → swing OBs    (window=50)

    Devuelve dict:
      internal : lista de OBs con internal_window
      swing    : lista de OBs con swing_window
    """
    internal = _detect_order_blocks(data, internal_window, mitigation, atr_period)
    swing    = _detect_order_blocks(data, swing_window,    mitigation, atr_period)
    return {"internal": internal, "swing": swing}


def get_active_order_blocks(data, swing_window=5, mitigation="highlow",
                            atr_period=200, min_move=0.0):
    """Devuelve solo los OBs que NO han sido mitigados."""
    obs = detect_order_blocks(data, swing_window, mitigation, atr_period)
    return [ob for ob in obs if not ob["mitigated"]]


def price_in_order_block(price, order_blocks, ob_type=None):
    """
    Comprueba si un precio está dentro de algún OB.

    ob_type : "bullish" | "bearish" | None (cualquiera)
    """
    for ob in order_blocks:
        if ob_type and ob["type"] != ob_type:
            continue
        if ob["ob_low"] <= price <= ob["ob_high"]:
            return True
    return False
