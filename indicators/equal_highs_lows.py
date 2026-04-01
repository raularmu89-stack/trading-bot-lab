"""
equal_highs_lows.py

Conversión desde Pine Script 'raul smc' (TradingView).

Equal Highs (EQH) y Equal Lows (EQL):
  Detecta cuando un nuevo pivot está muy cerca del anterior,
  indicando una zona de liquidez acumulada.

Pine equivalent:
    getCurrentStructure(equalHighsLowsLengthInput, true)
    → dentro de la función:
        if equalHighLow and abs(p_ivot.currentLevel - high[size]) < threshold * atr
            drawEqualHighLow(...)

Parametros de Pine por defecto:
    equalHighsLowsLengthInput    = 3  (bars confirmation)
    equalHighsLowsThresholdInput = 0.1 (sensibilidad, fracción de ATR)
"""

import numpy as np
from indicators.market_structure import _find_pivots


def _atr_simple(highs, lows, closes, period=14):
    """ATR simple para la comparación de umbral."""
    n   = len(closes)
    if n < 2:
        return 0.0
    period  = min(period, n - 1)
    tr      = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]),
                   np.abs(lows[1:]  - closes[:-1]))
    )
    return float(np.mean(tr[-period:]))


def detect_equal_highs_lows(data, window=3, threshold=0.1, atr_period=14):
    """
    Detecta Equal Highs (EQH) y Equal Lows (EQL).

    Pine equivalent:
        equalHighsLowsLengthInput    = window     (por defecto 3)
        equalHighsLowsThresholdInput = threshold  (por defecto 0.1)
        abs(p_ivot.currentLevel - level) < threshold * atrMeasure

    Parametros:
      window     : velas de confirmación del pivot (Pine: equalHighsLowsLengthInput)
      threshold  : sensibilidad en fracción de ATR (Pine: equalHighsLowsThresholdInput)
      atr_period : periodo del ATR de referencia

    Devuelve dict:
      eqh : lista de (index1, index2, price) → pares de highs iguales
      eql : lista de (index1, index2, price) → pares de lows iguales
    """
    if data is None or len(data) < window * 2 + 2:
        return {"eqh": [], "eql": []}

    h = data["high"].values.astype(float)
    l = data["low"].values.astype(float)
    c = data["close"].values.astype(float)

    atr_val      = _atr_simple(h, l, c, atr_period)
    atol         = threshold * atr_val

    pivot_highs, pivot_lows = _find_pivots(h, l, window, require_left=(window < 20))

    eqh = []
    for i in range(1, len(pivot_highs)):
        idx1, price1 = pivot_highs[i - 1]
        idx2, price2 = pivot_highs[i]
        if abs(price1 - price2) < atol:
            eqh.append({
                "index1": idx1,
                "index2": idx2,
                "price":  (price1 + price2) / 2,
            })

    eql = []
    for i in range(1, len(pivot_lows)):
        idx1, price1 = pivot_lows[i - 1]
        idx2, price2 = pivot_lows[i]
        if abs(price1 - price2) < atol:
            eql.append({
                "index1": idx1,
                "index2": idx2,
                "price":  (price1 + price2) / 2,
            })

    return {"eqh": eqh, "eql": eql}


def latest_equal_level(data, window=3, threshold=0.1, atr_period=14):
    """
    Devuelve el EQH y EQL más recientes como niveles de precio.

    Util para filtros en estrategias.
    """
    result = detect_equal_highs_lows(data, window, threshold, atr_period)
    last_eqh = result["eqh"][-1]["price"] if result["eqh"] else None
    last_eql = result["eql"][-1]["price"] if result["eql"] else None
    return {"eqh": last_eqh, "eql": last_eql}
