"""
liquidity.py

Detecta Liquidity Sweeps (barridas de liquidez) en SMC:

La liquidez se acumula encima de swing highs (buy stops) y
debajo de swing lows (sell stops). Un sweep ocurre cuando el
precio los toca brevemente y revierte, indicando que los market
makers han absorbido esas ordenes.

- Bullish sweep: el precio baja, toca por debajo de un swing low
  y cierra por encima de el → señal de posible reversal alcista.

- Bearish sweep: el precio sube, toca por encima de un swing high
  y cierra por debajo de el → señal de posible reversal bajista.
"""


def detect_liquidity_sweeps(data, swing_window=5):
    """
    Detecta liquidity sweeps en el dataset.

    Devuelve lista de dicts:
      type    : "bullish" (sweep bajista + reversal) o "bearish"
      level   : precio del swing barrido
      index   : indice de la vela donde ocurrio el sweep
      candle_low  / candle_high : extremo tocado en el sweep
    """
    if data is None or len(data) < swing_window * 2 + 2:
        return []

    highs  = data["high"].values
    lows   = data["low"].values
    closes = data["close"].values
    n = len(data)

    # Detectar swing highs y lows previos (excluir las ultimas swing_window velas)
    swing_highs = []
    swing_lows  = []
    for i in range(swing_window, n - swing_window):
        if highs[i] > highs[i - swing_window:i].max() and \
           highs[i] >= highs[i + 1:i + swing_window + 1].max():
            swing_highs.append((i, highs[i]))
        if lows[i] < lows[i - swing_window:i].min() and \
           lows[i] <= lows[i + 1:i + swing_window + 1].min():
            swing_lows.append((i, lows[i]))

    sweeps = []

    # Bullish sweep: vela baja por debajo de un swing low pero cierra por encima
    for i in range(swing_window + 1, n):
        for sl_idx, sl_price in swing_lows:
            if sl_idx >= i:
                continue
            if lows[i] < sl_price and closes[i] > sl_price:
                sweeps.append({
                    "type":       "bullish",
                    "level":      sl_price,
                    "index":      i,
                    "candle_low": lows[i],
                })

    # Bearish sweep: vela sube por encima de un swing high pero cierra por debajo
    for i in range(swing_window + 1, n):
        for sh_idx, sh_price in swing_highs:
            if sh_idx >= i:
                continue
            if highs[i] > sh_price and closes[i] < sh_price:
                sweeps.append({
                    "type":        "bearish",
                    "level":       sh_price,
                    "index":       i,
                    "candle_high": highs[i],
                })

    # Ordenar por indice
    sweeps.sort(key=lambda x: x["index"])
    return sweeps


def last_sweep(data, swing_window=5):
    """Devuelve el sweep mas reciente o None si no hay ninguno."""
    sweeps = detect_liquidity_sweeps(data, swing_window)
    return sweeps[-1] if sweeps else None
