def _find_swing_highs(data, window=5):
    highs = []
    for i in range(window, len(data) - window):
        local_high = float(data["high"].iloc[i])
        left = data["high"].iloc[i - window:i].astype(float)
        right = data["high"].iloc[i + 1:i + window + 1].astype(float)
        if local_high > left.max() and local_high >= right.max():
            highs.append((i, local_high))
    return highs


def _find_swing_lows(data, window=5):
    lows = []
    for i in range(window, len(data) - window):
        local_low = float(data["low"].iloc[i])
        left = data["low"].iloc[i - window:i].astype(float)
        right = data["low"].iloc[i + 1:i + window + 1].astype(float)
        if local_low < left.min() and local_low <= right.min():
            lows.append((i, local_low))
    return lows


def detect_market_structure(data, window=5):
    """
    Detecta estructura de mercado basada en swing highs y swing lows reales.

    - trend: bullish (HH+HL), bearish (LH+LL) o neutral
    - bos: True si el cierre actual rompe el ultimo swing high (bullish) o low (bearish)
    - choch: True si la rotura va en contra de la tendencia previa (cambio de caracter)
    """
    min_candles = window * 2 + 2
    if data is None or len(data) < min_candles:
        return None

    swing_highs = _find_swing_highs(data, window)
    swing_lows = _find_swing_lows(data, window)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    current_close = float(data["close"].iloc[-1])

    last_sh = swing_highs[-1][1]
    prev_sh = swing_highs[-2][1]
    last_sl = swing_lows[-1][1]
    prev_sl = swing_lows[-2][1]

    # Determinar tendencia por secuencia de swings
    higher_highs = last_sh > prev_sh
    higher_lows = last_sl > prev_sl
    lower_highs = last_sh < prev_sh
    lower_lows = last_sl < prev_sl

    if higher_highs and higher_lows:
        trend = "bullish"
    elif lower_highs and lower_lows:
        trend = "bearish"
    else:
        trend = "neutral"

    # Detectar BOS y CHoCH segun rotura del ultimo swing
    bos = False
    choch = False

    if current_close > last_sh:
        bos = True
        if trend == "bearish":
            choch = True  # rompe al alza en tendencia bajista
        trend = "bullish"
    elif current_close < last_sl:
        bos = True
        if trend == "bullish":
            choch = True  # rompe a la baja en tendencia alcista
        trend = "bearish"

    return {
        "trend": trend,
        "bos": bos,
        "choch": choch,
        "last_swing_high": last_sh,
        "last_swing_low": last_sl,
    }
