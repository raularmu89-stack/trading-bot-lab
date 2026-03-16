def detect_market_structure(data):
    """
    Detección base muy simple de estructura.
    Esto luego lo refinaremos.
    """
    if data is None or len(data) < 3:
        return None

    last_close = float(data["close"].iloc[-1])
    prev_close = float(data["close"].iloc[-2])
    prev_prev_close = float(data["close"].iloc[-3])

    trend = "neutral"
    bos = False
    choch = False

    if last_close > prev_close > prev_prev_close:
        trend = "bullish"
        bos = True

    elif last_close < prev_close < prev_prev_close:
        trend = "bearish"
        bos = True

    else:
        choch = True

    return {
        "trend": trend,
        "bos": bos,
        "choch": choch
    }
