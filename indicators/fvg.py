def detect_fvg(data):
    """
    Detector base simple de Fair Value Gaps.
    Devuelve una lista de gaps encontrados.
    """
    if data is None or len(data) < 3:
        return []

    fvgs = []

    for i in range(2, len(data)):
        candle_1_high = float(data["high"].iloc[i - 2])
        candle_1_low = float(data["low"].iloc[i - 2])

        candle_3_high = float(data["high"].iloc[i])
        candle_3_low = float(data["low"].iloc[i])

        # Bullish FVG
        if candle_3_low > candle_1_high:
            fvgs.append({
                "type": "bullish",
                "start": candle_1_high,
                "end": candle_3_low,
                "index": i
            })

        # Bearish FVG
        elif candle_3_high < candle_1_low:
            fvgs.append({
                "type": "bearish",
                "start": candle_3_high,
                "end": candle_1_low,
                "index": i
            })

    return fvgs
