def detect_market_structure(data):
    if data is None or len(data) < 3:
        return None

    return {
        "trend": "neutral",
        "bos": False,
        "choch": False
    }
