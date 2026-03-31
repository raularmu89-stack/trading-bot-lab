import pandas as pd
import pytest
from indicators.fvg import detect_fvg


def _make_df(opens, highs, lows, closes):
    return pd.DataFrame({
        "open": opens, "high": highs,
        "low": lows, "close": closes,
        "volume": [1000.0] * len(closes),
    })


def test_returns_empty_with_insufficient_data():
    df = _make_df([1, 2], [1.1, 2.1], [0.9, 1.9], [1, 2])
    assert detect_fvg(df) == []


def test_returns_empty_when_none():
    assert detect_fvg(None) == []


def test_bullish_fvg_detected():
    # Vela 3 low > vela 1 high → bullish FVG
    opens  = [10, 11, 15]
    highs  = [11, 12, 16]
    lows   = [9,  10, 14]   # lows[2]=14 > highs[0]=11 → bullish gap
    closes = [10, 11, 15]
    df = _make_df(opens, highs, lows, closes)
    fvgs = detect_fvg(df)
    assert len(fvgs) == 1
    assert fvgs[0]["type"] == "bullish"
    assert fvgs[0]["start"] == 11
    assert fvgs[0]["end"] == 14


def test_bearish_fvg_detected():
    # Vela 3 high < vela 1 low → bearish FVG
    opens  = [15, 14, 10]
    highs  = [16, 15, 11]   # highs[2]=11 < lows[0]=14 → bearish gap
    lows   = [14, 13, 9]
    closes = [15, 14, 10]
    df = _make_df(opens, highs, lows, closes)
    fvgs = detect_fvg(df)
    assert len(fvgs) == 1
    assert fvgs[0]["type"] == "bearish"


def test_no_fvg_when_candles_overlap():
    opens  = [10, 11, 10]
    highs  = [11, 12, 11]
    lows   = [9,  10,  9]
    closes = [10, 11, 10]
    df = _make_df(opens, highs, lows, closes)
    assert detect_fvg(df) == []


def test_fvg_index_is_correct():
    opens  = [10, 11, 15, 14]
    highs  = [11, 12, 16, 15]
    lows   = [9,  10, 14, 13]
    closes = [10, 11, 15, 14]
    df = _make_df(opens, highs, lows, closes)
    fvgs = detect_fvg(df)
    assert fvgs[0]["index"] == 2
