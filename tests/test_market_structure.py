import pandas as pd
import pytest
from indicators.market_structure import detect_market_structure


def _make_df(closes, highs=None, lows=None):
    n = len(closes)
    if highs is None:
        highs = [c * 1.005 for c in closes]
    if lows is None:
        lows = [c * 0.995 for c in closes]
    return pd.DataFrame({
        "open":   closes,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": [1000.0] * n,
    })


def test_returns_none_with_insufficient_data():
    df = _make_df([100, 101, 102])
    assert detect_market_structure(df, window=5) is None


def test_returns_none_when_data_is_none():
    assert detect_market_structure(None) is None


def test_bullish_trend_detected():
    # Swings HH+HL claros con window=2:
    # swing highs en i=2(15) e i=6(20) → HH
    # swing lows  en i=4(4)  e i=8(5)  → HL
    closes = [5, 8, 15, 7, 4, 10, 20, 9, 5, 12, 25, 11]
    df = _make_df(closes)
    result = detect_market_structure(df, window=2)
    assert result is not None
    assert result["trend"] == "bullish"


def test_bearish_trend_detected():
    # Swings LH+LL claros: inverso del bullish
    closes = [25, 12, 5, 9, 20, 10, 4, 7, 15, 8, 4, 6]
    df = _make_df(closes)
    result = detect_market_structure(df, window=2)
    assert result is not None
    assert result["trend"] == "bearish"


def test_bos_triggered_on_breakout():
    # Igual que bullish pero el ultimo cierre rompe por encima del swing high
    closes = [5, 8, 15, 7, 4, 10, 20, 9, 5, 12, 25, 30]
    highs  = [c * 1.005 for c in closes]
    lows   = [c * 0.995 for c in closes]
    df = _make_df(closes, highs, lows)
    result = detect_market_structure(df, window=2)
    assert result is not None
    assert result["bos"] is True


def test_result_has_required_keys():
    closes = [10, 11, 9, 12, 10, 13, 11, 14, 12, 15]
    df = _make_df(closes)
    result = detect_market_structure(df, window=2)
    if result is not None:
        for key in ("trend", "bos", "choch", "last_swing_high", "last_swing_low"):
            assert key in result
