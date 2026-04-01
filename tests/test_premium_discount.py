"""
test_premium_discount.py — Tests para indicators/premium_discount.py
"""

import pandas as pd
import pytest
from indicators.premium_discount import detect_premium_discount, price_zone


def _make_df(highs, lows, closes=None):
    n = len(highs)
    closes = closes or [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": [1000.0] * n,
    })


def _range_df(low=100.0, high=200.0, n=100, close_pct=0.5):
    """Dataset con rango fijo [low, high], precio en close_pct del rango."""
    close = low + (high - low) * close_pct
    highs  = [high] * n
    lows   = [low]  * n
    closes = [close] * n
    return _make_df(highs, lows, closes)


# ─── Datos insuficientes ─────────────────────────────────────────────────────

def test_returns_none_with_none():
    assert detect_premium_discount(None) is None


def test_returns_none_with_single_bar():
    df = _make_df([100], [99])
    assert detect_premium_discount(df) is None


def test_returns_none_with_zero_range():
    df = _make_df([100.0] * 20, [100.0] * 20, [100.0] * 20)
    assert detect_premium_discount(df) is None


# ─── Estructura de resultado ──────────────────────────────────────────────────

def test_result_has_required_keys():
    df = _range_df()
    result = detect_premium_discount(df)
    assert result is not None
    for key in ("swing_high", "swing_low", "premium_top", "premium_bot",
                "eq_top", "eq_bot", "discount_top", "discount_bot",
                "equilibrium", "zone", "zone_pct"):
        assert key in result, f"Falta clave: {key}"


def test_swing_high_and_low_correct():
    df = _range_df(low=100, high=200)
    r  = detect_premium_discount(df)
    assert abs(r["swing_high"] - 200) < 0.01
    assert abs(r["swing_low"]  - 100) < 0.01


def test_equilibrium_is_midpoint():
    df = _range_df(low=100, high=200)
    r  = detect_premium_discount(df)
    assert abs(r["equilibrium"] - 150) < 0.01


# ─── Niveles de Pine (porcentajes exactos) ───────────────────────────────────

def test_premium_bot_is_95pct_from_top():
    """Pine: premium_bot = 0.95*top + 0.05*bottom."""
    df  = _range_df(low=100, high=200)
    r   = detect_premium_discount(df)
    expected = 0.95 * 200 + 0.05 * 100
    assert abs(r["premium_bot"] - expected) < 0.01


def test_discount_top_is_5pct_from_top():
    """Pine: discount_top = 0.05*top + 0.95*bottom."""
    df  = _range_df(low=100, high=200)
    r   = detect_premium_discount(df)
    expected = 0.05 * 200 + 0.95 * 100
    assert abs(r["discount_top"] - expected) < 0.01


def test_eq_zone_spans_around_midpoint():
    df = _range_df(low=100, high=200)
    r  = detect_premium_discount(df)
    assert r["eq_bot"] < r["equilibrium"] < r["eq_top"]


def test_zones_are_ordered():
    df = _range_df(low=100, high=200)
    r  = detect_premium_discount(df)
    assert r["discount_bot"] < r["discount_top"] <= r["eq_bot"] < r["eq_top"] <= r["premium_bot"] < r["premium_top"]


# ─── Clasificación de zona ────────────────────────────────────────────────────

def test_price_at_top_is_premium():
    df = _range_df(low=100, high=200, close_pct=0.99)
    r  = detect_premium_discount(df)
    assert r["zone"] == "premium"


def test_price_at_bottom_is_discount():
    df = _range_df(low=100, high=200, close_pct=0.01)
    r  = detect_premium_discount(df)
    assert r["zone"] == "discount"


def test_price_at_midpoint_is_equilibrium():
    df = _range_df(low=100, high=200, close_pct=0.50)
    r  = detect_premium_discount(df)
    assert r["zone"] == "equilibrium"


def test_zone_pct_at_top_near_one():
    df = _range_df(low=100, high=200, close_pct=0.99)
    r  = detect_premium_discount(df)
    assert r["zone_pct"] > 0.9


def test_zone_pct_at_bottom_near_zero():
    df = _range_df(low=100, high=200, close_pct=0.01)
    r  = detect_premium_discount(df)
    assert r["zone_pct"] < 0.1


# ─── price_zone ───────────────────────────────────────────────────────────────

def test_price_zone_returns_unknown_for_none():
    assert price_zone(150.0, None) == "unknown"


def test_price_zone_premium():
    df = _range_df(low=100, high=200)
    r  = detect_premium_discount(df)
    assert price_zone(195.0, r) == "premium"


def test_price_zone_discount():
    df = _range_df(low=100, high=200)
    r  = detect_premium_discount(df)
    assert price_zone(105.0, r) == "discount"


def test_price_zone_equilibrium():
    df = _range_df(low=100, high=200)
    r  = detect_premium_discount(df)
    assert price_zone(150.0, r) == "equilibrium"
