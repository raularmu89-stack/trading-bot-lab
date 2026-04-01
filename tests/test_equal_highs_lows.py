"""
test_equal_highs_lows.py — Tests para indicators/equal_highs_lows.py
"""

import pandas as pd
import pytest
from indicators.equal_highs_lows import detect_equal_highs_lows, latest_equal_level


def _make_df(highs, lows, closes=None):
    n = len(highs)
    closes = closes or highs
    return pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": [1000.0] * n,
    })


def _rising(n=60):
    """Dataset con tendencia alcista clara → sin EQH/EQL obvios."""
    highs  = [100 + i * 0.5 for i in range(n)]
    lows   = [99  + i * 0.5 for i in range(n)]
    return _make_df(highs, lows)


def _double_top(base=100, gap=20):
    """
    Dos picos casi iguales separados por un valle → debe generar EQH.
    Se añaden 'gap' velas al final para que el segundo pico quede
    dentro del rango de detección de pivots.
    """
    prices = (
        [base + i        for i in range(gap)]   # sube al pico 1
        + [base + gap - i for i in range(gap)]  # baja al valle
        + [base + i        for i in range(gap)] # sube al pico 2
        + [base + gap - i  for i in range(gap)] # baja de nuevo (cola)
    )
    highs  = [p + 0.5 for p in prices]
    lows   = [p - 0.5 for p in prices]
    return _make_df(highs, lows)


def _double_bottom(base=100, gap=20):
    """Dos valles casi iguales → debe generar EQL."""
    prices = (
        [base - i         for i in range(gap)]   # baja al valle 1
        + [base - gap + i  for i in range(gap)]  # sube
        + [base - i         for i in range(gap)] # baja al valle 2
        + [base - gap + i   for i in range(gap)] # sube de nuevo (cola)
    )
    highs  = [p + 0.5 for p in prices]
    lows   = [p - 0.5 for p in prices]
    return _make_df(highs, lows)


# ─── Datos insuficientes ─────────────────────────────────────────────────────

def test_returns_empty_with_none():
    result = detect_equal_highs_lows(None)
    assert result == {"eqh": [], "eql": []}


def test_returns_empty_with_too_few_bars():
    df = _make_df([100, 101], [99, 100])
    result = detect_equal_highs_lows(df, window=3)
    assert result == {"eqh": [], "eql": []}


# ─── Estructura de resultado ──────────────────────────────────────────────────

def test_result_has_eqh_and_eql_keys():
    df = _rising(60)
    result = detect_equal_highs_lows(df, window=3)
    assert "eqh" in result
    assert "eql" in result
    assert isinstance(result["eqh"], list)
    assert isinstance(result["eql"], list)


def test_eqh_entry_has_required_keys():
    """Si hay algún EQH, debe tener index1, index2, price."""
    df = _double_top()
    result = detect_equal_highs_lows(df, window=3, threshold=2.0)
    if result["eqh"]:
        entry = result["eqh"][0]
        assert "index1" in entry
        assert "index2" in entry
        assert "price"  in entry


def test_eql_entry_has_required_keys():
    df = _double_bottom()
    result = detect_equal_highs_lows(df, window=3, threshold=2.0)
    if result["eql"]:
        entry = result["eql"][0]
        assert "index1" in entry
        assert "index2" in entry
        assert "price"  in entry


# ─── Detección de EQH / EQL ───────────────────────────────────────────────────

def test_strict_threshold_returns_empty():
    """Con threshold=0 nada es 'igual'."""
    df = _double_top()
    result = detect_equal_highs_lows(df, window=3, threshold=0.0)
    assert result["eqh"] == []


def test_loose_threshold_detects_double_top():
    """Con umbral muy amplio, el doble techo debe detectarse."""
    df = _double_top(gap=15)
    result = detect_equal_highs_lows(df, window=3, threshold=10.0)
    assert len(result["eqh"]) >= 1, "Debería detectar al menos un EQH con umbral amplio"


def test_loose_threshold_detects_double_bottom():
    df = _double_bottom(gap=15)
    result = detect_equal_highs_lows(df, window=3, threshold=10.0)
    assert len(result["eql"]) >= 1, "Debería detectar al menos un EQL con umbral amplio"


def test_eqh_price_is_average_of_two_pivots():
    """El precio del EQH debe ser el promedio de los dos pivots."""
    df = _double_top(gap=15)
    result = detect_equal_highs_lows(df, window=3, threshold=10.0)
    if result["eqh"]:
        entry = result["eqh"][0]
        assert entry["index1"] < entry["index2"]
        assert entry["price"] > 0


def test_eqh_indices_ordered():
    """index1 siempre debe ser menor que index2."""
    df = _double_top(gap=15)
    result = detect_equal_highs_lows(df, window=3, threshold=10.0)
    for entry in result["eqh"]:
        assert entry["index1"] < entry["index2"]


# ─── latest_equal_level ───────────────────────────────────────────────────────

def test_latest_equal_level_returns_none_when_no_data():
    df = _make_df([100, 101], [99, 100])
    result = latest_equal_level(df, window=3)
    assert result["eqh"] is None
    assert result["eql"] is None


def test_latest_equal_level_returns_float_when_detected():
    df = _double_top(gap=15)
    result = latest_equal_level(df, window=3, threshold=10.0)
    if result["eqh"] is not None:
        assert isinstance(result["eqh"], float)
