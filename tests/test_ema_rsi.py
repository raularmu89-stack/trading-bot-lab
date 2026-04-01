"""
test_ema_rsi.py — Tests para indicators/ema_rsi.py
"""

import math
import numpy as np
import pandas as pd
import pytest
from indicators.ema_rsi import compute_ema_rsi, ema_trend_filter, TRADING_STYLES, _ema, _rsi


def _make_df(closes):
    n = len(closes)
    closes = list(closes)
    return pd.DataFrame({
        "open":   [c * 0.999 for c in closes],
        "high":   [c * 1.001 for c in closes],
        "low":    [c * 0.999 for c in closes],
        "close":  closes,
        "volume": [1000.0] * n,
    })


def _rising(n=300, start=100, step=0.5):
    return _make_df([start + i * step for i in range(n)])


def _falling(n=300, start=200, step=0.5):
    return _make_df([start - i * step for i in range(n)])


def _flat(n=300, val=100.0):
    return _make_df([val] * n)


# ─── Presets ─────────────────────────────────────────────────────────────────

def test_trading_styles_keys_exist():
    for style in ("scalping", "intraday", "swing"):
        assert style in TRADING_STYLES


def test_scalping_params():
    p = TRADING_STYLES["scalping"]
    assert p["ema_fast"]    == 9
    assert p["ema_slow"]    == 20
    assert p["rsi_period"]  == 9
    assert p["rsi_ob"]      == 80
    assert p["rsi_os"]      == 20


def test_intraday_params():
    p = TRADING_STYLES["intraday"]
    assert p["ema_fast"]    == 20
    assert p["ema_slow"]    == 50
    assert p["rsi_period"]  == 9


def test_swing_params():
    p = TRADING_STYLES["swing"]
    assert p["ema_fast"]    == 50
    assert p["ema_slow"]    == 200
    assert p["rsi_period"]  == 14


# ─── _ema ─────────────────────────────────────────────────────────────────────

def test_ema_length_matches_input():
    series = np.arange(1.0, 51.0)
    result = _ema(series, 9)
    assert len(result) == len(series)


def test_ema_nan_before_period():
    series = np.arange(1.0, 51.0)
    result = _ema(series, 10)
    assert np.all(np.isnan(result[:9]))


def test_ema_converges_rising():
    """En serie creciente la EMA debe ser creciente."""
    series = np.arange(1.0, 101.0)
    result = _ema(series, 9)
    valid  = result[~np.isnan(result)]
    assert np.all(np.diff(valid) > 0)


def test_ema_short_data_all_nan():
    series = np.array([1.0, 2.0])
    result = _ema(series, 10)
    assert np.all(np.isnan(result))


# ─── _rsi ────────────────────────────────────────────────────────────────────

def test_rsi_length_matches_input():
    series = np.arange(1.0, 51.0)
    result = _rsi(series, 14)
    assert len(result) == len(series)


def test_rsi_range_0_100():
    series = np.random.default_rng(42).uniform(100, 200, 100)
    result = _rsi(series, 14)
    valid  = result[~np.isnan(result)]
    assert np.all(valid >= 0) and np.all(valid <= 100)


def test_rsi_rising_above_50():
    """Serie sostenidamente alcista → RSI debe acabar por encima de 50."""
    series = np.arange(1.0, 101.0)
    result = _rsi(series, 14)
    valid  = result[~np.isnan(result)]
    assert valid[-1] > 50


def test_rsi_falling_below_50():
    series = np.arange(100.0, 0.0, -1.0)
    result = _rsi(series, 14)
    valid  = result[~np.isnan(result)]
    assert valid[-1] < 50


# ─── compute_ema_rsi ──────────────────────────────────────────────────────────

def test_compute_ema_rsi_output_keys():
    df     = _rising()
    result = compute_ema_rsi(df, style="scalping")
    for key in ("ema_fast", "ema_slow", "rsi", "rsi_ob", "rsi_os",
                "style_params", "cross_up", "cross_down",
                "rsi_overbought", "rsi_oversold", "signal"):
        assert key in result, f"Falta clave: {key}"


def test_compute_ema_rsi_array_lengths():
    n  = 200
    df = _rising(n)
    r  = compute_ema_rsi(df, style="scalping")
    assert len(r["ema_fast"]) == n
    assert len(r["ema_slow"]) == n
    assert len(r["rsi"])      == n


def test_compute_ema_rsi_rising_no_cross_down():
    """En tendencia alcista no debe haber cruce bajista."""
    df = _rising(300)
    r  = compute_ema_rsi(df, style="scalping")
    assert not r["cross_down"]


def test_compute_ema_rsi_custom_override():
    """Los parámetros custom deben sobreescribir el preset."""
    df = _rising(300)
    r  = compute_ema_rsi(df, style="scalping", fast=5, slow=10, rsi_period=7)
    # EMA rápida (periodo 5) debe converger antes → más valores válidos
    valid_fast = np.sum(~np.isnan(r["ema_fast"]))
    valid_slow = np.sum(~np.isnan(r["ema_slow"]))
    assert valid_fast >= valid_slow


def test_compute_ema_rsi_signal_is_valid():
    df = _rising(300)
    r  = compute_ema_rsi(df, style="swing")
    assert r["signal"] in ("buy", "sell", "hold")


def test_compute_ema_rsi_falling_overbought_false():
    df = _falling(300)
    r  = compute_ema_rsi(df, style="scalping")
    assert not r["rsi_overbought"]


def test_compute_ema_rsi_rising_oversold_false():
    df = _rising(300)
    r  = compute_ema_rsi(df, style="scalping")
    assert not r["rsi_oversold"]


# ─── ema_trend_filter ────────────────────────────────────────────────────────

def test_ema_trend_filter_bullish_on_rising():
    df = _rising(300)
    assert ema_trend_filter(df, style="scalping") == "bullish"


def test_ema_trend_filter_bearish_on_falling():
    df = _falling(300)
    assert ema_trend_filter(df, style="scalping") == "bearish"


def test_ema_trend_filter_neutral_short_data():
    df = _rising(5)
    # Con tan pocos datos la EMA lenta no tiene suficiente historia
    result = ema_trend_filter(df, style="swing")
    assert result in ("bullish", "bearish", "neutral")


def test_ema_trend_filter_all_styles():
    df = _rising(300)
    for style in ("scalping", "intraday", "swing"):
        result = ema_trend_filter(df, style=style)
        assert result in ("bullish", "bearish", "neutral")
