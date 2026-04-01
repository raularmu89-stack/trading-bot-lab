"""Tests para indicators/regime_detector.py"""
import pytest
import numpy as np
import pandas as pd
from indicators.regime_detector import (
    detect_regime, detect_regime_series, RegimeDetector,
    _atr, _ema, _adx, _rsi, _range_width,
)

VALID_REGIMES = {
    "strong_trend_bull", "strong_trend_bear",
    "weak_trend_bull", "weak_trend_bear",
    "ranging", "breakout", "high_volatility",
    "mean_reversion_bull", "mean_reversion_bear",
    "insufficient_data",
}

def _make(n=200, trend="bull", noise=0.4, seed=42):
    rng = np.random.default_rng(seed)
    prices = [100.0]
    slope = {"bull": 0.3, "bear": -0.3, "flat": 0.0}[trend]
    for _ in range(n - 1):
        prices.append(max(1.0, prices[-1] + slope + rng.standard_normal() * noise))
    prices = np.array(prices)
    spread = abs(rng.standard_normal(n)) * 0.3
    return pd.DataFrame({
        "open": prices, "high": prices + spread,
        "low": np.maximum(prices - spread, prices * 0.98),
        "close": prices, "volume": np.ones(n) * 1000,
    })


class TestATRAdxRsi:
    def test_atr_positive(self):
        df = _make()
        atr = _atr(df["high"].values, df["low"].values, df["close"].values)
        assert all(a >= 0 for a in atr)

    def test_ema_length(self):
        c = np.linspace(100, 200, 100)
        assert len(_ema(c, 20)) == 100

    def test_adx_keys(self):
        df = _make()
        result = _adx(df["high"].values, df["low"].values, df["close"].values)
        assert "adx" in result and "plus_di" in result and "minus_di" in result

    def test_adx_non_negative(self):
        df = _make()
        result = _adx(df["high"].values, df["low"].values, df["close"].values)
        assert all(v >= 0 for v in result["adx"])

    def test_rsi_range(self):
        df = _make()
        rsi = _rsi(df["close"].values)
        assert all(0 <= v <= 100 for v in rsi)


class TestDetectRegime:
    def test_returns_dict(self):
        df = _make(200)
        result = detect_regime(df)
        assert isinstance(result, dict)
        assert "regime" in result

    def test_regime_is_valid(self):
        df = _make(200)
        r = detect_regime(df)
        assert r["regime"] in VALID_REGIMES

    def test_insufficient_data_returns_flag(self):
        df = _make(10)
        r = detect_regime(df)
        assert r["regime"] == "insufficient_data"

    def test_none_returns_flag(self):
        r = detect_regime(None)
        assert r["regime"] == "insufficient_data"

    def test_required_keys_present(self):
        df = _make(200)
        r = detect_regime(df)
        for key in ["regime", "adx", "atr_ratio", "rsi", "ema_fast", "ema_slow"]:
            assert key in r, f"Falta key: {key}"

    def test_atr_ratio_positive(self):
        df = _make(200)
        r = detect_regime(df)
        assert r["atr_ratio"] > 0

    def test_flat_market_tends_to_ranging(self):
        # Mercado plano → suele ser ranging o mean_reversion
        df = _make(300, trend="flat", noise=0.05)
        r = detect_regime(df)
        assert r["regime"] in ("ranging", "mean_reversion_bull", "mean_reversion_bear",
                               "weak_trend_bull", "weak_trend_bear")


class TestDetectRegimeSeries:
    def test_returns_list_same_length(self):
        df = _make(200)
        result = detect_regime_series(df)
        assert len(result) == len(df)

    def test_all_values_valid(self):
        df = _make(200)
        result = detect_regime_series(df)
        for r in result:
            assert r in VALID_REGIMES

    def test_none_returns_empty(self):
        result = detect_regime_series(None)
        assert result == []


class TestRegimeDetectorClass:
    def test_detect_same_as_function(self):
        df = _make(200)
        rd = RegimeDetector()
        assert rd.detect(df)["regime"] == detect_regime(df)["regime"]

    def test_detect_all_length(self):
        df = _make(200)
        rd = RegimeDetector()
        assert len(rd.detect_all(df)) == 200

    def test_is_trending(self):
        assert RegimeDetector.is_trending("strong_trend_bull")
        assert RegimeDetector.is_trending("weak_trend_bear")
        assert not RegimeDetector.is_trending("ranging")
        assert not RegimeDetector.is_trending("breakout")

    def test_is_bullish_bearish(self):
        assert RegimeDetector.is_bullish("strong_trend_bull")
        assert RegimeDetector.is_bearish("strong_trend_bear")
        assert not RegimeDetector.is_bullish("ranging")

    def test_regime_label_not_empty(self):
        label = RegimeDetector.regime_label("strong_trend_bull")
        assert len(label) > 3
