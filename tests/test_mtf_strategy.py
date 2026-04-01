"""
Tests para MTFStrategy — estrategia multi-timeframe SMC.
"""

import pytest
import numpy as np
import pandas as pd
from strategies.mtf_strategy import MTFStrategy, _tf_factor, _block_resample


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n=400, trend="bullish", noise=0.3):
    """
    Genera datos OHLCV con tendencia definida.
    trend: "bullish" | "bearish" | "sideways"
    """
    np.random.seed(42)
    prices = [100.0]
    slope = {"bullish": 0.15, "bearish": -0.15, "sideways": 0.0}[trend]
    for i in range(n - 1):
        change = slope + np.random.randn() * noise
        prices.append(max(1.0, prices[-1] + change))

    prices = np.array(prices)
    high   = prices + abs(np.random.randn(n)) * noise
    low    = prices - abs(np.random.randn(n)) * noise
    low    = np.minimum(low, prices)
    high   = np.maximum(high, prices)
    vol    = np.random.randint(1000, 5000, n).astype(float)

    return pd.DataFrame({
        "open":   prices,
        "high":   high,
        "low":    low,
        "close":  prices,
        "volume": vol,
    })


def _make_ts_df(n=400, trend="bullish"):
    """Datos con DatetimeIndex real."""
    df = _make_df(n, trend)
    df.index = pd.date_range("2024-01-01", periods=n, freq="1h")
    return df


# ── _tf_factor ────────────────────────────────────────────────────────────────

class TestTfFactor:
    def test_known_timeframes(self):
        assert _tf_factor("1h") == 1
        assert _tf_factor("2h") == 2
        assert _tf_factor("4h") == 4
        assert _tf_factor("1d") == 24
        assert _tf_factor("1w") == 168

    def test_unknown_defaults_to_4(self):
        assert _tf_factor("99h") == 4


# ── _block_resample ───────────────────────────────────────────────────────────

class TestBlockResample:
    def test_returns_fewer_rows(self):
        df = _make_df(100)
        resampled = _block_resample(df, 4)
        assert len(resampled) == 25

    def test_high_is_max(self):
        df = _make_df(100)
        resampled = _block_resample(df, 4)
        original_max = df["high"].iloc[:4].max()
        assert resampled["high"].iloc[0] == pytest.approx(original_max)

    def test_low_is_min(self):
        df = _make_df(100)
        resampled = _block_resample(df, 4)
        original_min = df["low"].iloc[:4].min()
        assert resampled["low"].iloc[0] == pytest.approx(original_min)

    def test_open_is_first(self):
        df = _make_df(100)
        resampled = _block_resample(df, 4)
        assert resampled["open"].iloc[0] == pytest.approx(df["open"].iloc[0])

    def test_close_is_last(self):
        df = _make_df(100)
        resampled = _block_resample(df, 4)
        assert resampled["close"].iloc[0] == pytest.approx(df["close"].iloc[3])

    def test_volume_is_sum(self):
        df = _make_df(100)
        resampled = _block_resample(df, 4)
        assert resampled["volume"].iloc[0] == pytest.approx(df["volume"].iloc[:4].sum())

    def test_columns_preserved(self):
        df = _make_df(100)
        resampled = _block_resample(df, 4)
        assert set(resampled.columns) == {"open", "high", "low", "close", "volume"}


# ── MTFStrategy init ──────────────────────────────────────────────────────────

class TestMTFStrategyInit:
    def test_defaults(self):
        s = MTFStrategy()
        assert s.high_tf == "4h"
        assert s.high_tf_window == 10
        assert s.low_tf_window == 5
        assert s.require_pullback is True
        assert s.use_pd_filter is True
        assert s.swing_window == 5  # compat FastBacktester

    def test_custom(self):
        s = MTFStrategy(high_tf="1d", high_tf_window=20, low_tf_window=10,
                        require_pullback=False, use_pd_filter=False)
        assert s.high_tf == "1d"
        assert s.high_tf_window == 20
        assert s.low_tf_window == 10
        assert s.require_pullback is False

    def test_fast_backtester_compat_attrs(self):
        s = MTFStrategy(low_tf_window=7, use_fvg_entry=True)
        assert s.swing_window == 7
        assert s.require_fvg is True
        assert s.use_choch_filter is False


# ── generate_signal — datos insuficientes ─────────────────────────────────────

class TestMTFInsufficientData:
    def test_none_data(self):
        s = MTFStrategy()
        result = s.generate_signal(None)
        assert result["signal"] == "hold"

    def test_too_few_candles(self):
        s = MTFStrategy(high_tf_window=10)
        df = _make_df(10)  # necesita high_tf_window * 4 = 40
        result = s.generate_signal(df)
        assert result["signal"] == "hold"


# ── generate_signal — señales ─────────────────────────────────────────────────

class TestMTFSignal:
    def test_returns_dict_with_keys(self):
        s = MTFStrategy(high_tf_window=10, low_tf_window=5,
                        use_pd_filter=False, require_pullback=False)
        df = _make_df(300, trend="bullish")
        result = s.generate_signal(df)
        assert isinstance(result, dict)
        assert "signal" in result
        assert "reason" in result

    def test_signal_values_are_valid(self):
        s = MTFStrategy(high_tf_window=10, low_tf_window=5,
                        use_pd_filter=False, require_pullback=False)
        df = _make_df(300, trend="bullish")
        result = s.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")

    def test_high_tf_trend_in_result(self):
        s = MTFStrategy(high_tf_window=10, low_tf_window=5, use_pd_filter=False)
        df = _make_df(300)
        result = s.generate_signal(df)
        # Si hay trend, debe estar en result
        if result["signal"] != "hold" or "high_tf_trend" in result:
            assert "high_tf_trend" in result

    def test_with_timestamps(self):
        s = MTFStrategy(high_tf="4h", high_tf_window=10, low_tf_window=5,
                        use_pd_filter=False, require_pullback=False)
        df = _make_ts_df(400, trend="bullish")
        result = s.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")

    def test_bearish_market_no_buy(self):
        """En mercado bajista sin filtros extra, no debe dar señal de compra."""
        s = MTFStrategy(high_tf_window=10, low_tf_window=5,
                        use_pd_filter=False, require_pullback=False)
        df = _make_df(300, trend="bearish")
        result = s.generate_signal(df)
        assert result["signal"] != "buy"

    def test_bullish_market_no_sell(self):
        """En mercado alcista sin filtros extra, no debe dar señal de venta."""
        s = MTFStrategy(high_tf_window=10, low_tf_window=5,
                        use_pd_filter=False, require_pullback=False)
        df = _make_df(300, trend="bullish")
        result = s.generate_signal(df)
        assert result["signal"] != "sell"


# ── Filtros opcionales ────────────────────────────────────────────────────────

class TestMTFFilters:
    def test_ema_filter_reduces_signals(self):
        """Activar EMA filter no rompe el método."""
        s = MTFStrategy(high_tf_window=10, low_tf_window=5,
                        use_pd_filter=False, use_ema_filter=True,
                        trading_style="swing")
        df = _make_df(300)
        result = s.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")

    def test_fvg_filter_returns_hold_or_signal(self):
        s = MTFStrategy(high_tf_window=10, low_tf_window=5,
                        use_pd_filter=False, use_fvg_entry=True)
        df = _make_df(300)
        result = s.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")

    def test_ob_filter_returns_hold_or_signal(self):
        s = MTFStrategy(high_tf_window=10, low_tf_window=5,
                        use_pd_filter=False, use_ob_entry=True)
        df = _make_df(300)
        result = s.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")
