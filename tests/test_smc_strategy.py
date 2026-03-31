import pandas as pd
import pytest
from strategies.smc_strategy import SMCStrategy


def _make_df(n=30, trend="bullish"):
    """Genera un DataFrame simple con tendencia controlada."""
    import numpy as np
    rng = np.random.default_rng(0)
    if trend == "bullish":
        closes = 100 + np.cumsum(np.abs(rng.normal(0.5, 0.3, n)))
    elif trend == "bearish":
        closes = 100 - np.cumsum(np.abs(rng.normal(0.5, 0.3, n)))
    else:
        closes = 100 + rng.normal(0, 0.5, n)

    noise = rng.uniform(0.003, 0.01, n)
    return pd.DataFrame({
        "open":   closes * (1 - noise / 2),
        "high":   closes * (1 + noise),
        "low":    closes * (1 - noise),
        "close":  closes,
        "volume": [1000.0] * n,
    })


def test_returns_hold_with_too_little_data():
    strategy = SMCStrategy()
    df = _make_df(n=3)
    result = strategy.generate_signal(df)
    assert result["signal"] == "hold"


def test_returns_hold_when_none():
    strategy = SMCStrategy()
    result = strategy.generate_signal(None)
    assert result["signal"] == "hold"


def test_signal_is_valid_value():
    strategy = SMCStrategy()
    df = _make_df(n=40)
    result = strategy.generate_signal(df)
    assert result["signal"] in ("buy", "sell", "hold")


def test_signal_contains_reason():
    strategy = SMCStrategy()
    df = _make_df(n=40)
    result = strategy.generate_signal(df)
    assert "reason" in result


def test_swing_window_param_respected():
    s1 = SMCStrategy(swing_window=3)
    s2 = SMCStrategy(swing_window=15)
    assert s1.swing_window == 3
    assert s2.swing_window == 15


def test_require_fvg_param():
    strategy = SMCStrategy(require_fvg=True)
    assert strategy.require_fvg is True


def test_use_choch_filter_param():
    strategy = SMCStrategy(use_choch_filter=False)
    assert strategy.use_choch_filter is False
