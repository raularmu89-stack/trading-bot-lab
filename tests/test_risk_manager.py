import pandas as pd
import numpy as np
import pytest
from strategies.risk_manager import RiskManager


def _make_df(n=20):
    closes = 100 + np.cumsum(np.random.default_rng(1).normal(0, 0.5, n))
    noise = np.random.default_rng(1).uniform(0.002, 0.01, n)
    return pd.DataFrame({
        "open":   closes * (1 - noise / 2),
        "high":   closes * (1 + noise),
        "low":    closes * (1 - noise),
        "close":  closes,
        "volume": [1000.0] * n,
    })


def test_fixed_buy_sl_below_entry():
    rm = RiskManager(sl_pct=0.02, rr_ratio=2.0, method="fixed")
    df = _make_df()
    sl, tp = rm.calculate_levels(df, entry_price=100.0, side="buy")
    assert sl < 100.0
    assert tp > 100.0


def test_fixed_sell_sl_above_entry():
    rm = RiskManager(sl_pct=0.02, rr_ratio=2.0, method="fixed")
    df = _make_df()
    sl, tp = rm.calculate_levels(df, entry_price=100.0, side="sell")
    assert sl > 100.0
    assert tp < 100.0


def test_fixed_rr_ratio_respected():
    rm = RiskManager(sl_pct=0.02, rr_ratio=3.0, method="fixed")
    df = _make_df()
    sl, tp = rm.calculate_levels(df, entry_price=100.0, side="buy")
    sl_dist = 100.0 - sl
    tp_dist = tp - 100.0
    assert abs(tp_dist / sl_dist - 3.0) < 0.01


def test_atr_method_returns_valid_levels():
    rm = RiskManager(rr_ratio=2.0, method="atr", atr_multiplier=1.5)
    df = _make_df(n=30)
    sl, tp = rm.calculate_levels(df, entry_price=100.0, side="buy")
    assert sl < 100.0
    assert tp > 100.0


def test_atr_rr_ratio_respected():
    rm = RiskManager(rr_ratio=2.0, method="atr", atr_multiplier=1.5)
    df = _make_df(n=30)
    sl, tp = rm.calculate_levels(df, entry_price=100.0, side="buy")
    sl_dist = 100.0 - sl
    tp_dist = tp - 100.0
    assert abs(tp_dist / sl_dist - 2.0) < 0.01


def test_short_data_falls_back_gracefully():
    rm = RiskManager(method="atr")
    df = _make_df(n=3)
    sl, tp = rm.calculate_levels(df, entry_price=50.0, side="buy")
    assert sl < 50.0
    assert tp > 50.0
