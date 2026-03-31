import numpy as np
import pandas as pd
import pytest
from strategies.smc_strategy import SMCStrategy
from strategies.risk_manager import RiskManager
from backtests.backtester_fast import FastBacktester, _precompute_signals, _run_trades, _metrics


def _make_df(n=60, seed=42):
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(0.001, 0.015, n))
    noise = rng.uniform(0.003, 0.01, n)
    return pd.DataFrame({
        "open":   closes * (1 - noise / 2),
        "high":   closes * (1 + noise),
        "low":    closes * (1 - noise),
        "close":  closes,
        "volume": [1000.0] * n,
    })


def test_precompute_signals_length():
    df = _make_df(60)
    signals = _precompute_signals(df, swing_window=3)
    assert len(signals) == len(df)


def test_precompute_signals_valid_values():
    df = _make_df(60)
    signals = _precompute_signals(df, swing_window=3)
    assert all(s in ("buy", "sell", "hold") for s in signals)


def test_metrics_empty_trades():
    result = _metrics([], [1.0])
    assert result["trades"] == 0
    assert result["winrate"] == 0.0
    assert result["profit_factor"] == 0.0
    assert result["total_pnl"] == 0.0


def test_metrics_all_wins():
    trades = [{"pnl": 0.05, "win": True, "exit": "tp"} for _ in range(5)]
    equity = [1.0, 1.05, 1.1025, 1.157625, 1.21550625, 1.2762815625]
    result = _metrics(trades, equity)
    assert result["winrate"] == 1.0
    assert result["profit_factor"] == float("inf")
    assert result["total_pnl"] > 0


def test_metrics_mixed():
    trades = [
        {"pnl": 0.04, "win": True,  "exit": "tp"},
        {"pnl": -0.02, "win": False, "exit": "sl"},
        {"pnl": 0.04, "win": True,  "exit": "tp"},
        {"pnl": -0.02, "win": False, "exit": "sl"},
    ]
    equity = [1.0, 1.04, 1.0192, 1.059968, 1.038769]
    result = _metrics(trades, equity)
    assert result["winrate"] == 0.5
    assert abs(result["profit_factor"] - 2.0) < 0.01
    assert result["sl_hits"] == 2
    assert result["tp_hits"] == 2


def test_fast_backtester_returns_dict():
    df = _make_df(80)
    strategy = SMCStrategy(swing_window=3)
    result = FastBacktester(strategy, df, max_hold=5).run()
    for key in ("trades", "winrate", "profit_factor", "total_pnl"):
        assert key in result


def test_fast_backtester_with_risk_manager():
    df = _make_df(100)
    strategy = SMCStrategy(swing_window=3)
    rm = RiskManager(sl_pct=0.02, rr_ratio=2.0, method="fixed")
    result = FastBacktester(strategy, df, max_hold=10, risk_manager=rm).run()
    assert result["trades"] >= 0
    assert 0.0 <= result["winrate"] <= 1.0


def test_sl_tp_hits_tracked():
    df = _make_df(200, seed=7)
    strategy = SMCStrategy(swing_window=3, use_choch_filter=False)
    rm = RiskManager(sl_pct=0.02, rr_ratio=2.0, method="fixed")
    result = FastBacktester(strategy, df, max_hold=20, risk_manager=rm).run()
    # sl_hits + tp_hits <= trades (algunos pueden cerrar por señal o fin)
    assert result["sl_hits"] + result["tp_hits"] <= result["trades"]
