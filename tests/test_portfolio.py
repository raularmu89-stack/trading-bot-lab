"""
test_portfolio.py — Tests para bot/portfolio.py
"""

import pytest
from bot.mock_data import generate_mock_klines
from bot.portfolio import Portfolio
from strategies.smc_strategy import SMCStrategy
from strategies.risk_manager import RiskManager


# ─── Fixtures ────────────────────────────────────────────────────────────────

PAIRS = ["BTCUSDT", "ETHUSDT"]
N_CANDLES = 300


@pytest.fixture
def mock_data():
    return {s: generate_mock_klines(s, n_candles=N_CANDLES, seed=42) for s in PAIRS}


@pytest.fixture
def strategy():
    return SMCStrategy(swing_window=5, require_fvg=False, use_choch_filter=True)


@pytest.fixture
def risk_mgr():
    return RiskManager(sl_pct=0.02, rr_ratio=2.0, method="fixed")


# ─── Tests de construccion ───────────────────────────────────────────────────

def test_portfolio_builds(mock_data, strategy, risk_mgr):
    pf = Portfolio(
        pairs=PAIRS,
        data=mock_data,
        strategy=strategy,
        risk_manager=risk_mgr,
        initial_capital=100.0,
    )
    assert pf.initial_capital == 100.0
    assert pf.pairs == PAIRS


# ─── Tests de run ────────────────────────────────────────────────────────────

def test_portfolio_run_returns_expected_keys(mock_data, strategy, risk_mgr):
    pf = Portfolio(
        pairs=PAIRS,
        data=mock_data,
        strategy=strategy,
        risk_manager=risk_mgr,
        initial_capital=100.0,
    )
    results = pf.run()
    assert "pairs" in results
    assert "weights" in results
    assert "capital_per_pair" in results
    assert "portfolio_equity" in results
    assert "metrics" in results
    assert "total_return" in results
    assert "final_capital" in results


def test_portfolio_equity_starts_near_initial(mock_data, strategy, risk_mgr):
    initial = 100.0
    pf = Portfolio(
        pairs=PAIRS,
        data=mock_data,
        strategy=strategy,
        risk_manager=risk_mgr,
        initial_capital=initial,
    )
    results = pf.run()
    eq = results["portfolio_equity"]
    assert len(eq) > 0
    assert abs(eq[0] - initial) < 1.0, f"Primer valor de equity debe ser ~{initial}: {eq[0]}"


def test_portfolio_weights_sum_to_one(mock_data, strategy):
    pf = Portfolio(
        pairs=PAIRS,
        data=mock_data,
        strategy=strategy,
        initial_capital=100.0,
        allocation="equal",
    )
    results = pf.run()
    total_weight = sum(results["weights"].values())
    assert abs(total_weight - 1.0) < 1e-9, f"Pesos no suman 1: {total_weight}"


def test_portfolio_capital_per_pair_sums_to_initial(mock_data, strategy):
    initial = 200.0
    pf = Portfolio(
        pairs=PAIRS,
        data=mock_data,
        strategy=strategy,
        initial_capital=initial,
        allocation="equal",
    )
    results = pf.run()
    total = sum(results["capital_per_pair"].values())
    assert abs(total - initial) < 1e-6, f"Capital total no coincide: {total}"


# ─── Tests de modos de asignacion ────────────────────────────────────────────

def test_allocation_equal(mock_data, strategy):
    pf = Portfolio(
        pairs=PAIRS, data=mock_data, strategy=strategy,
        initial_capital=100.0, allocation="equal",
    )
    results = pf.run()
    w = results["weights"]
    assert abs(w["BTCUSDT"] - 0.5) < 1e-9
    assert abs(w["ETHUSDT"] - 0.5) < 1e-9


def test_allocation_inv_vol(mock_data, strategy):
    pf = Portfolio(
        pairs=PAIRS, data=mock_data, strategy=strategy,
        initial_capital=100.0, allocation="inv_vol",
    )
    results = pf.run()
    w = results["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # Ambos pesos deben ser positivos
    assert all(v > 0 for v in w.values())


def test_allocation_custom(mock_data, strategy):
    custom = {"BTCUSDT": 0.7, "ETHUSDT": 0.3}
    pf = Portfolio(
        pairs=PAIRS, data=mock_data, strategy=strategy,
        initial_capital=100.0, allocation=custom,
    )
    results = pf.run()
    w = results["weights"]
    assert abs(w["BTCUSDT"] - 0.7) < 1e-9
    assert abs(w["ETHUSDT"] - 0.3) < 1e-9


def test_allocation_unknown_raises(mock_data, strategy):
    pf = Portfolio(
        pairs=PAIRS, data=mock_data, strategy=strategy,
        initial_capital=100.0, allocation="magic",
    )
    with pytest.raises(ValueError, match="allocation desconocido"):
        pf.run()


# ─── Tests de metricas del portfolio ─────────────────────────────────────────

def test_portfolio_metrics_keys(mock_data, strategy):
    pf = Portfolio(
        pairs=PAIRS, data=mock_data, strategy=strategy, initial_capital=100.0
    )
    results = pf.run()
    m = results["metrics"]
    for key in ("total_return", "ann_return", "volatility",
                "max_drawdown", "sharpe", "sortino", "calmar"):
        assert key in m, f"Falta clave: {key}"


def test_portfolio_pair_results_have_metrics(mock_data, strategy, risk_mgr):
    pf = Portfolio(
        pairs=PAIRS, data=mock_data, strategy=strategy,
        risk_manager=risk_mgr, initial_capital=100.0,
    )
    results = pf.run()
    for s in PAIRS:
        pr = results["pairs"][s]
        assert "sharpe" in pr
        assert "max_drawdown" in pr
        assert "sortino" in pr
        assert "calmar" in pr


# ─── Tests con 4 pares ───────────────────────────────────────────────────────

def test_portfolio_four_pairs():
    pairs4 = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]
    data4 = {s: generate_mock_klines(s, n_candles=200, seed=7) for s in pairs4}
    strategy = SMCStrategy()
    pf = Portfolio(
        pairs=pairs4, data=data4, strategy=strategy, initial_capital=100.0
    )
    results = pf.run()
    assert len(results["weights"]) == 4
    assert abs(sum(results["weights"].values()) - 1.0) < 1e-9
