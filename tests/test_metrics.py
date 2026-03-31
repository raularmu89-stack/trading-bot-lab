"""
test_metrics.py — Tests para backtests/metrics.py
"""

import math
import pytest
from backtests.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    drawdown_series,
    calmar_ratio,
    compute_all,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

FLAT      = [100.0] * 10                         # sin variacion
RISING    = [100.0 * (1.01 ** i) for i in range(50)]   # sube 1% cada periodo
FALLING   = [100.0 * (0.99 ** i) for i in range(50)]   # baja 1% cada periodo
DRAWDOWN  = [100, 110, 120, 90, 80, 95, 100]    # cae de 120 a 80 = -33.3%
VOLATILE  = [100, 115, 95, 130, 85, 120, 100]   # sube y baja bruscamente


# ─── sharpe_ratio ────────────────────────────────────────────────────────────

def test_sharpe_flat_returns_zero():
    assert sharpe_ratio(FLAT) == 0.0


def test_sharpe_rising_positive():
    s = sharpe_ratio(RISING)
    assert s > 0, f"Sharpe deberia ser positivo para equity creciente: {s}"


def test_sharpe_falling_negative():
    s = sharpe_ratio(FALLING)
    assert s < 0, f"Sharpe deberia ser negativo para equity decreciente: {s}"


def test_sharpe_short_data():
    assert sharpe_ratio([100.0]) == 0.0


def test_sharpe_empty():
    assert sharpe_ratio([]) == 0.0


# ─── sortino_ratio ───────────────────────────────────────────────────────────

def test_sortino_all_positive_returns():
    # Si todos los retornos son positivos, sortino = inf
    assert sortino_ratio(RISING) == float("inf")


def test_sortino_falling_negative():
    s = sortino_ratio(FALLING)
    assert s < 0


def test_sortino_volatile():
    s = sortino_ratio(VOLATILE)
    assert isinstance(s, float)


def test_sortino_short_data():
    assert sortino_ratio([100.0]) == 0.0


# ─── max_drawdown ─────────────────────────────────────────────────────────────

def test_max_drawdown_known_value():
    mdd = max_drawdown(DRAWDOWN)
    # Pico en 120, valle en 80 → (80-120)/120 = -0.3333
    assert abs(mdd - (-1/3)) < 0.001, f"MDD esperado ~-0.333, obtenido {mdd}"


def test_max_drawdown_rising_near_zero():
    mdd = max_drawdown(RISING)
    assert mdd >= -0.01, f"Equity creciente no deberia tener gran drawdown: {mdd}"


def test_max_drawdown_falling():
    mdd = max_drawdown(FALLING)
    assert mdd < -0.3, f"Equity cayendo deberia tener gran drawdown: {mdd}"


def test_max_drawdown_short():
    assert max_drawdown([100.0]) == 0.0


# ─── drawdown_series ──────────────────────────────────────────────────────────

def test_drawdown_series_length():
    ds = drawdown_series(DRAWDOWN)
    assert len(ds) == len(DRAWDOWN)


def test_drawdown_series_all_non_positive():
    ds = drawdown_series(RISING)
    assert all(v <= 1e-9 for v in ds), "Drawdown serie no puede ser positiva"


# ─── calmar_ratio ────────────────────────────────────────────────────────────

def test_calmar_no_drawdown():
    # Equity perfectamente creciente → calmar = inf
    result = calmar_ratio(RISING)
    assert result == float("inf") or result > 50


def test_calmar_falling():
    c = calmar_ratio(FALLING)
    assert c < 0


def test_calmar_short():
    assert calmar_ratio([100.0]) == 0.0


# ─── compute_all ─────────────────────────────────────────────────────────────

def test_compute_all_keys():
    result = compute_all(RISING)
    expected_keys = {"total_return", "ann_return", "volatility",
                     "max_drawdown", "sharpe", "sortino", "calmar"}
    assert expected_keys == set(result.keys())


def test_compute_all_rising():
    result = compute_all(RISING)
    assert result["total_return"] > 0
    assert result["sharpe"] > 0
    assert result["max_drawdown"] >= -0.01


def test_compute_all_flat():
    result = compute_all(FLAT)
    assert result["total_return"] == 0.0
    assert result["sharpe"] == 0.0
    assert result["max_drawdown"] == 0.0


def test_compute_all_values_are_finite():
    result = compute_all(VOLATILE)
    for key, val in result.items():
        if val != float("inf"):
            assert math.isfinite(val), f"{key}={val} no es finito"
