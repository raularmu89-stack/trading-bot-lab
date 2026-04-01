"""
Tests para KellySizer — Kelly criterion position sizing.
"""

import pytest
from strategies.kelly_sizer import KellySizer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trades(wins, losses):
    """Genera historial de trades con wins ganancias y losses pérdidas."""
    history = []
    for _ in range(wins):
        history.append({"pnl": 100.0, "win": True})
    for _ in range(losses):
        history.append({"pnl": -60.0, "win": False})
    return history


def _float_trades(wins, losses):
    """Historial como lista de floats."""
    return [100.0] * wins + [-60.0] * losses


# ── Fallback antes de min_trades ──────────────────────────────────────────────

class TestFallback:
    def test_few_trades_returns_fixed_pct(self):
        ks = KellySizer(min_trades=20, fixed_pct=0.02)
        result = ks.kelly_fraction(_trades(5, 5))
        assert result == 0.02

    def test_exactly_min_trades_uses_kelly(self):
        ks = KellySizer(min_trades=10, fixed_pct=0.02)
        history = _trades(6, 4)  # 10 trades exactos
        result = ks.kelly_fraction(history)
        # Con WR=0.6, b=100/60≈1.667 → f*=(0.6*1.667-0.4)/1.667 ≈ 0.36
        assert result != 0.02
        assert result > 0

    def test_empty_history_returns_fixed_pct(self):
        ks = KellySizer(min_trades=5, fixed_pct=0.03)
        assert ks.kelly_fraction([]) == 0.03


# ── Cálculo de kelly_fraction ─────────────────────────────────────────────────

class TestKellyFraction:
    def test_positive_edge(self):
        ks = KellySizer(variant="full_kelly", min_trades=10)
        history = _trades(7, 3)  # WR=0.7, b=100/60≈1.667
        f = ks.kelly_fraction(history)
        # f* = (0.7*1.667 - 0.3) / 1.667 = (1.1669 - 0.3) / 1.667 ≈ 0.52
        assert 0.40 < f < 0.65

    def test_half_kelly(self):
        ks = KellySizer(variant="half_kelly", min_trades=10)
        full_ks = KellySizer(variant="full_kelly", min_trades=10)
        history = _trades(7, 3)
        assert ks.kelly_fraction(history) == pytest.approx(
            full_ks.kelly_fraction(history) / 2, rel=1e-6
        )

    def test_quarter_kelly(self):
        ks = KellySizer(variant="quarter_kelly", min_trades=10)
        full_ks = KellySizer(variant="full_kelly", min_trades=10)
        history = _trades(7, 3)
        assert ks.kelly_fraction(history) == pytest.approx(
            full_ks.kelly_fraction(history) / 4, rel=1e-6
        )

    def test_fixed_pct_variant(self):
        ks = KellySizer(variant="fixed_pct", fixed_pct=0.05, min_trades=10)
        history = _trades(7, 3)
        assert ks.kelly_fraction(history) == 0.05

    def test_negative_edge_returns_zero_or_less(self):
        """WR muy baja → Kelly negativo = no edge → 0.0"""
        ks = KellySizer(variant="full_kelly", min_trades=10)
        # WR=0.2 con avg_win=100, avg_loss=60 → f* < 0
        history = _trades(2, 8)
        f = ks.kelly_fraction(history)
        assert f <= 0.0

    def test_float_list_history(self):
        ks = KellySizer(variant="full_kelly", min_trades=10)
        history_dict = _trades(7, 3)
        history_float = _float_trades(7, 3)
        assert ks.kelly_fraction(history_dict) == pytest.approx(
            ks.kelly_fraction(history_float), rel=1e-6
        )

    def test_all_wins_returns_fixed_pct(self):
        ks = KellySizer(min_trades=10, fixed_pct=0.02)
        history = [{"pnl": 100.0}] * 20
        # No losses → fallback
        assert ks.kelly_fraction(history) == 0.02

    def test_all_losses_returns_fixed_pct(self):
        ks = KellySizer(min_trades=10, fixed_pct=0.02)
        history = [{"pnl": -50.0}] * 20
        assert ks.kelly_fraction(history) == 0.02


# ── position_fraction (min/max clamp) ────────────────────────────────────────

class TestPositionFraction:
    def test_clamp_max(self):
        ks = KellySizer(variant="full_kelly", max_fraction=0.10, min_trades=10)
        history = _trades(9, 1)  # WR altísimo → Kelly > max
        f = ks.position_fraction(history)
        assert f <= 0.10

    def test_clamp_min(self):
        ks = KellySizer(variant="full_kelly", min_fraction=0.01, min_trades=10)
        # Edge negativo → Kelly ≤ 0 → se clampea a min
        history = _trades(2, 8)
        f = ks.position_fraction(history)
        assert f >= 0.01

    def test_within_bounds(self):
        ks = KellySizer(
            variant="half_kelly", min_fraction=0.005,
            max_fraction=0.25, min_trades=10
        )
        history = _trades(6, 4)
        f = ks.position_fraction(history)
        assert 0.005 <= f <= 0.25


# ── position_size ─────────────────────────────────────────────────────────────

class TestPositionSize:
    def test_basic_calculation(self):
        ks = KellySizer(variant="half_kelly", min_trades=10)
        history = _trades(6, 4)
        result = ks.position_size(balance=1000, entry=45000, sl=44100, trade_history=history)
        assert "fraction" in result
        assert "risk_amount" in result
        assert "units" in result
        assert "position_value" in result

    def test_risk_amount_equals_balance_times_fraction(self):
        ks = KellySizer(variant="fixed_pct", fixed_pct=0.02, min_trades=1)
        history = _trades(5, 5)
        result = ks.position_size(balance=10000, entry=50000, sl=49000, trade_history=history)
        expected_risk = 10000 * result["fraction"]
        assert result["risk_amount"] == pytest.approx(expected_risk, rel=1e-4)

    def test_sl_equals_entry_returns_zero(self):
        ks = KellySizer(min_trades=10)
        history = _trades(6, 4)
        result = ks.position_size(balance=1000, entry=45000, sl=45000, trade_history=history)
        assert result["units"] == 0
        assert result["position_value"] == 0

    def test_units_correct_for_risk(self):
        """units * entry * sl_distance ≈ risk_amount"""
        ks = KellySizer(variant="fixed_pct", fixed_pct=0.02, min_trades=1)
        history = _trades(5, 5)
        balance = 10000
        entry = 50000
        sl = 49000
        result = ks.position_size(balance=balance, entry=entry, sl=sl, trade_history=history)
        sl_dist = abs(entry - sl) / entry
        implied_risk = result["units"] * entry * sl_dist
        assert implied_risk == pytest.approx(result["risk_amount"], rel=1e-4)

    def test_position_value_equals_units_times_entry(self):
        ks = KellySizer(variant="half_kelly", min_trades=10)
        history = _trades(6, 4)
        result = ks.position_size(balance=5000, entry=30000, sl=29500, trade_history=history)
        assert result["position_value"] == pytest.approx(
            result["units"] * 30000, rel=1e-4
        )


# ── stats ─────────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_keys(self):
        ks = KellySizer(min_trades=10)
        history = _trades(6, 4)
        s = ks.stats(history)
        for key in ["win_rate", "avg_win", "avg_loss", "profit_factor",
                    "full_kelly", "half_kelly", "quarter_kelly", "applied_fraction"]:
            assert key in s

    def test_win_rate_correct(self):
        ks = KellySizer(min_trades=10)
        history = _trades(7, 3)
        s = ks.stats(history)
        assert s["win_rate"] == pytest.approx(0.7, rel=1e-3)

    def test_half_kelly_is_half_full(self):
        ks = KellySizer(min_trades=10)
        history = _trades(6, 4)
        s = ks.stats(history)
        assert s["half_kelly"] == pytest.approx(s["full_kelly"] / 2, rel=1e-6)

    def test_quarter_kelly_is_quarter_full(self):
        ks = KellySizer(min_trades=10)
        history = _trades(6, 4)
        s = ks.stats(history)
        assert s["quarter_kelly"] == pytest.approx(s["full_kelly"] / 4, rel=1e-6)

    def test_empty_stats(self):
        ks = KellySizer()
        s = ks.stats([])
        assert s == {}

    def test_single_trade_stats(self):
        ks = KellySizer()
        s = ks.stats([{"pnl": 100}])
        # len < 2 → {}
        assert s == {}
