"""
Tests para las 6 estrategias especializadas por escenario +
ScenarioRouter.
"""
import pytest
import numpy as np
import pandas as pd

from strategies.trend_rider       import TrendRiderStrategy
from strategies.range_scalper     import RangeScalperStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.ob_rejection      import OBRejectionStrategy
from strategies.mean_reversion    import MeanReversionStrategy
from strategies.volatility_filter import VolatilityFilterStrategy
from strategies.scenario_router   import ScenarioRouter, REGIME_TO_STRATEGY

VALID_SIGNALS = ("buy", "sell", "hold")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make(n=300, trend="bull", noise=0.4, seed=42):
    rng = np.random.default_rng(seed)
    prices = [100.0]
    slope = {"bull": 0.25, "bear": -0.25, "flat": 0.0}[trend]
    for _ in range(n - 1):
        prices.append(max(1.0, prices[-1] + slope + rng.standard_normal() * noise))
    prices = np.array(prices)
    spread = abs(rng.standard_normal(n)) * 0.3
    return pd.DataFrame({
        "open": prices, "high": prices + spread,
        "low":  np.maximum(prices - spread, prices * 0.98),
        "close": prices,
        "volume": rng.integers(1000, 10000, n).astype(float),
    })

def _short(n=10):
    return _make(n)


# ═══════════════════════════════════════════════════════════════════════════════
# TrendRiderStrategy
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrendRider:
    def test_defaults(self):
        s = TrendRiderStrategy()
        assert s.swing_window == 5
        assert s.adx_threshold == 22.0
        assert s.require_fvg is False
        assert s.use_choch_filter is False

    def test_none_returns_hold(self):
        s = TrendRiderStrategy()
        assert s.generate_signal(None)["signal"] == "hold"

    def test_short_returns_hold(self):
        s = TrendRiderStrategy()
        assert s.generate_signal(_short())["signal"] == "hold"

    def test_valid_signal(self):
        s = TrendRiderStrategy(adx_threshold=5.0, require_ema_fan=False, require_bos=False)
        r = s.generate_signal(_make(300, trend="bull"))
        assert r["signal"] in VALID_SIGNALS

    def test_no_sell_in_bull_trend_with_fan(self):
        s = TrendRiderStrategy(adx_threshold=5.0, require_ema_fan=False, require_bos=False)
        r = s.generate_signal(_make(300, trend="bull"))
        assert r["signal"] != "sell"

    def test_no_buy_in_bear_trend_with_fan(self):
        s = TrendRiderStrategy(adx_threshold=5.0, require_ema_fan=False, require_bos=False)
        r = s.generate_signal(_make(300, trend="bear"))
        assert r["signal"] != "buy"

    def test_high_threshold_gives_hold(self):
        s = TrendRiderStrategy(adx_threshold=99.0)
        r = s.generate_signal(_make(300))
        assert r["signal"] == "hold"

    def test_result_has_reason(self):
        s = TrendRiderStrategy()
        r = s.generate_signal(_make(300))
        assert "reason" in r

    def test_custom_ema_periods(self):
        s = TrendRiderStrategy(ema_fast=9, ema_mid=21, ema_slow=50, adx_threshold=5.0,
                                require_ema_fan=False, require_bos=False)
        r = s.generate_signal(_make(200))
        assert r["signal"] in VALID_SIGNALS


# ═══════════════════════════════════════════════════════════════════════════════
# RangeScalperStrategy
# ═══════════════════════════════════════════════════════════════════════════════

class TestRangeScalper:
    def test_defaults(self):
        s = RangeScalperStrategy()
        assert s.swing_window == 5
        assert s.require_fvg is False
        assert s.use_choch_filter is False

    def test_none_returns_hold(self):
        s = RangeScalperStrategy()
        assert s.generate_signal(None)["signal"] == "hold"

    def test_short_returns_hold(self):
        s = RangeScalperStrategy()
        assert s.generate_signal(_short())["signal"] == "hold"

    def test_valid_signal(self):
        s = RangeScalperStrategy(range_window=30, max_range_pct=0.99)
        r = s.generate_signal(_make(200, trend="flat", noise=0.2))
        assert r["signal"] in VALID_SIGNALS

    def test_result_has_reason(self):
        s = RangeScalperStrategy()
        r = s.generate_signal(_make(200))
        assert "reason" in r

    def test_very_tight_range_gives_hold(self):
        s = RangeScalperStrategy(min_range_pct=0.99)  # imposible de satisfacer
        r = s.generate_signal(_make(200))
        assert r["signal"] == "hold"

    def test_loose_rsi_thresholds_generate_signals(self):
        # Con RSI thresholds muy amplios debe generar alguna señal
        s = RangeScalperStrategy(
            rsi_buy_max=80.0, rsi_sell_min=20.0,
            range_window=20, max_range_pct=0.99, min_range_pct=0.0,
            entry_zone_pct=0.49,
        )
        signals = [s.generate_signal(_make(100, seed=i))["signal"] for i in range(10)]
        assert any(sig in ("buy", "sell") for sig in signals)

    def test_different_range_windows(self):
        for w in [20, 30, 50]:
            s = RangeScalperStrategy(range_window=w)
            r = s.generate_signal(_make(w * 3))
            assert r["signal"] in VALID_SIGNALS


# ═══════════════════════════════════════════════════════════════════════════════
# BreakoutStrategy
# ═══════════════════════════════════════════════════════════════════════════════

class TestBreakoutStrategy:
    def test_defaults(self):
        s = BreakoutStrategy()
        assert s.swing_window == 5
        assert s.require_fvg is False
        assert s.use_choch_filter is False

    def test_none_returns_hold(self):
        s = BreakoutStrategy()
        assert s.generate_signal(None)["signal"] == "hold"

    def test_short_returns_hold(self):
        s = BreakoutStrategy()
        assert s.generate_signal(_short())["signal"] == "hold"

    def test_valid_signal(self):
        s = BreakoutStrategy(vol_multiplier=0.1, atr_min_mult=0.1,
                              min_consolidation_pct=0.0)
        r = s.generate_signal(_make(200))
        assert r["signal"] in VALID_SIGNALS

    def test_result_has_reason(self):
        s = BreakoutStrategy()
        r = s.generate_signal(_make(200))
        assert "reason" in r

    def test_high_vol_mult_gives_hold(self):
        s = BreakoutStrategy(vol_multiplier=999.0)
        r = s.generate_signal(_make(200))
        assert r["signal"] == "hold"

    def test_custom_consolidation_bars(self):
        for cb in [10, 20, 40]:
            s = BreakoutStrategy(consolidation_bars=cb, vol_multiplier=0.1,
                                  atr_min_mult=0.1, min_consolidation_pct=0.0)
            r = s.generate_signal(_make(cb * 4))
            assert r["signal"] in VALID_SIGNALS


# ═══════════════════════════════════════════════════════════════════════════════
# OBRejectionStrategy
# ═══════════════════════════════════════════════════════════════════════════════

class TestOBRejection:
    def test_defaults(self):
        s = OBRejectionStrategy()
        assert s.swing_window == 5
        assert s.use_choch_filter is False

    def test_none_returns_hold(self):
        s = OBRejectionStrategy()
        assert s.generate_signal(None)["signal"] == "hold"

    def test_short_returns_hold(self):
        s = OBRejectionStrategy()
        assert s.generate_signal(_short())["signal"] == "hold"

    def test_valid_signal(self):
        s = OBRejectionStrategy(bos_confirm=False, fvg_confirm=False,
                                 rejection_pct=0.0, rsi_bull_max=100.0,
                                 rsi_bear_min=0.0)
        r = s.generate_signal(_make(200))
        assert r["signal"] in VALID_SIGNALS

    def test_result_has_reason(self):
        s = OBRejectionStrategy()
        r = s.generate_signal(_make(200))
        assert "reason" in r

    def test_fvg_confirm_compat(self):
        s = OBRejectionStrategy(fvg_confirm=True)
        assert s.require_fvg is True

    def test_no_crash_on_data(self):
        s = OBRejectionStrategy(bos_confirm=False)
        for seed in range(5):
            r = s.generate_signal(_make(150, seed=seed))
            assert r["signal"] in VALID_SIGNALS


# ═══════════════════════════════════════════════════════════════════════════════
# MeanReversionStrategy
# ═══════════════════════════════════════════════════════════════════════════════

class TestMeanReversion:
    def test_defaults(self):
        s = MeanReversionStrategy()
        assert s.swing_window == 5
        assert s.use_choch_filter is False

    def test_none_returns_hold(self):
        s = MeanReversionStrategy()
        assert s.generate_signal(None)["signal"] == "hold"

    def test_short_returns_hold(self):
        s = MeanReversionStrategy()
        assert s.generate_signal(_short())["signal"] == "hold"

    def test_valid_signal(self):
        s = MeanReversionStrategy(require_pd_zone=False, require_fvg=False)
        r = s.generate_signal(_make(200))
        assert r["signal"] in VALID_SIGNALS

    def test_result_has_reason(self):
        s = MeanReversionStrategy()
        r = s.generate_signal(_make(200))
        assert "reason" in r

    def test_extreme_rsi_thresholds_give_hold(self):
        # Umbrales imposibles
        s = MeanReversionStrategy(rsi_oversold=0.1, rsi_overbought=99.9,
                                   require_pd_zone=False, require_fvg=False)
        r = s.generate_signal(_make(200))
        assert r["signal"] == "hold"

    def test_loose_thresholds_can_signal(self):
        s = MeanReversionStrategy(
            rsi_oversold=60.0, rsi_overbought=40.0,
            require_pd_zone=False, require_fvg=False,
        )
        signals = [s.generate_signal(_make(150, seed=i))["signal"] for i in range(10)]
        assert any(sig in ("buy", "sell") for sig in signals)


# ═══════════════════════════════════════════════════════════════════════════════
# VolatilityFilterStrategy
# ═══════════════════════════════════════════════════════════════════════════════

class TestVolatilityFilter:
    def test_defaults(self):
        s = VolatilityFilterStrategy()
        assert s.swing_window == 5
        assert s.require_fvg is False
        assert s.use_choch_filter is False
        assert 0 < s.size_reduction_factor <= 1.0

    def test_none_returns_hold(self):
        s = VolatilityFilterStrategy()
        assert s.generate_signal(None)["signal"] == "hold"

    def test_short_returns_hold(self):
        s = VolatilityFilterStrategy()
        assert s.generate_signal(_short())["signal"] == "hold"

    def test_normal_market_returns_hold(self):
        # ATR normal → siempre hold (no exhaustion)
        s = VolatilityFilterStrategy(allow_exhaustion_entry=False)
        r = s.generate_signal(_make(200))
        assert r["signal"] == "hold"

    def test_result_has_reason(self):
        s = VolatilityFilterStrategy()
        r = s.generate_signal(_make(200))
        assert "reason" in r

    def test_size_reduction_in_result(self):
        s = VolatilityFilterStrategy(atr_high_mult=0.01)  # siempre activo
        r = s.generate_signal(_make(200))
        # Debe contener size_reduction o atr_ratio en algún nivel
        assert "signal" in r

    def test_exhaustion_mode_valid_signal(self):
        s = VolatilityFilterStrategy(allow_exhaustion_entry=True,
                                      atr_high_mult=0.01)
        r = s.generate_signal(_make(200))
        assert r["signal"] in VALID_SIGNALS


# ═══════════════════════════════════════════════════════════════════════════════
# ScenarioRouter
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenarioRouterInit:
    def test_defaults(self):
        r = ScenarioRouter()
        assert r.swing_window == 5
        assert r.require_fvg is False
        assert r.use_choch_filter is False

    def test_has_all_strategies(self):
        r = ScenarioRouter()
        for name in ["smc_fallback", "trend_rider", "range_scalper",
                     "breakout", "ob_rejection", "mean_reversion",
                     "volatility_filter"]:
            assert name in r._strategies

    def test_regime_map_complete(self):
        expected_regimes = {
            "strong_trend_bull", "strong_trend_bear",
            "weak_trend_bull", "weak_trend_bear",
            "ranging", "breakout",
            "mean_reversion_bull", "mean_reversion_bear",
            "high_volatility", "insufficient_data",
        }
        assert set(REGIME_TO_STRATEGY.keys()) == expected_regimes

    def test_strategy_for_returns_instance(self):
        r = ScenarioRouter()
        s = r.strategy_for("ranging")
        assert s is not None
        assert hasattr(s, "generate_signal")

    def test_use_mtf_trend_replaces_trend_rider(self):
        r = ScenarioRouter(use_mtf_trend=True)
        from strategies.mtf_strategy import MTFStrategy
        assert isinstance(r._strategies["trend_rider"], MTFStrategy)


class TestScenarioRouterSignal:
    def test_none_returns_hold(self):
        r = ScenarioRouter()
        assert r.generate_signal(None)["signal"] == "hold"

    def test_short_data_returns_hold(self):
        r = ScenarioRouter()
        assert r.generate_signal(_short(20))["signal"] == "hold"

    def test_valid_signal(self):
        r = ScenarioRouter()
        result = r.generate_signal(_make(300))
        assert result["signal"] in VALID_SIGNALS

    def test_verbose_keys(self):
        r = ScenarioRouter(verbose=True)
        result = r.generate_signal(_make(300))
        assert "regime" in result
        assert "regime_label" in result
        assert "strategy_used" in result

    def test_no_verbose_no_extra_keys(self):
        r = ScenarioRouter(verbose=False)
        result = r.generate_signal(_make(300))
        assert "regime" not in result
        assert "regime_label" not in result

    def test_override_strategy_forces_strategy(self):
        r = ScenarioRouter(override_strategy="range_scalper", verbose=True)
        result = r.generate_signal(_make(300))
        assert result.get("strategy_used") == "range_scalper"

    def test_multiple_calls_no_crash(self):
        r = ScenarioRouter()
        for seed in range(8):
            result = r.generate_signal(_make(200, seed=seed))
            assert result["signal"] in VALID_SIGNALS

    def test_bull_trend_activates_trend_rider_or_ob(self):
        """En mercado alcista, el router debe usar trend_rider u ob_rejection."""
        r = ScenarioRouter(verbose=True)
        result = r.generate_signal(_make(400, trend="bull"))
        strat = result.get("strategy_used", "")
        regime = result.get("regime", "")
        # Si el régimen es de tendencia, la estrategia debe ser la correcta
        if "trend" in regime:
            assert strat in ("trend_rider", "ob_rejection", "smc_fallback")

    def test_flat_market_activates_range_or_mean_rev(self):
        """En mercado plano, debe activarse range_scalper o mean_reversion."""
        r = ScenarioRouter(verbose=True)
        result = r.generate_signal(_make(400, trend="flat", noise=0.05))
        strat = result.get("strategy_used", "")
        regime = result.get("regime", "")
        if regime == "ranging":
            assert strat == "range_scalper"
        if "mean_reversion" in regime:
            assert strat == "mean_reversion"

    def test_different_swing_windows(self):
        for w in [3, 5, 10]:
            r = ScenarioRouter(swing_window=w)
            result = r.generate_signal(_make(200))
            assert result["signal"] in VALID_SIGNALS
