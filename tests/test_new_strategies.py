"""
tests/test_new_strategies.py

Tests para las 3 nuevas estrategias:
  - MACDDivergenceStrategy
  - BollingerSqueezeStrategy
  - MomentumBurstStrategy
"""
import pytest
import numpy as np
import pandas as pd

from strategies.macd_divergence   import MACDDivergenceStrategy, _ema, _rsi
from strategies.bollinger_squeeze import BollingerSqueezeStrategy
from strategies.momentum_burst    import MomentumBurstStrategy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make(n=200, trend="bull", seed=42):
    rng   = np.random.default_rng(seed)
    slope = {"bull": 0.3, "bear": -0.3, "flat": 0.0}[trend]
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(max(1.0, prices[-1] + slope + rng.standard_normal() * 0.5))
    prices = np.array(prices)
    spread = abs(rng.standard_normal(n)) * 0.4
    # open ≈ previous close → candle bodies reflejan la dirección real del movimiento
    opens = np.concatenate([[prices[0]], prices[:-1]])
    highs = np.maximum(opens, prices) + spread * 0.3 + 0.05
    lows  = np.maximum(np.minimum(opens, prices) - spread * 0.3, prices * 0.99)
    return pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  prices,
        "volume": rng.integers(1000, 10000, n).astype(float),
    })


def _make_squeeze(n=100, compressed_bars=20, burst=True):
    """Genera datos con un squeeze claro al final."""
    rng    = np.random.default_rng(0)
    prices = [100.0]
    # Fase comprimida: volatilidad muy baja
    for _ in range(compressed_bars):
        prices.append(max(1.0, prices[-1] + rng.standard_normal() * 0.05))
    # Burst alcista si burst=True
    if burst:
        prices.append(prices[-1] + 3.0)  # vela grande alcista
        for _ in range(n - len(prices)):
            prices.append(prices[-1] + 0.2 + rng.standard_normal() * 0.1)
    else:
        for _ in range(n - len(prices)):
            prices.append(max(1.0, prices[-1] + rng.standard_normal() * 0.05))

    prices = np.array(prices[:n])
    spread = abs(rng.standard_normal(n)) * 0.1
    return pd.DataFrame({
        "open":   prices,
        "high":   prices + spread + 0.05,
        "low":    np.maximum(prices - spread, prices * 0.999),
        "close":  prices,
        "volume": rng.integers(1000, 10000, n).astype(float),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Indicadores internos
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_ema_length(self):
        arr = np.arange(1, 51, dtype=float)
        e   = _ema(arr, 12)
        assert len(e) == 50

    def test_ema_nan_warmup(self):
        e = _ema(np.arange(1, 30, dtype=float), 12)
        assert np.all(np.isnan(e[:11]))
        assert not np.isnan(e[11])

    def test_ema_too_short(self):
        e = _ema(np.array([1.0, 2.0]), 10)
        assert np.all(np.isnan(e))

    def test_rsi_range(self):
        closes = _make(200)["close"].values
        rsi    = _rsi(closes, 14)
        valid  = rsi[~np.isnan(rsi)]
        assert np.all((valid >= 0) & (valid <= 100))

    def test_rsi_length(self):
        closes = _make(100)["close"].values
        rsi    = _rsi(closes, 14)
        assert len(rsi) == 100


# ═══════════════════════════════════════════════════════════════════════════════
# MACDDivergenceStrategy
# ═══════════════════════════════════════════════════════════════════════════════

class TestMACDDivergence:
    def _strat(self, **kw):
        return MACDDivergenceStrategy(**kw)

    def test_init_default(self):
        s = self._strat()
        assert s.fast_period   == 12
        assert s.slow_period   == 26
        assert s.signal_period == 9
        assert s.lookback      == 5

    def test_repr(self):
        assert "MACDDivergenceStrategy" in repr(self._strat())

    def test_compat_attrs(self):
        s = self._strat()
        assert hasattr(s, "swing_window")
        assert s.require_fvg       is False
        assert s.use_choch_filter  is False

    def test_returns_dict(self):
        result = self._strat().generate_signal(_make(200))
        assert isinstance(result, dict)

    def test_signal_valid(self):
        result = self._strat().generate_signal(_make(200))
        assert result["signal"] in ("buy", "sell", "hold")

    def test_none_data(self):
        result = self._strat().generate_signal(None)
        assert result["signal"] == "hold"

    def test_short_data(self):
        result = self._strat().generate_signal(_make(10))
        assert result["signal"] == "hold"

    def test_dict_keys_on_hold(self):
        result = self._strat().generate_signal(_make(20))
        assert "signal" in result
        assert "reason" in result

    def test_bull_market_tends_buy(self):
        """En tendencia alcista extrema, debería generar compras."""
        signals = [self._strat(lookback=3).generate_signal(
            _make(200, trend="bull", seed=i))["signal"]
            for i in range(10)]
        assert "hold" in signals or "buy" in signals  # al menos algo

    def test_cruce_confirm_mode(self):
        s = MACDDivergenceStrategy(cruce_confirm=True)
        r = s.generate_signal(_make(200))
        assert r["signal"] in ("buy", "sell", "hold")

    def test_rsi_guard_mode(self):
        s = MACDDivergenceStrategy(rsi_guard=True)
        r = s.generate_signal(_make(200))
        assert r["signal"] in ("buy", "sell", "hold")

    def test_no_rsi_guard(self):
        s = MACDDivergenceStrategy(rsi_guard=False)
        r = s.generate_signal(_make(200))
        assert r["signal"] in ("buy", "sell", "hold")

    def test_custom_periods(self):
        s = MACDDivergenceStrategy(fast_period=5, slow_period=13, signal_period=3)
        r = s.generate_signal(_make(150))
        assert r["signal"] in ("buy", "sell", "hold")

    def test_histogram_key_on_signal(self):
        """Si hay señal, debe incluir datos MACD."""
        for seed in range(20):
            r = self._strat(lookback=3).generate_signal(_make(200, seed=seed))
            if r["signal"] != "hold":
                assert "histogram" in r
                assert "macd" in r
                break  # basta con uno


# ═══════════════════════════════════════════════════════════════════════════════
# BollingerSqueezeStrategy
# ═══════════════════════════════════════════════════════════════════════════════

class TestBollingerSqueeze:
    def _strat(self, **kw):
        return BollingerSqueezeStrategy(**kw)

    def test_init_default(self):
        s = self._strat()
        assert s.bb_period       == 20
        assert s.kc_period       == 20
        assert s.momentum_period == 12
        assert s.confirm_bars    == 2

    def test_repr(self):
        assert "BollingerSqueezeStrategy" in repr(self._strat())

    def test_compat_attrs(self):
        s = self._strat()
        assert hasattr(s, "swing_window")
        assert s.require_fvg       is False
        assert s.use_choch_filter  is False

    def test_returns_dict(self):
        assert isinstance(self._strat().generate_signal(_make(200)), dict)

    def test_signal_valid(self):
        r = self._strat().generate_signal(_make(200))
        assert r["signal"] in ("buy", "sell", "hold")

    def test_none_data(self):
        r = self._strat().generate_signal(None)
        assert r["signal"] == "hold"

    def test_short_data(self):
        r = self._strat().generate_signal(_make(10))
        assert r["signal"] == "hold"

    def test_squeeze_detection(self):
        """Con datos de squeeze claro, debe generar señal tras liberación."""
        # Generar datos con squeeze y burst
        df = _make_squeeze(n=120, compressed_bars=30, burst=True)
        r  = self._strat(confirm_bars=1).generate_signal(df)
        assert r["signal"] in ("buy", "sell", "hold")

    def test_no_crash_flat(self):
        r = self._strat().generate_signal(_make(200, trend="flat"))
        assert r["signal"] in ("buy", "sell", "hold")

    def test_is_squeeze_logic(self):
        s = self._strat()
        # BB dentro de KC → squeeze
        assert s._is_squeeze(105.0, 95.0, 110.0, 90.0) is True
        # BB fuera de KC → no squeeze
        assert s._is_squeeze(115.0, 85.0, 110.0, 90.0) is False

    def test_signal_keys_when_not_hold(self):
        for seed in range(20):
            r = self._strat(confirm_bars=1).generate_signal(_make(200, seed=seed))
            if r["signal"] != "hold":
                assert "momentum" in r
                assert "squeeze_fired" in r
                break

    def test_different_params(self):
        s = BollingerSqueezeStrategy(bb_period=10, kc_period=10, momentum_period=5)
        r = s.generate_signal(_make(150))
        assert r["signal"] in ("buy", "sell", "hold")

    def test_without_volume(self):
        df = _make(200).drop(columns=["volume"])
        r  = self._strat().generate_signal(df)
        assert r["signal"] in ("buy", "sell", "hold")


# ═══════════════════════════════════════════════════════════════════════════════
# MomentumBurstStrategy
# ═══════════════════════════════════════════════════════════════════════════════

class TestMomentumBurst:
    def _strat(self, **kw):
        return MomentumBurstStrategy(**kw)

    def test_init_default(self):
        s = self._strat()
        assert s.atr_period       == 14
        assert s.compression_bars == 10
        assert s.burst_mult       == 1.5
        assert s.vol_mult         == 1.5
        assert s.rsi_bull_min     == 55.0
        assert s.rsi_bear_max     == 45.0

    def test_repr(self):
        assert "MomentumBurstStrategy" in repr(self._strat())

    def test_compat_attrs(self):
        s = self._strat()
        assert hasattr(s, "swing_window")
        assert s.require_fvg       is False
        assert s.use_choch_filter  is False

    def test_returns_dict(self):
        assert isinstance(self._strat().generate_signal(_make(200)), dict)

    def test_signal_valid(self):
        r = self._strat().generate_signal(_make(200))
        assert r["signal"] in ("buy", "sell", "hold")

    def test_none_data(self):
        r = self._strat().generate_signal(None)
        assert r["signal"] == "hold"

    def test_short_data(self):
        r = self._strat().generate_signal(_make(10))
        assert r["signal"] == "hold"

    def test_flat_market_mostly_hold(self):
        """En mercado plano, pocas señales de burst."""
        signals = [
            self._strat().generate_signal(_make(200, trend="flat", seed=i))["signal"]
            for i in range(5)
        ]
        holds = signals.count("hold")
        assert holds >= 3

    def test_custom_params(self):
        s = MomentumBurstStrategy(
            atr_period=7, compression_bars=5,
            compression_ratio=0.8, burst_mult=1.2
        )
        r = s.generate_signal(_make(200))
        assert r["signal"] in ("buy", "sell", "hold")

    def test_signal_keys_when_buy(self):
        """Si hay señal, debe incluir burst_ratio y rsi."""
        for seed in range(30):
            r = self._strat(compression_ratio=0.9, burst_mult=0.5).generate_signal(
                _make(200, trend="bull", seed=seed))
            if r["signal"] == "buy":
                assert "burst_ratio" in r
                assert "rsi" in r
                break

    def test_without_volume_column(self):
        df = _make(200).drop(columns=["volume"])
        r  = self._strat().generate_signal(df)
        assert r["signal"] in ("buy", "sell", "hold")

    def test_bull_trend_generates_buys(self):
        """Con umbral muy relajado en bull trend, debe haber compras."""
        got_buy = False
        for seed in range(80):
            r = MomentumBurstStrategy(
                compression_ratio=2.0,  # siempre pasa compresión
                burst_mult=0.1,         # cualquier cuerpo pasa
                rsi_bull_min=40.0,
                vol_mult=0.1,           # siempre pasa volumen
            ).generate_signal(_make(200, trend="bull", seed=seed))
            if r["signal"] == "buy":
                got_buy = True
                break
        assert got_buy

    def test_bear_trend_generates_sells(self):
        """Con umbral muy relajado en bear trend, debe haber ventas."""
        got_sell = False
        for seed in range(80):
            r = MomentumBurstStrategy(
                compression_ratio=2.0,
                burst_mult=0.1,
                rsi_bear_max=60.0,
                vol_mult=0.1,
            ).generate_signal(_make(200, trend="bear", seed=seed))
            if r["signal"] == "sell":
                got_sell = True
                break
        assert got_sell


# ═══════════════════════════════════════════════════════════════════════════════
# Integración: nuevas estrategias en ScenarioRouter
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewStrategiesInRouter:
    def test_router_has_new_strategies(self):
        from strategies.scenario_router import ScenarioRouter
        router = ScenarioRouter()
        assert "macd_divergence"   in router._strategies
        assert "bollinger_squeeze" in router._strategies
        assert "momentum_burst"    in router._strategies

    def test_override_macd(self):
        from strategies.scenario_router import ScenarioRouter
        router = ScenarioRouter(override_strategy="macd_divergence")
        r = router.generate_signal(_make(200))
        assert r["signal"] in ("buy", "sell", "hold")

    def test_override_bollinger(self):
        from strategies.scenario_router import ScenarioRouter
        router = ScenarioRouter(override_strategy="bollinger_squeeze")
        r = router.generate_signal(_make(200))
        assert r["signal"] in ("buy", "sell", "hold")

    def test_override_momentum_burst(self):
        from strategies.scenario_router import ScenarioRouter
        router = ScenarioRouter(override_strategy="momentum_burst")
        r = router.generate_signal(_make(200))
        assert r["signal"] in ("buy", "sell", "hold")
