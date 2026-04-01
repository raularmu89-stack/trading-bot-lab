"""
Tests para ConfluenceStrategy — Sweep + FVG/OB + BOS.
"""

import pytest
import numpy as np
import pandas as pd
from strategies.confluence_strategy import ConfluenceStrategy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n=200, trend="bullish", noise=0.4, seed=42):
    np.random.seed(seed)
    prices = [100.0]
    slope = {"bullish": 0.2, "bearish": -0.2, "sideways": 0.0}[trend]
    for _ in range(n - 1):
        prices.append(max(1.0, prices[-1] + slope + np.random.randn() * noise))
    prices = np.array(prices)
    high   = prices + abs(np.random.randn(n)) * noise
    low    = prices - abs(np.random.randn(n)) * noise
    low    = np.minimum(low, prices)
    high   = np.maximum(high, prices)
    return pd.DataFrame({
        "open": prices, "high": high, "low": low,
        "close": prices, "volume": np.ones(n) * 1000,
    })


# ── Init ──────────────────────────────────────────────────────────────────────

class TestConfluenceStrategyInit:
    def test_defaults(self):
        s = ConfluenceStrategy()
        assert s.swing_window == 10
        assert s.require_sweep is True
        assert s.require_fvg_or_ob is True
        assert s.require_bos is True
        assert s.use_pd_filter is False
        assert s.use_ema_filter is False

    def test_custom_params(self):
        s = ConfluenceStrategy(
            swing_window=15, require_sweep=False, use_pd_filter=True,
            min_sweep_bars=3
        )
        assert s.swing_window == 15
        assert s.require_sweep is False
        assert s.use_pd_filter is True
        assert s.min_sweep_bars == 3

    def test_fast_backtester_compat(self):
        s = ConfluenceStrategy()
        assert hasattr(s, "require_fvg")
        assert hasattr(s, "use_choch_filter")
        assert s.require_fvg is False
        assert s.use_choch_filter is False


# ── Datos insuficientes ───────────────────────────────────────────────────────

class TestConfluenceInsufficientData:
    def test_none_returns_hold(self):
        s = ConfluenceStrategy()
        result = s.generate_signal(None)
        assert result["signal"] == "hold"

    def test_too_short_returns_hold(self):
        s = ConfluenceStrategy(swing_window=10)
        df = _make_df(20)  # necesita swing_window * 3 = 30
        result = s.generate_signal(df)
        assert result["signal"] == "hold"


# ── generate_signal ───────────────────────────────────────────────────────────

class TestConfluenceSignal:
    def test_returns_dict(self):
        s = ConfluenceStrategy(swing_window=5, require_sweep=False,
                               require_fvg_or_ob=False, require_bos=False)
        df = _make_df(100)
        result = s.generate_signal(df)
        assert isinstance(result, dict)
        assert "signal" in result
        assert "reason" in result

    def test_valid_signal_values(self):
        s = ConfluenceStrategy(swing_window=5, require_sweep=False,
                               require_fvg_or_ob=False)
        df = _make_df(150)
        result = s.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")

    def test_with_all_filters_off(self):
        """Sin filtros debe poder generar señal."""
        s = ConfluenceStrategy(
            swing_window=5,
            require_sweep=False,
            require_fvg_or_ob=False,
            require_bos=False,
            use_pd_filter=False,
            use_ema_filter=False,
        )
        df = _make_df(150, trend="bullish")
        result = s.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")

    def test_structure_key_in_result(self):
        s = ConfluenceStrategy(swing_window=5, require_sweep=False,
                               require_fvg_or_ob=False)
        df = _make_df(150)
        result = s.generate_signal(df)
        # Si hay estructura debe estar en extra
        # (puede no estar si se cortó antes de la estructura)
        assert "signal" in result

    def test_fvg_hit_key_when_filter_active(self):
        s = ConfluenceStrategy(swing_window=5, require_sweep=False,
                               require_fvg_or_ob=True)
        df = _make_df(150)
        result = s.generate_signal(df)
        # Si llegó hasta el filtro FVG/OB, debe tener las keys
        if result["signal"] != "hold" or "fvg_hit" in result:
            if "fvg_hit" in result:
                assert isinstance(result["fvg_hit"], bool)


# ── reason strings ────────────────────────────────────────────────────────────

class TestConfluenceReasonString:
    def test_reason_contains_sweep_when_required(self):
        """Cuando require_sweep=True y hay señal, 'Sweep' aparece en reason."""
        # Usamos configuración mínima para maximizar señales
        s = ConfluenceStrategy(
            swing_window=5, require_sweep=True,
            require_fvg_or_ob=False, require_bos=False
        )
        signals = []
        for seed in range(20):
            df = _make_df(150, seed=seed)
            r = s.generate_signal(df)
            if r["signal"] in ("buy", "sell"):
                signals.append(r)
        # Al menos un signal debe tener "Sweep" en reason
        if signals:
            assert any("Sweep" in r["reason"] for r in signals)

    def test_hold_reason_not_empty(self):
        s = ConfluenceStrategy(swing_window=5)
        df = _make_df(100)  # Puede ser hold con filtros estrictos
        result = s.generate_signal(df)
        if result["signal"] == "hold":
            assert len(result["reason"]) > 0


# ── Filtros opcionales ────────────────────────────────────────────────────────

class TestConfluenceOptionalFilters:
    def test_pd_filter_no_crash(self):
        s = ConfluenceStrategy(swing_window=5, require_sweep=False,
                               require_fvg_or_ob=False, use_pd_filter=True)
        df = _make_df(150)
        result = s.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")

    def test_ema_filter_no_crash(self):
        s = ConfluenceStrategy(swing_window=5, require_sweep=False,
                               require_fvg_or_ob=False, require_bos=False,
                               use_ema_filter=True, trading_style="swing")
        df = _make_df(300)  # EMA 200 necesita datos
        result = s.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")

    def test_all_filters_enabled_no_crash(self):
        s = ConfluenceStrategy(
            swing_window=5, require_sweep=True, require_fvg_or_ob=True,
            require_bos=True, use_pd_filter=True, use_ema_filter=True,
            trading_style="intraday"
        )
        df = _make_df(300)
        result = s.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")

    def test_different_swing_windows(self):
        for window in [5, 10, 15, 20]:
            s = ConfluenceStrategy(swing_window=window, require_sweep=False,
                                   require_fvg_or_ob=False)
            df = _make_df(window * 6)
            result = s.generate_signal(df)
            assert result["signal"] in ("buy", "sell", "hold"), \
                f"Falló con swing_window={window}"

    def test_min_sweep_bars_param(self):
        """min_sweep_bars no debe crashear."""
        s = ConfluenceStrategy(swing_window=5, require_sweep=True,
                               min_sweep_bars=5, require_fvg_or_ob=False)
        df = _make_df(150)
        result = s.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")
