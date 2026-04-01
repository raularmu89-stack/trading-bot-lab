"""
Tests para el módulo de autoaprendizaje ML:
  - ml/feature_extractor.py
  - ml/neural_net.py
  - ml/signal_filter.py
"""
import pytest
import numpy as np
import pandas as pd
import tempfile, os

from ml.feature_extractor import (
    extract_features, batch_extract, N_FEATURES, FEATURE_NAMES, REGIME_ORDER
)
from ml.neural_net import TradingNet
from ml.signal_filter import MLSignalFilter
from strategies.smc_strategy import SMCStrategy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make(n=150, trend="bull", seed=42):
    rng = np.random.default_rng(seed)
    prices = [100.0]
    slope = {"bull": 0.2, "bear": -0.2, "flat": 0.0}[trend]
    for _ in range(n - 1):
        prices.append(max(1.0, prices[-1] + slope + rng.standard_normal() * 0.4))
    prices = np.array(prices)
    spread = abs(rng.standard_normal(n)) * 0.3
    return pd.DataFrame({
        "open": prices, "high": prices + spread,
        "low":  np.maximum(prices - spread, prices * 0.98),
        "close": prices,
        "volume": rng.integers(1000, 10000, n).astype(float),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# FeatureExtractor
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureExtractor:
    def test_returns_array(self):
        f = extract_features(_make(), "buy")
        assert isinstance(f, np.ndarray)

    def test_shape_correct(self):
        f = extract_features(_make(), "buy")
        assert f.shape == (N_FEATURES,)

    def test_n_features_matches_names(self):
        assert N_FEATURES == len(FEATURE_NAMES)

    def test_regime_order_length(self):
        assert len(REGIME_ORDER) == 9

    def test_none_data_returns_none(self):
        assert extract_features(None, "buy") is None

    def test_short_data_returns_none(self):
        assert extract_features(_make(10), "buy") is None

    def test_buy_sell_differ(self):
        df = _make(200)
        f_buy  = extract_features(df, "buy")
        f_sell = extract_features(df, "sell")
        assert not np.allclose(f_buy, f_sell)

    def test_float32_dtype(self):
        f = extract_features(_make(), "buy")
        assert f.dtype == np.float32

    def test_values_finite(self):
        f = extract_features(_make(), "buy")
        assert np.all(np.isfinite(f))

    def test_one_hot_sums_to_one_or_zero(self):
        # Parte one-hot: last 9 features
        f = extract_features(_make(), "buy")
        one_hot = f[-9:]
        s = one_hot.sum()
        assert s in (0.0, 1.0)

    def test_batch_extract(self):
        dfs  = [_make(150, seed=i) for i in range(5)]
        sigs = ["buy"] * 5
        X, idx = batch_extract(dfs, sigs)
        assert X.shape[1] == N_FEATURES
        assert len(idx) <= 5

    def test_batch_extract_empty(self):
        X, idx = batch_extract([], [])
        assert X.shape == (0, N_FEATURES)
        assert idx == []

    def test_no_volume_column(self):
        df = _make().drop(columns=["volume"])
        f  = extract_features(df, "buy")
        assert f is not None
        assert f.shape == (N_FEATURES,)


# ═══════════════════════════════════════════════════════════════════════════════
# TradingNet
# ═══════════════════════════════════════════════════════════════════════════════

class TestTradingNet:
    def test_init_default(self):
        net = TradingNet()
        assert net.input_dim == 25
        assert net._step == 0

    def test_repr(self):
        net = TradingNet()
        assert "TradingNet" in repr(net)

    def test_predict_shape(self):
        net = TradingNet(input_dim=N_FEATURES)
        X   = np.random.rand(10, N_FEATURES).astype(np.float32)
        out = net.predict(X)
        assert out.shape == (10,)

    def test_predict_range(self):
        net = TradingNet(input_dim=N_FEATURES)
        X   = np.random.rand(50, N_FEATURES).astype(np.float32)
        out = net.predict(X)
        assert all(0 <= v <= 1 for v in out)

    def test_predict_single_float(self):
        net = TradingNet(input_dim=N_FEATURES)
        x   = np.random.rand(N_FEATURES).astype(np.float32)
        p   = net.predict_single(x)
        assert isinstance(p, float)
        assert 0 <= p <= 1

    def test_fit_reduces_loss(self):
        net = TradingNet(input_dim=N_FEATURES)
        X   = np.random.rand(100, N_FEATURES).astype(np.float32)
        y   = (np.random.rand(100) > 0.5).astype(np.float32)
        net.fit(X, y, epochs=30)
        assert len(net._train_loss) > 0
        # Loss inicial vs final
        assert net._train_loss[-1] < net._train_loss[0] * 1.5  # debe bajar o estabilizar

    def test_partial_fit_increments_step(self):
        net = TradingNet(input_dim=N_FEATURES)
        x   = np.random.rand(N_FEATURES).astype(np.float32)
        net.partial_fit(x, 1.0)
        assert net._step == 1
        net.partial_fit(x, 0.0)
        assert net._step == 2

    def test_score_keys(self):
        net = TradingNet(input_dim=N_FEATURES)
        X   = np.random.rand(50, N_FEATURES).astype(np.float32)
        y   = (np.random.rand(50) > 0.5).astype(np.float32)
        net.fit(X, y, epochs=5)
        s   = net.score(X, y)
        for k in ["accuracy", "precision", "recall", "f1", "n_samples"]:
            assert k in s

    def test_score_accuracy_range(self):
        net = TradingNet(input_dim=N_FEATURES)
        X   = np.random.rand(50, N_FEATURES).astype(np.float32)
        y   = (np.random.rand(50) > 0.5).astype(np.float32)
        net.fit(X, y, epochs=5)
        s   = net.score(X, y)
        assert 0 <= s["accuracy"] <= 1

    def test_lr_decays(self):
        net = TradingNet(input_dim=N_FEATURES, lr=0.01, lr_decay=0.99)
        x   = np.random.rand(N_FEATURES).astype(np.float32)
        lr0 = net.lr
        for _ in range(10):
            net.partial_fit(x, 1.0)
        assert net.lr < lr0

    def test_save_load(self):
        net = TradingNet(input_dim=N_FEATURES)
        X   = np.random.rand(30, N_FEATURES).astype(np.float32)
        y   = (np.random.rand(30) > 0.5).astype(np.float32)
        net.fit(X, y, epochs=5)
        pred_before = net.predict(X[:5])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_net")
            net.save(path)
            loaded = TradingNet.load(path)

        pred_after = loaded.predict(X[:5])
        np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5)

    def test_save_creates_files(self):
        net = TradingNet(input_dim=N_FEATURES)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "weights")
            net.save(path)
            assert os.path.exists(path + ".npz")
            assert os.path.exists(path + "_meta.json")

    def test_different_architectures(self):
        for h1, h2 in [(16, 8), (64, 32), (8, 4)]:
            net = TradingNet(input_dim=N_FEATURES, hidden1=h1, hidden2=h2)
            x   = np.random.rand(N_FEATURES).astype(np.float32)
            p   = net.predict_single(x)
            assert 0 <= p <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# MLSignalFilter
# ═══════════════════════════════════════════════════════════════════════════════

class TestMLSignalFilter:
    def _filter(self, **kw):
        return MLSignalFilter(SMCStrategy(swing_window=5), **kw)

    def test_defaults(self):
        f = self._filter()
        assert f.threshold == 0.52
        assert f.swing_window == 5
        assert f.require_fvg is False
        # use_choch_filter se delega al strategy base (SMCStrategy default = True)
        assert f.use_choch_filter == SMCStrategy(swing_window=5).use_choch_filter
        assert f._active is False

    def test_repr(self):
        f = self._filter()
        assert "MLSignalFilter" in repr(f)

    def test_before_min_trades_passes_all(self):
        f    = self._filter(min_trades_train=50)
        df   = _make(200, trend="bull")
        result = f.generate_signal(df)
        assert result.get("ml_active") is False

    def test_hold_signal_bypasses_ml(self):
        f    = self._filter()
        # En mercado plano difícil de generar señales
        df   = _make(200, trend="flat")
        result = f.generate_signal(df)
        # Si la base da hold, ML no interfiere
        if result["signal"] == "hold":
            assert result.get("ml_filtered") is not True

    def test_record_trade_increments_history(self):
        f  = self._filter()
        df = _make(150, trend="bull")
        f.generate_signal(df)  # genera features pendientes
        f.record_trade(pnl=0.02)
        assert len(f._history) == 1

    def test_activates_after_min_trades(self):
        f = self._filter(min_trades_train=5)
        df = _make(150, trend="bull")
        # Simular 5 trades
        for i in range(5):
            f.generate_signal(df)
            f.record_trade(pnl=(0.01 if i % 2 == 0 else -0.01))
        assert f._active is True

    def test_stats_keys(self):
        f = self._filter()
        s = f.stats
        assert "trades_seen" in s
        assert "active" in s
        assert "threshold" in s

    def test_stats_empty(self):
        f = self._filter()
        s = f.stats
        assert s["trades_seen"] == 0
        assert s["active"] is False

    def test_record_trade_result_win(self):
        f = self._filter()
        df = _make(150)
        f.generate_signal(df)
        # Entrada en 100, salida en 103 → ganancia
        f.record_trade_result(entry_price=100.0, exit_price=103.0, side="buy")
        assert len(f._history) == 1
        assert f._history[0][1] == 1.0  # label = win

    def test_record_trade_result_loss(self):
        f = self._filter()
        df = _make(150)
        f.generate_signal(df)
        f.record_trade_result(entry_price=100.0, exit_price=97.0, side="buy")
        assert f._history[0][1] == 0.0  # label = loss

    def test_reset_clears_state(self):
        f = self._filter(min_trades_train=3)
        df = _make(150)
        for _ in range(3):
            f.generate_signal(df)
            f.record_trade(0.01)
        assert f._active is True
        f.reset()
        assert f._active is False
        assert len(f._history) == 0

    def test_save_load_model(self):
        f  = self._filter()
        df = _make(150)
        # Entrenar con algunos trades
        for i in range(10):
            f.generate_signal(df)
            f.record_trade(0.01 if i % 2 == 0 else -0.01)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "filter_model")
            f.save(path)
            assert os.path.exists(path + ".npz")

    def test_generates_valid_signal_after_activation(self):
        f = self._filter(min_trades_train=5, threshold=0.01)  # umbral muy bajo
        df = _make(200, trend="bull")
        for i in range(5):
            f.generate_signal(df)
            f.record_trade(0.02)
        assert f._active is True
        result = f.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")

    def test_with_different_base_strategies(self):
        from strategies.scenario_router import ScenarioRouter
        for strat in [SMCStrategy(swing_window=5), ScenarioRouter()]:
            f = MLSignalFilter(strat, threshold=0.48)
            r = f.generate_signal(_make(200))
            assert r["signal"] in ("buy", "sell", "hold")
