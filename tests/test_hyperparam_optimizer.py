"""
tests/test_hyperparam_optimizer.py

Tests para el optimizador de hiperparámetros.
"""
import pytest
import pandas as pd
import numpy as np

from backtests.hyperparam_optimizer import (
    HyperparamOptimizer, _gen_data, _evaluate,
    sensitivity_analysis, FAST_PARAM_GRID,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

MINI_GRID = {
    "swing_window":    [3, 5],
    "max_hold":        [4, 8],
    "kelly_variant":   ["half_kelly"],
    "require_fvg":     [False],
    "use_choch_filter":[True],
}


def _mini_opt(**kw):
    return HyperparamOptimizer(
        symbols    = ["BTC", "ETH"],
        param_grid = MINI_GRID,
        n_candles  = 5_000,
        verbose    = False,
        **kw,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# _gen_data
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenData:
    def test_returns_dataframe(self):
        assert isinstance(_gen_data("BTC", n=500), pd.DataFrame)

    def test_columns(self):
        df = _gen_data("BTC", n=500)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_length(self):
        df = _gen_data("ETH", n=1000)
        assert len(df) == 1000

    def test_high_gte_low(self):
        df = _gen_data("SOL", n=300)
        assert (df["high"] >= df["low"]).all()

    def test_different_seeds_differ(self):
        d1 = _gen_data("BTC", n=500, seed=1)
        d2 = _gen_data("BTC", n=500, seed=2)
        assert not d1["close"].equals(d2["close"])


# ═══════════════════════════════════════════════════════════════════════════════
# _evaluate
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluate:
    def _df(self):
        return _gen_data("BTC", n=5000)

    def test_returns_dict(self):
        assert isinstance(_evaluate(self._df(), 5, 8, "half_kelly", False, True), dict)

    def test_sharpe_key(self):
        m = _evaluate(self._df(), 5, 8, "half_kelly", False, True)
        assert "sharpe" in m

    def test_trades_nonneg(self):
        m = _evaluate(self._df(), 5, 8, "half_kelly", False, True)
        assert m["trades"] >= 0

    def test_winrate_range(self):
        m = _evaluate(self._df(), 5, 8, "half_kelly", False, True)
        if m["trades"] > 0:
            assert 0 <= m["winrate"] <= 1

    def test_equity_curve_present(self):
        m = _evaluate(self._df(), 5, 8, "half_kelly", False, True)
        assert "equity_curve" in m
        assert len(m["equity_curve"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# HyperparamOptimizer
# ═══════════════════════════════════════════════════════════════════════════════

class TestHyperparamOptimizer:
    def test_init_default(self):
        opt = HyperparamOptimizer(verbose=False)
        assert opt.objective == "sharpe"
        assert opt.use_real_data is False

    def test_run_returns_dataframe(self):
        df = _mini_opt().run()
        assert isinstance(df, pd.DataFrame)

    def test_run_has_results(self):
        df = _mini_opt().run()
        assert len(df) > 0

    def test_sorted_by_objective(self):
        df = _mini_opt().run()
        # Top row should have the highest sharpe
        assert df.iloc[0]["sharpe"] >= df.iloc[-1]["sharpe"]

    def test_rank_column(self):
        df = _mini_opt().run()
        assert "rank" in df.columns
        assert df["rank"].iloc[0] == 1

    def test_required_columns(self):
        df = _mini_opt().run()
        for col in ["sharpe", "winrate", "max_drawdown", "trades_per_sym", "compound_return"]:
            assert col in df.columns

    def test_objective_sortino(self):
        df = _mini_opt(objective="sortino").run()
        assert df.iloc[0]["sortino"] >= df.iloc[-1]["sortino"]

    def test_objective_winrate(self):
        df = _mini_opt(objective="winrate").run()
        assert df.iloc[0]["winrate"] >= df.iloc[-1]["winrate"]

    def test_best_params_returns_dict(self):
        opt = _mini_opt()
        df  = opt.run()
        bp  = opt.best_params(df)
        assert isinstance(bp, dict)

    def test_best_params_keys(self):
        opt = _mini_opt()
        df  = opt.run()
        bp  = opt.best_params(df)
        for k in ["swing_window", "max_hold", "kelly_variant", "require_fvg", "use_choch_filter"]:
            assert k in bp

    def test_best_params_empty_df(self):
        opt = _mini_opt()
        assert opt.best_params(pd.DataFrame()) == {}

    def test_swing_window_in_results(self):
        df = _mini_opt().run()
        assert "swing_window" in df.columns
        assert set(df["swing_window"].unique()).issubset({3, 5})

    def test_n_combinations(self):
        # 2 swing × 2 hold × 1 kelly × 1 fvg × 1 choch = 4 combos
        df = _mini_opt().run()
        assert len(df) == 4

    def test_save_results(self, tmp_path):
        opt = _mini_opt()
        df  = opt.run()
        path = str(tmp_path / "opt_results.csv")
        opt.save_results(df, path)
        import os
        assert os.path.exists(path)
        loaded = pd.read_csv(path)
        assert len(loaded) == len(df)

    def test_save_plots(self, tmp_path):
        opt = _mini_opt()
        df  = opt.run()
        path = str(tmp_path / "heatmap.png")
        opt.save_plots(df, path)
        import os
        assert os.path.exists(path)

    def test_print_results_no_crash(self, capsys):
        opt = _mini_opt()
        df  = opt.run()
        opt.print_results(df, top_n=3)
        out = capsys.readouterr().out
        assert "TOP" in out

    def test_print_results_empty(self, capsys):
        opt = _mini_opt()
        opt.print_results(pd.DataFrame())
        out = capsys.readouterr().out
        assert "No hay" in out


# ═══════════════════════════════════════════════════════════════════════════════
# sensitivity_analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestSensitivityAnalysis:
    def test_returns_dataframe(self):
        df = _mini_opt().run()
        sa = sensitivity_analysis(df, "swing_window")
        assert isinstance(sa, pd.DataFrame)

    def test_has_mean_std(self):
        df = _mini_opt().run()
        sa = sensitivity_analysis(df, "swing_window")
        assert "mean" in sa.columns
        assert "std"  in sa.columns

    def test_has_all_param_values(self):
        df = _mini_opt().run()
        sa = sensitivity_analysis(df, "swing_window")
        assert set(sa["swing_window"].values).issubset({3, 5})
