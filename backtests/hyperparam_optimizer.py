"""
hyperparam_optimizer.py

Grid search sobre hiperparámetros del sistema de trading.

Optimiza combinaciones de:
  - swing_window  : tamaño de ventana para detectar swings SMC
  - max_hold      : velas máximas por trade
  - interval      : timeframe (solo en modo KuCoin)
  - threshold     : umbral ML (si use_ml=True)
  - kelly_variant : full_kelly / half_kelly / quarter_kelly

Métricas objetivo (configurable):
  - sharpe   (default)
  - sortino
  - calmar
  - winrate
  - profit_factor
  - compound_return

Uso rápido (datos simulados):
    from backtests.hyperparam_optimizer import HyperparamOptimizer
    opt = HyperparamOptimizer(use_real_data=False)
    results = opt.run()
    opt.print_results(results)

Uso con KuCoin (requiere red):
    opt = HyperparamOptimizer(use_real_data=True, symbols=["BTC-USDT"], interval="15min")
    results = opt.run()
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtests.backtester_fast import _precompute_signals
from backtests.kelly_backtest import _run_kelly_trades, _kelly_metrics
from strategies.kelly_sizer import KellySizer


# ── Generador de datos sintéticos ─────────────────────────────────────────────

def _gen_data(name: str, n: int = 35_040, seed: int = 42) -> pd.DataFrame:
    """Genera OHLCV sintético con ciclos bull/bear/sideways realistas."""
    rng    = np.random.default_rng(seed)
    prices = [100.0]
    slopes = {
        "BTC": 0.004, "ETH": 0.005, "SOL": 0.007,
        "BNB": 0.003, "LINK": 0.006,
    }
    sl       = slopes.get(name, 0.004)
    phase_len = n // 8
    patterns  = ["bull", "sideways", "bull", "bear", "sideways", "bull", "bear", "bull"]
    phases    = []
    for ph in patterns:
        v = sl if ph == "bull" else (-sl * 0.75 if ph == "bear" else 0.0)
        phases.extend([v] * phase_len)
    while len(phases) < n:
        phases.append(sl * 0.3)

    vol = 0.6
    for i in range(n - 1):
        shock = rng.standard_normal() * vol
        vol   = np.clip(0.95 * vol + 0.05 * (abs(shock) * 0.5 + 0.3), 0.2, 2.5)
        prices.append(max(1.0, prices[-1] * (1 + phases[i] / 100 + shock / 100)))
    prices = np.array(prices)
    spread = abs(rng.standard_normal(n)) * prices * 0.002
    # open ≈ previous close → cuerpos de vela realistas
    opens  = np.concatenate([[prices[0]], prices[:-1]])
    highs  = np.maximum(opens, prices) + spread
    lows   = np.maximum(np.minimum(opens, prices) - spread, prices * 0.98)
    return pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  prices,
        "volume": rng.integers(5000, 80000, n).astype(float),
    })


# ── Función de evaluación ─────────────────────────────────────────────────────

def _evaluate(data: pd.DataFrame,
              swing_window: int,
              max_hold: int,
              kelly_variant: str,
              require_fvg: bool,
              use_choch_filter: bool,
              periods_per_year: int = 35_040) -> dict:
    """Ejecuta un backtest con los hiperparámetros dados y devuelve métricas."""
    sizer   = KellySizer(variant=kelly_variant, min_trades=20, max_fraction=0.30)
    signals = _precompute_signals(
        data,
        swing_window=swing_window,
        require_fvg=require_fvg,
        use_choch_filter=use_choch_filter,
    )
    trades, equity = _run_kelly_trades(signals, data, sizer, max_hold=max_hold)
    metrics = _kelly_metrics(trades, equity, periods_per_year=periods_per_year)
    return metrics


# ── Grid search ───────────────────────────────────────────────────────────────

# Espacio de hiperparámetros por defecto
DEFAULT_PARAM_GRID = {
    "swing_window":    [3, 5, 8, 12, 20],
    "max_hold":        [4, 8, 16, 24],
    "kelly_variant":   ["half_kelly", "quarter_kelly", "full_kelly"],
    "require_fvg":     [False, True],
    "use_choch_filter":[True, False],
}

# Espacio reducido para búsqueda rápida
FAST_PARAM_GRID = {
    "swing_window":    [3, 5, 10],
    "max_hold":        [4, 8, 16],
    "kelly_variant":   ["half_kelly", "quarter_kelly"],
    "require_fvg":     [False],
    "use_choch_filter":[True],
}


class HyperparamOptimizer:
    """
    Optimizador de hiperparámetros por grid search.

    Parámetros
    ----------
    symbols         : lista de pares (solo para use_real_data=True)
    param_grid      : dict con listas de valores por parámetro
    objective       : métrica a maximizar ("sharpe","sortino","calmar","winrate")
    n_candles       : número de velas por símbolo (datos simulados)
    use_real_data   : si True, descarga datos reales de KuCoin
    interval        : timeframe KuCoin (solo use_real_data=True)
    periods_per_year: para Sharpe/Sortino (default 35040 = 15m anual)
    verbose         : imprimir progreso
    """

    def __init__(self,
                 symbols: list = None,
                 param_grid: dict = None,
                 objective: str = "sharpe",
                 n_candles: int = 35_040,
                 use_real_data: bool = False,
                 interval: str = "15min",
                 periods_per_year: int = 35_040,
                 verbose: bool = True):

        self.symbols          = symbols or ["BTC", "ETH", "SOL", "BNB", "LINK"]
        self.param_grid       = param_grid or FAST_PARAM_GRID
        self.objective        = objective
        self.n_candles        = n_candles
        self.use_real_data    = use_real_data
        self.interval         = interval
        self.periods_per_year = periods_per_year
        self.verbose          = verbose

    # ── Carga de datos ────────────────────────────────────────────────

    def _load_data(self) -> dict:
        """Devuelve {symbol: DataFrame}."""
        datasets = {}
        if self.use_real_data:
            from data.kucoin_client import KuCoinClient
            client = KuCoinClient(verbose=False)
            for sym in self.symbols:
                pair = sym if "-USDT" in sym else f"{sym}-USDT"
                if self.verbose:
                    print(f"  Descargando {pair} {self.interval}…")
                try:
                    df = client.get_ohlcv(pair, interval=self.interval,
                                          limit=min(self.n_candles, 1500))
                    if df is not None and len(df) > 100:
                        datasets[sym] = df
                except Exception as e:
                    print(f"  ERROR {pair}: {e}")
        else:
            for i, sym in enumerate(self.symbols):
                datasets[sym] = _gen_data(sym, self.n_candles, seed=i * 7 + 42)
        return datasets

    # ── Grid search principal ─────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Ejecuta el grid search y devuelve DataFrame con todos los resultados
        ordenados por la métrica objetivo (descendente).
        """
        datasets   = self._load_data()
        keys       = list(self.param_grid.keys())
        values     = list(self.param_grid.values())
        combos     = list(itertools.product(*values))
        n_combos   = len(combos)
        n_datasets = len(datasets)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  OPTIMIZADOR DE HIPERPARÁMETROS — Grid Search")
            print(f"  Combinaciones: {n_combos} × {n_datasets} pares = {n_combos*n_datasets} evaluaciones")
            print(f"  Objetivo: maximizar {self.objective}")
            print(f"{'='*70}")

        records = []
        t0      = time.time()

        for combo_idx, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            combo_metrics = []

            for sym, df in datasets.items():
                try:
                    m = _evaluate(
                        df,
                        swing_window     = params["swing_window"],
                        max_hold         = params["max_hold"],
                        kelly_variant    = params["kelly_variant"],
                        require_fvg      = params["require_fvg"],
                        use_choch_filter = params["use_choch_filter"],
                        periods_per_year = self.periods_per_year,
                    )
                    combo_metrics.append(m)
                except Exception:
                    pass  # combinación inválida, ignorar

            if not combo_metrics:
                continue

            # Agregar métricas media entre pares
            def _avg(key):
                vals = [m.get(key, 0) for m in combo_metrics if m.get(key) is not None]
                return float(np.mean(vals)) if vals else 0.0

            def _avg_eq_end():
                ends = [m["equity_curve"][-1] for m in combo_metrics
                        if m.get("equity_curve") and len(m["equity_curve"]) > 0]
                return float(np.mean(ends)) if ends else 1.0

            record = {
                **params,
                "sharpe":          round(_avg("sharpe"), 4),
                "sortino":         round(_avg("sortino"), 4),
                "calmar":          round(_avg("calmar"), 4),
                "winrate":         round(_avg("winrate") * 100, 2),
                "profit_factor":   round(_avg("profit_factor"), 4),
                "max_drawdown":    round(_avg("max_drawdown") * 100, 2),
                "trades_per_sym":  round(_avg("trades"), 1),
                "compound_return": round((_avg_eq_end() - 1) * 100, 2),
                "n_symbols":       len(combo_metrics),
            }
            records.append(record)

            if self.verbose and (combo_idx % max(1, n_combos // 20) == 0 or combo_idx == n_combos - 1):
                elapsed = time.time() - t0
                pct     = (combo_idx + 1) / n_combos * 100
                best_so_far = max(r[self.objective] for r in records) if records else 0.0
                print(f"  [{pct:5.1f}%] t={elapsed:5.1f}s  combinaciones={combo_idx+1}/{n_combos}"
                      f"  mejor_{self.objective}={best_so_far:.4f}")

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(self.objective, ascending=False).reset_index(drop=True)
            # Añadir columna de ranking
            df.insert(0, "rank", range(1, len(df) + 1))

        return df

    # ── Utilidades de resultado ───────────────────────────────────────

    def print_results(self, df: pd.DataFrame, top_n: int = 10):
        """Imprime tabla con los mejores resultados."""
        if df.empty:
            print("No hay resultados.")
            return

        print(f"\n{'='*90}")
        print(f"  TOP {top_n} COMBINACIONES — objetivo: {self.objective}")
        print(f"{'='*90}")
        print(f"  {'#':>3}  {'sw':>4}  {'hold':>5}  {'kelly':>14}  {'fvg':>4}  "
              f"{'choch':>6}  {'Sharpe':>7}  {'WR%':>6}  {'Ret%':>9}  "
              f"{'DD%':>6}  {'T/par':>6}")
        print(f"  {'-'*86}")

        for _, row in df.head(top_n).iterrows():
            print(f"  {int(row['rank']):>3}  "
                  f"{int(row['swing_window']):>4}  "
                  f"{int(row['max_hold']):>5}  "
                  f"{row['kelly_variant']:>14}  "
                  f"{'Y' if row['require_fvg'] else 'N':>4}  "
                  f"{'Y' if row['use_choch_filter'] else 'N':>6}  "
                  f"{row['sharpe']:>7.3f}  "
                  f"{row['winrate']:>6.1f}%  "
                  f"{row['compound_return']:>+9.1f}%  "
                  f"{row['max_drawdown']:>6.1f}%  "
                  f"{row['trades_per_sym']:>6.0f}")

    def best_params(self, df: pd.DataFrame) -> dict:
        """Retorna el mejor conjunto de hiperparámetros."""
        if df.empty:
            return {}
        row = df.iloc[0]
        return {
            "swing_window":     int(row["swing_window"]),
            "max_hold":         int(row["max_hold"]),
            "kelly_variant":    row["kelly_variant"],
            "require_fvg":      bool(row["require_fvg"]),
            "use_choch_filter": bool(row["use_choch_filter"]),
        }

    def save_plots(self, df: pd.DataFrame, path: str = "data/plots/25_hyperparam_heatmap.png"):
        """Genera heatmaps de Sharpe por pares de hiperparámetros."""
        if df.empty:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Grid Search — {self.objective.capitalize()} por hiperparámetro",
                     fontsize=13, fontweight="bold")

        pairs = [
            ("swing_window", "max_hold"),
            ("swing_window", "kelly_variant"),
            ("max_hold",     "kelly_variant"),
        ]

        for ax, (xp, yp) in zip(axes, pairs):
            pivot = df.groupby([xp, yp])[self.objective].mean().unstack(yp)
            if pivot.empty:
                continue
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels([str(c) for c in pivot.columns], rotation=30, fontsize=8)
            ax.set_yticklabels([str(i) for i in pivot.index], fontsize=8)
            ax.set_xlabel(yp, fontsize=9)
            ax.set_ylabel(xp, fontsize=9)
            ax.set_title(f"{xp} vs {yp}", fontweight="bold")
            plt.colorbar(im, ax=ax, label=self.objective)

            # Anotar valores
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=7, color="black")

        plt.tight_layout()
        plt.savefig(path, dpi=130)
        plt.close()
        print(f"  Guardado: {path}")

    def save_results(self, df: pd.DataFrame, path: str = "data/hyperparam_results.csv"):
        """Guarda resultados en CSV."""
        df.to_csv(path, index=False)
        print(f"  Guardado: {path}")


# ── Análisis adicional — curva de sensibilidad ────────────────────────────────

def sensitivity_analysis(df: pd.DataFrame, param: str, objective: str = "sharpe") -> pd.DataFrame:
    """
    Analiza la sensibilidad de la métrica objetivo a un único parámetro
    (media y std entre todos los valores del resto de parámetros).
    """
    grouped = df.groupby(param)[objective].agg(["mean", "std", "min", "max", "count"])
    grouped.columns = ["mean", "std", "min", "max", "n_combinations"]
    grouped = grouped.sort_values("mean", ascending=False)
    return grouped.reset_index()


# ── Runner standalone ─────────────────────────────────────────────────────────

def run_fast_optimization(use_real_data: bool = False,
                           symbols: list = None,
                           objective: str = "sharpe") -> pd.DataFrame:
    """
    Ejecuta una optimización rápida (~18 combinaciones × N pares).
    Retorna DataFrame con resultados ordenados.
    """
    opt = HyperparamOptimizer(
        symbols       = symbols or ["BTC", "ETH", "SOL"],
        param_grid    = FAST_PARAM_GRID,
        objective     = objective,
        use_real_data = use_real_data,
        verbose       = True,
    )
    df = opt.run()
    opt.print_results(df, top_n=10)

    best = opt.best_params(df)
    print(f"\n  MEJOR CONFIGURACIÓN ({objective}):")
    for k, v in best.items():
        print(f"    {k:20s} = {v}")

    opt.save_plots(df)
    opt.save_results(df)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimizador de hiperparámetros")
    parser.add_argument("--real",      action="store_true", help="Usar datos reales de KuCoin")
    parser.add_argument("--full",      action="store_true", help="Grid completo (lento)")
    parser.add_argument("--objective", default="sharpe",    help="Métrica objetivo")
    parser.add_argument("--symbols",   default="BTC,ETH,SOL,BNB,LINK", help="Pares (sin -USDT)")
    args = parser.parse_args()

    symbols = args.symbols.split(",")
    grid    = DEFAULT_PARAM_GRID if args.full else FAST_PARAM_GRID

    opt = HyperparamOptimizer(
        symbols       = symbols,
        param_grid    = grid,
        objective     = args.objective,
        use_real_data = args.real,
        verbose       = True,
    )
    results = opt.run()
    opt.print_results(results, top_n=15)
    opt.save_plots(results)
    opt.save_results(results)

    print("\n  Análisis de sensibilidad por parámetro:")
    for p in ["swing_window", "max_hold", "kelly_variant"]:
        if p in results.columns:
            sa = sensitivity_analysis(results, p, args.objective)
            print(f"\n  [{p}]")
            print(sa.to_string(index=False))
