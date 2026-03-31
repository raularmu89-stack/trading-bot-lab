"""
strategy_selector.py

Genera 100 variantes de SMCStrategy combinando parametros,
las backtestea sobre los mismos datos y devuelve el top 10.

Grid de parametros (5 x 5 x 2 x 2 = 100 estrategias):
  swing_window      : velas a cada lado para swings reales
  max_hold          : velas maximas antes de forzar cierre del trade
  require_fvg       : exige FVG del mismo tipo para confirmar entrada
  use_choch_filter  : CHoCH fuerza hold en lugar de operar

Criterio de seleccion:
  - Se descartan estrategias con menos de MIN_TRADES trades
  - Ranking por total_pnl (beneficio total acumulado)
  - En caso de empate, desempata profit_factor
"""

import itertools
import os

import pandas as pd

from bot.data_fetcher import fetch_klines
from strategies.smc_strategy import SMCStrategy
from backtests.backtester import Backtester

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results.csv")

MIN_TRADES = 3

PARAM_GRID = {
    "swing_window":     [3, 5, 7, 10, 15],
    "max_hold":         [5, 10, 15, 20, 30],
    "require_fvg":      [True, False],
    "use_choch_filter": [True, False],
}


def _build_combinations():
    keys = list(PARAM_GRID.keys())
    for values in itertools.product(*[PARAM_GRID[k] for k in keys]):
        yield dict(zip(keys, values))


def _score(metrics):
    if metrics["trades"] < MIN_TRADES:
        return (float("-inf"), 0.0)
    return (metrics["total_pnl"], metrics["profit_factor"])


def run_selection(data, top_n=10):
    combos = list(_build_combinations())
    total = len(combos)
    print(f"Ejecutando {total} estrategias...\n")

    results = []
    for i, params in enumerate(combos):
        strategy = SMCStrategy(
            swing_window=params["swing_window"],
            require_fvg=params["require_fvg"],
            use_choch_filter=params["use_choch_filter"],
        )
        backtester = Backtester(
            strategy=strategy,
            data=data,
            max_hold=params["max_hold"],
            verbose=False,
        )
        metrics = backtester.run()

        results.append({
            "params": params,
            "metrics": metrics,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1:3d}/{total}] completadas...")

    results.sort(key=lambda x: _score(x["metrics"]), reverse=True)
    return results, results[:top_n]


def save_csv(results, path=RESULTS_PATH):
    rows = []
    for rank, entry in enumerate(results, 1):
        p = entry["params"]
        m = entry["metrics"]
        rows.append({
            "rank":             rank,
            "total_pnl":        m["total_pnl"],
            "winrate":          m["winrate"],
            "profit_factor":    m["profit_factor"],
            "trades":           m["trades"],
            "swing_window":     p["swing_window"],
            "max_hold":         p["max_hold"],
            "require_fvg":      p["require_fvg"],
            "use_choch_filter": p["use_choch_filter"],
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Resultados guardados en: {os.path.abspath(path)}")


def print_top(top):
    print("\n" + "=" * 72)
    print(f"{'TOP ' + str(len(top)) + ' ESTRATEGIAS':^72}")
    print("=" * 72)
    header = (
        f"{'#':>3}  {'pnl':>7}  {'winrate':>7}  {'pf':>6}  "
        f"{'trades':>6}  {'swing':>5}  {'hold':>4}  {'fvg':>5}  {'choch':>5}"
    )
    print(header)
    print("-" * 72)
    for rank, entry in enumerate(top, 1):
        p = entry["params"]
        m = entry["metrics"]
        pf = m["profit_factor"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "  inf"
        print(
            f"{rank:>3}  "
            f"{m['total_pnl']:>+7.2%}  "
            f"{m['winrate']:>7.1%}  "
            f"{pf_str:>6}  "
            f"{m['trades']:>6}  "
            f"{p['swing_window']:>5}  "
            f"{p['max_hold']:>4}  "
            f"{'si' if p['require_fvg'] else 'no':>5}  "
            f"{'si' if p['use_choch_filter'] else 'no':>5}"
        )
    print("=" * 72)
    print("Columnas: pnl=beneficio_total | pf=profit_factor | fvg=require_fvg | choch=use_choch_filter\n")


def main():
    print("Descargando datos de mercado...")
    data = fetch_klines()
    if data is None:
        print("Error al obtener datos.")
        return
    print(f"Datos cargados: {len(data)} velas\n")

    all_results, top = run_selection(data, top_n=10)
    print_top(top)
    save_csv(all_results)

    return all_results, top


if __name__ == "__main__":
    main()
