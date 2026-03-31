"""
run_demo.py

Ejecuta el grid search y el backtest anual completo usando datos sinteticos
(sin necesidad de conexion a Binance API).

Genera los mismos CSVs y muestra los mismos resultados que los scripts reales.

Uso:
    python run_demo.py
"""

import os
import pandas as pd

from bot.mock_data import generate_mock_klines
from backtests.strategy_selector import run_selection, save_csv, print_top
from backtests.annual_backtest import (
    PAIRS, TIMEFRAMES,
    _flatten_results, _print_summary,
    ANNUAL_CSV, DETAIL_CSV,
)

# Candles por timeframe para simular 1 año
_TF_CANDLES = {"1d": 365, "4h": 365 * 6, "1h": 365 * 24}


# ─────────────────────────────────────────────
#  PARTE 1 — Grid search BTCUSDT (200 velas 1m)
# ─────────────────────────────────────────────

def demo_strategy_selector():
    print("=" * 60)
    print("  PARTE 1 — Grid search BTCUSDT (200 velas mock 1m)")
    print("=" * 60 + "\n")

    data = generate_mock_klines("BTCUSDT", n_candles=200, timeframe="1m")
    print(f"Datos mock generados: {len(data)} velas\n")

    all_results, top = run_selection(data, top_n=10, fast=True)
    print_top(top)
    save_csv(all_results)


# ─────────────────────────────────────────────
#  PARTE 2 — Backtest anual multi-par x TF
# ─────────────────────────────────────────────

def demo_annual_backtest():
    print("\n" + "=" * 60)
    print("  PARTE 2 — Backtest anual (datos mock)")
    print(f"  Pares: {', '.join(PAIRS)}")
    print(f"  Timeframes: {', '.join(TIMEFRAMES)}")
    print("=" * 60 + "\n")

    summary_rows = []
    detail_rows = []
    total = len(PAIRS) * len(TIMEFRAMES)
    n = 0

    for pair in PAIRS:
        for tf in TIMEFRAMES:
            n += 1
            n_candles = _TF_CANDLES.get(tf, 365)
            print(f"[{n:2d}/{total}]  {pair} · {tf}  ({n_candles} velas mock)")

            data = generate_mock_klines(pair, n_candles=n_candles, timeframe=tf)

            all_results, top = run_selection(data, top_n=10, fast=True)
            detail_rows.extend(_flatten_results(pair, tf, all_results))

            best = next((e for e in all_results if e["metrics"]["trades"] >= 3), None)
            if best is None:
                print(f"         Sin estrategias validas para {pair} {tf}")
                continue

            p = best["params"]
            m = best["metrics"]
            pf = m["profit_factor"]
            pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
            print(
                f"         Mejor → PnL={m['total_pnl']:+.2%}  "
                f"WR={m['winrate']:.1%}  PF={pf_str}  "
                f"trades={m['trades']}  "
                f"(sw={p['swing_window']} hold={p['max_hold']})"
            )

            summary_rows.append({
                "pair":             pair,
                "timeframe":        tf,
                "total_pnl":        m["total_pnl"],
                "winrate":          m["winrate"],
                "profit_factor":    pf,
                "trades":           m["trades"],
                "swing_window":     p["swing_window"],
                "max_hold":         p["max_hold"],
                "require_fvg":      p["require_fvg"],
                "use_choch_filter": p["use_choch_filter"],
            })

    if not summary_rows:
        print("No se generaron resultados.")
        return

    _print_summary(summary_rows)

    os.makedirs("data", exist_ok=True)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(ANNUAL_CSV, index=False)
    print(f"Resumen guardado en:  {os.path.abspath(ANNUAL_CSV)}")

    df_detail = pd.DataFrame(detail_rows)
    df_detail.to_csv(DETAIL_CSV, index=False)
    print(f"Detalle guardado en:  {os.path.abspath(DETAIL_CSV)}")


# ─────────────────────────────────────────────

if __name__ == "__main__":
    demo_strategy_selector()
    demo_annual_backtest()
    print("\nDemo completado. Abre el notebook para ver las graficas:")
    print("  jupyter notebook notebooks/strategy_analysis.ipynb\n")
