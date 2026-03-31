"""
annual_backtest.py

Backtest de 1 año para 4 pares en 3 timeframes distintos:

  Pares:       BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT
  Timeframes:  1d (365 velas), 4h (~2190 velas), 1h (~8760 velas)

Para cada combinacion par × timeframe:
  - Descarga 1 año completo de velas via paginacion automatica
  - Ejecuta el grid search de 100 estrategias
  - Selecciona la mejor por PnL (minimo 3 trades)

Salida:
  data/annual_results.csv   — mejor estrategia por par × timeframe
  data/annual_detail.csv    — los 100 resultados de cada combinacion

Nota: XRPUSD corregido a XRPUSDT (formato Binance).
Nota: 1h puede tardar varios minutos por el volumen de datos.
"""

import os

import pandas as pd

from bot.data_fetcher import fetch_klines_year
from backtests.strategy_selector import run_selection

PAIRS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]
TIMEFRAMES = ["1d", "4h", "1h"]

ANNUAL_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "annual_results.csv")
DETAIL_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "annual_detail.csv")


def _flatten_results(pair, timeframe, results):
    rows = []
    for rank, entry in enumerate(results, 1):
        p = entry["params"]
        m = entry["metrics"]
        rows.append({
            "pair":             pair,
            "timeframe":        timeframe,
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
    return rows


def _print_summary(summary_rows):
    print("\n" + "=" * 84)
    print(f"{'BACKTEST ANUAL — MEJOR ESTRATEGIA POR PAR × TIMEFRAME':^84}")
    print("=" * 84)
    header = (
        f"{'Par':<10}  {'TF':<4}  {'PnL':>8}  {'Winrate':>8}  {'PF':>6}  "
        f"{'Trades':>6}  {'swing':>5}  {'hold':>4}  {'fvg':>5}  {'choch':>5}"
    )
    print(header)
    print("-" * 84)

    last_pair = None
    for r in summary_rows:
        if last_pair and r["pair"] != last_pair:
            print()
        last_pair = r["pair"]
        pf = r["profit_factor"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "  inf"
        print(
            f"{r['pair']:<10}  "
            f"{r['timeframe']:<4}  "
            f"{r['total_pnl']:>+8.2%}  "
            f"{r['winrate']:>8.1%}  "
            f"{pf_str:>6}  "
            f"{int(r['trades']):>6}  "
            f"{int(r['swing_window']):>5}  "
            f"{int(r['max_hold']):>4}  "
            f"{'si' if r['require_fvg'] else 'no':>5}  "
            f"{'si' if r['use_choch_filter'] else 'no':>5}"
        )

    print("=" * 84)
    pnls = [r["total_pnl"] for r in summary_rows]
    wrs = [r["winrate"] for r in summary_rows]
    best = max(summary_rows, key=lambda x: x["total_pnl"])
    print(f"\nMedia PnL:     {sum(pnls)/len(pnls):+.2%}")
    print(f"Media Winrate: {sum(wrs)/len(wrs):.1%}")
    print(f"Mejor combo:   {best['pair']} {best['timeframe']} ({best['total_pnl']:+.2%})\n")


def run_annual_backtest():
    summary_rows = []
    detail_rows = []

    total_combos = len(PAIRS) * len(TIMEFRAMES)
    combo_n = 0

    for pair in PAIRS:
        for timeframe in TIMEFRAMES:
            combo_n += 1
            print(f"\n{'─' * 56}")
            print(f"  [{combo_n}/{total_combos}]  {pair}  ·  {timeframe}  —  1 año")
            print(f"{'─' * 56}")

            data = fetch_klines_year(symbol=pair, timeframe=timeframe)
            if data is None or data.empty:
                print(f"  ERROR: no se pudieron obtener datos para {pair} {timeframe}")
                continue
            print(f"  Datos cargados: {len(data)} velas\n")

            all_results, top = run_selection(data, top_n=10)

            detail_rows.extend(_flatten_results(pair, timeframe, all_results))

            best = None
            for entry in all_results:
                if entry["metrics"]["trades"] >= 3:
                    best = entry
                    break

            if best is None:
                print(f"  Sin estrategias validas (< 3 trades) para {pair} {timeframe}")
                continue

            p = best["params"]
            m = best["metrics"]
            pf = m["profit_factor"]

            print(f"  Mejor estrategia:")
            print(f"    PnL:           {m['total_pnl']:+.2%}")
            print(f"    Winrate:       {m['winrate']:.1%}")
            if pf == float("inf"):
                print(f"    Profit factor: inf")
            else:
                print(f"    Profit factor: {pf:.2f}")
            print(f"    Trades:        {m['trades']}")
            print(
                f"    Params:        swing={p['swing_window']}  hold={p['max_hold']}  "
                f"fvg={'si' if p['require_fvg'] else 'no'}  "
                f"choch={'si' if p['use_choch_filter'] else 'no'}"
            )

            summary_rows.append({
                "pair":             pair,
                "timeframe":        timeframe,
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
        return None, None

    _print_summary(summary_rows)

    os.makedirs(os.path.dirname(os.path.abspath(ANNUAL_CSV)), exist_ok=True)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(ANNUAL_CSV, index=False)
    print(f"Resumen guardado en:  {os.path.abspath(ANNUAL_CSV)}")

    df_detail = pd.DataFrame(detail_rows)
    df_detail.to_csv(DETAIL_CSV, index=False)
    print(f"Detalle guardado en:  {os.path.abspath(DETAIL_CSV)}")

    return df_summary, df_detail


def main():
    run_annual_backtest()


if __name__ == "__main__":
    main()
