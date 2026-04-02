"""
annual_fee_backtest.py

Backtest a 1 AÑO de todas las estrategias rentables encontradas en los
4 torneos anteriores.

Estrategias incluidas (confirmadas rentables post-comisiones):
  Torneo Fee-Aware (4H):
    - Turtle(en15,ex7,et100)         → mejor retorno absoluto (+3%)
    - WR(p28,os-85,et100)            → mejor Sharpe neto
    - WR(p21,os-85,et100)
    - WR(p21,os-80,et100)
    - MultiEMA(34/89/200, ADX>30)    → mejor WR (69%)
    - SAR(af0.01,m0.15,et100)        → buena relación riesgo/retorno
    - Chandelier(p14,m3.0,et200)
    - CCI(p20,t100,et100)
    - HA(cb2,et100,os45)
  Torneo R2 Grand Final:
    - SMC(sw5,hold48)                → ganador histórico
    - HeikinAshi(cb2,et100)
    - ParabolicSAR(et50)
    - WilliamsR(p28,os-80)

Datos: 4H, 1 año (~2190 velas), 5 pares KuCoin
Fees:  0.1% × 2 = 0.2% round-trip (taker KuCoin)

Uso:
    python backtests/annual_fee_backtest.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtests.kelly_backtest  import _run_kelly_trades, _kelly_metrics
from backtests.backtester_fast import _precompute_signals
from strategies.kelly_sizer    import KellySizer

from strategies.smc_strategy   import SMCStrategy
from strategies.strategy_zoo2  import (
    ParabolicSARStrategy, WilliamsRStrategy, HeikinAshiStrategy,
)
from strategies.strategy_zoo   import CCIStrategy
from strategies.swing_strategies import (
    ChandelierExitStrategy, TurtleBreakoutStrategy, MultiEMASwingStrategy,
)


# ── Configuración ─────────────────────────────────────────────────────────────

PAIRS   = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BNB-USDT"]
MAX_HOLD = 48     # 48 × 4h = 8 días máx por trade
PPY      = 2190   # 4H candles per year
SIZER    = KellySizer(variant="half_kelly", min_trades=20)
FEE_RT   = 0.002  # 0.2% round-trip

# ── Estrategias candidatas ────────────────────────────────────────────────────

STRATEGIES = [
    # ── Ganadoras Torneo Fee-Aware ───────────────────────────────────────
    ("Turtle(en15,ex7,et100)",
     TurtleBreakoutStrategy(entry_period=15, exit_period=7, ema_trend=100)),

    ("WR(p28,os-85,et100)",
     WilliamsRStrategy(period=28, oversold=-85, ema_trend=100)),

    ("WR(p21,os-85,et100)",
     WilliamsRStrategy(period=21, oversold=-85, ema_trend=100)),

    ("WR(p21,os-80,et100)",
     WilliamsRStrategy(period=21, oversold=-80, ema_trend=100)),

    ("MultiEMA(34/89/200,ADX>30)",
     MultiEMASwingStrategy(e1=34, e2=89, e3=200, adx_thresh=30)),

    ("SAR(af0.01,m0.15,et100)",
     ParabolicSARStrategy(af_start=0.01, af_max=0.15, ema_trend=100)),

    ("Chandelier(p14,m3,et200)",
     ChandelierExitStrategy(period=14, mult=3.0, ema_trend=200)),

    ("CCI(p20,t100,et100)",
     CCIStrategy(period=20, threshold=100, ema_trend=100)),

    ("HA(cb2,et100)",
     HeikinAshiStrategy(confirm_bars=2, ema_trend=100)),

    # ── Ganadoras Torneos R1/R2 (validadas en Fee Analysis anterior) ─────
    ("SMC(sw5,hold48)",
     SMCStrategy(swing_window=5)),

    ("HeikinAshi(cb2,et100) [R2]",
     HeikinAshiStrategy(confirm_bars=2, ema_trend=100)),

    ("SAR(et50) [R2]",
     ParabolicSARStrategy(af_start=0.02, af_max=0.2, ema_trend=50)),

    ("WR(p28,os-80) [R2]",
     WilliamsRStrategy(period=28, oversold=-80, ema_trend=50)),
]


# ── Motor de backtest ─────────────────────────────────────────────────────────

def _backtest_one(name, strat, datasets):
    all_m  = []
    all_eq = []

    for pair, df in datasets.items():
        try:
            if isinstance(strat, SMCStrategy):
                sigs = _precompute_signals(
                    df,
                    swing_window     = strat.swing_window,
                    require_fvg      = strat.require_fvg,
                    use_choch_filter = strat.use_choch_filter,
                )
            else:
                sigs = strat.generate_signals_batch(df)

            trades, eq = _run_kelly_trades(sigs, df, SIZER, max_hold=MAX_HOLD)
            m = _kelly_metrics(trades, eq, periods_per_year=PPY)
            all_m.append(m)
            all_eq.append(eq)
        except Exception as e:
            print(f"    ⚠ {pair} error: {e}")

    if not all_m:
        return None

    def avg(k):
        vals = [m[k] for m in all_m]
        return float(np.mean(vals))

    total_trades = sum(m["trades"] for m in all_m)
    n_pairs      = len(all_m)
    avg_frac     = avg("avg_fraction")

    gross_returns = [(m["equity_curve"][-1] - 1.0) * 100
                     for m in all_m if m["equity_curve"]]
    gross_pct     = float(np.mean(gross_returns)) if gross_returns else 0.0

    trades_pp     = total_trades / n_pairs if n_pairs else 0
    fee_cost_pct  = trades_pp * FEE_RT * avg_frac * 100
    net_pct       = gross_pct - fee_cost_pct

    gross_sharpe  = avg("sharpe")
    net_sharpe    = gross_sharpe * (net_pct / gross_pct) if abs(gross_pct) > 1e-6 else 0.0

    # Candles totales / meses (4h: ~180 velas/mes)
    candles_total = sum(len(df) for df in datasets.values())
    months        = candles_total / (180 * n_pairs) if n_pairs else 1
    t_month       = trades_pp / months

    # Equity curve media entre pares (para el gráfico)
    min_len = min(len(e) for e in all_eq)
    eq_mean = np.mean([e[:min_len] for e in all_eq], axis=0).tolist()

    return {
        "name":         name,
        "net_sharpe":   round(net_sharpe,   3),
        "gross_sharpe": round(gross_sharpe, 3),
        "net_pct":      round(net_pct,      2),
        "gross_pct":    round(gross_pct,    2),
        "fee_cost_pct": round(fee_cost_pct, 2),
        "winrate":      round(avg("winrate") * 100, 1),
        "profit_factor":round(avg("profit_factor"), 2),
        "max_dd":       round(avg("max_drawdown") * 100, 1),
        "t_month":      round(t_month, 1),
        "n_trades":     total_trades,
        "calmar":       round(avg("calmar"), 2),
        "eq_curve":     eq_mean,
    }


# ── Presentación ──────────────────────────────────────────────────────────────

HEADER = f"""
{'='*90}
  BACKTEST ANUAL — ESTRATEGIAS RENTABLES  (4H, 1 año, 5 pares KuCoin)
  Comisiones reales: KuCoin taker 0.1% × 2 = 0.2% round-trip
{'='*90}
"""

COL_FMT = (
    f"  {'#':>3}  {'Estrategia':<35}  "
    f"{'NetSh':>7}  {'Net%':>7}  {'Gross%':>7}  "
    f"{'Fees%':>6}  {'WR%':>5}  {'PF':>5}  "
    f"{'t/mes':>5}  {'MaxDD%':>7}  {'Calmar':>6}"
)


def print_results(rows):
    print(HEADER)
    profitable = [r for r in rows if r and r["net_pct"] > 0]
    losing     = [r for r in rows if r and r["net_pct"] <= 0]

    print(f"  Rentables post-fees: {len(profitable)}/{len(rows)}")
    print()
    print(COL_FMT)
    print("  " + "-" * 88)

    medals = ["🥇", "🥈", "🥉"]
    for i, r in enumerate(sorted(profitable, key=lambda x: -x["net_sharpe"])):
        m = medals[i] if i < 3 else "  "
        print(
            f"  {i+1:>3}. {m} {r['name']:<33}  "
            f"{r['net_sharpe']:>7.3f}  "
            f"{r['net_pct']:>+7.2f}%  "
            f"{r['gross_pct']:>+7.2f}%  "
            f"{r['fee_cost_pct']:>6.2f}%  "
            f"{r['winrate']:>5.1f}%  "
            f"{r['profit_factor']:>5.2f}  "
            f"{r['t_month']:>5.1f}  "
            f"{r['max_dd']:>7.1f}%  "
            f"{r['calmar']:>6.2f}"
        )

    if losing:
        print(f"\n  ── No rentables ({len(losing)}) ──")
        for r in sorted(losing, key=lambda x: -x["net_pct"]):
            print(
                f"  ✗  {r['name']:<35}  "
                f"net {r['net_pct']:>+7.2f}%  "
                f"(gross {r['gross_pct']:>+6.2f}% - fees {r['fee_cost_pct']:.2f}%)"
                f"  {r['t_month']:.1f} t/mes"
            )

    print(f"\n{'='*90}")
    if profitable:
        best = max(profitable, key=lambda x: x["net_sharpe"])
        best_ret = max(profitable, key=lambda x: x["net_pct"])
        print(f"  ★ MEJOR SHARPE NET  : {best['name']}  →  {best['net_sharpe']:.3f}")
        print(f"  ★ MEJOR RETORNO NET : {best_ret['name']}  →  {best_ret['net_pct']:+.2f}%/año")
    print(f"{'='*90}\n")


def save_plots(rows, datasets):
    profitable = [r for r in rows if r and r["net_pct"] > 0]
    if not profitable:
        return

    profitable = sorted(profitable, key=lambda x: -x["net_sharpe"])
    n = len(profitable)

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#0d1117")
    gs  = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)

    def _ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        return ax

    # ── 1. Equity curves ─────────────────────────────────────────────────
    ax1 = _ax(gs[0, :])
    colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(profitable)))
    for r, col in zip(profitable, colors):
        eq = np.array(r["eq_curve"])
        ax1.plot(eq, label=r["name"][:30], linewidth=1.5, color=col, alpha=0.9)
    ax1.axhline(1.0, color="#30363d", linestyle="--", linewidth=0.8)
    ax1.set_title("Equity Curves — 1 Año (media 5 pares)", color="#e6edf3", fontsize=11)
    ax1.set_ylabel("Capital (1.0 = inicio)", color="#8b949e")
    ax1.legend(fontsize=7, framealpha=0.2, labelcolor="#c9d1d9",
               loc="upper left", ncol=2)

    # ── 2. Net% bar chart ─────────────────────────────────────────────────
    ax2 = _ax(gs[1, 0])
    names_short = [r["name"][:22] for r in profitable]
    bar_colors  = ["#3fb950" if r["net_pct"] > 0 else "#f85149" for r in profitable]
    bars = ax2.barh(names_short, [r["net_pct"] for r in profitable],
                    color=bar_colors, alpha=0.85)
    ax2.axvline(0, color="#8b949e", linewidth=0.8)
    ax2.set_xlabel("Net Return % / año", color="#8b949e")
    ax2.set_title("Retorno Neto Annual", color="#e6edf3")
    ax2.tick_params(axis="y", labelsize=7.5, colors="#c9d1d9")
    # etiquetas
    for bar, r in zip(bars, profitable):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{r['net_pct']:+.1f}%", va="center", fontsize=7, color="#c9d1d9")

    # ── 3. Scatter: Trades/mes vs Net% (burbuja = Sharpe) ────────────────
    ax3 = _ax(gs[1, 1])
    sizes   = [max(30, r["net_sharpe"] * 200) for r in profitable]
    sc_cols = [r["net_sharpe"] for r in profitable]
    sc = ax3.scatter([r["t_month"] for r in profitable],
                     [r["net_pct"] for r in profitable],
                     s=sizes, c=sc_cols, cmap="plasma",
                     alpha=0.85, edgecolors="#30363d", linewidths=0.5)
    for r in profitable:
        ax3.annotate(r["name"][:18], (r["t_month"], r["net_pct"]),
                     fontsize=6.5, color="#8b949e",
                     xytext=(4, 4), textcoords="offset points")
    ax3.axhline(0, color="#30363d", linestyle="--", linewidth=0.8)
    ax3.set_xlabel("Trades / mes", color="#8b949e")
    ax3.set_ylabel("Net Return % / año", color="#8b949e")
    ax3.set_title("Frecuencia vs Retorno Neto", color="#e6edf3")
    cb = fig.colorbar(sc, ax=ax3, pad=0.02)
    cb.set_label("Net Sharpe", color="#8b949e", fontsize=8)
    cb.ax.tick_params(colors="#8b949e", labelsize=7)

    fig.suptitle(
        "Backtest Anual — Estrategias Rentables Post-Comisiones  |  4H KuCoin  |  5 Pares",
        color="#e6edf3", fontsize=12, y=0.99
    )

    out = "data/annual_backtest.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Gráfico guardado: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    from data.kucoin_client import KuCoinClient
    client = KuCoinClient()

    print("\n📡 Descargando 1 año de datos 4H de KuCoin (paginando)...")
    datasets = {}
    for pair in PAIRS:
        try:
            df = client.get_ohlcv_paginated(pair, interval="4hour", days=365)
            if df is not None and len(df) > 300:
                datasets[pair] = df
                start = df.index[0].strftime("%Y-%m-%d")
                end   = df.index[-1].strftime("%Y-%m-%d")
                print(f"  ✅ {pair}: {len(df)} velas  [{start} → {end}]")
            else:
                candles = len(df) if df is not None else 0
                print(f"  ⚠ {pair}: solo {candles} velas, omitido")
        except Exception as e:
            print(f"  ❌ {pair}: {e}")
        time.sleep(0.5)

    if not datasets:
        print("❌ Sin datos. Verifica conexión.")
        return

    print(f"\n  Pares activos: {list(datasets.keys())}")
    min_candles = min(len(d) for d in datasets.values())
    max_candles = max(len(d) for d in datasets.values())
    print(f"  Rango de velas: {min_candles} – {max_candles}")

    # ── Backtests ─────────────────────────────────────────────────────────
    print(f"\n  Corriendo {len(STRATEGIES)} estrategias × {len(datasets)} pares...\n")
    rows = []
    for name, strat in STRATEGIES:
        t0  = time.time()
        row = _backtest_one(name, strat, datasets)
        dt  = time.time() - t0
        if row:
            sign = "✅" if row["net_pct"] > 0 else "❌"
            print(f"  {sign} {name:<35}  net {row['net_pct']:>+7.2f}%  "
                  f"sharpe {row['net_sharpe']:>6.3f}  "
                  f"{row['t_month']:>4.1f} t/mes  [{dt:.1f}s]")
            rows.append(row)
        else:
            print(f"  ⚠ {name}: sin resultados")

    # ── Guardar CSV ───────────────────────────────────────────────────────
    rows_for_csv = [{k: v for k, v in r.items() if k != "eq_curve"} for r in rows]
    df_out = pd.DataFrame(rows_for_csv).sort_values("net_sharpe", ascending=False)
    csv_out = "data/annual_backtest_results.csv"
    df_out.to_csv(csv_out, index=False)
    print(f"\n  💾 CSV: {csv_out}")

    # ── Resumen en consola ────────────────────────────────────────────────
    print_results(rows)

    # ── Gráficos ─────────────────────────────────────────────────────────
    save_plots(rows, datasets)

    return rows


if __name__ == "__main__":
    main()
