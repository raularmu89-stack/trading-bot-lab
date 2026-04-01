"""
scenario_backtest.py

Backtest del ScenarioRouter vs estrategias individuales.
Muestra qué régimen predomina por par y cuánto aporta cada sub-estrategia.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

from backtests.kelly_backtest import _run_kelly_trades, _kelly_metrics
from strategies.scenario_router import (
    ScenarioRouter, precompute_router_signals_fast, REGIME_TO_STRATEGY
)
from strategies.kelly_sizer import KellySizer
from indicators.regime_detector import RegimeDetector

BASE_SLOPES = {
    "BTC": 0.004, "ETH": 0.005, "SOL": 0.007,
    "BNB": 0.003, "XAUT": 0.002, "LINK": 0.006,
}

def _gen(name, n, seed=None):
    rng = np.random.default_rng(seed or hash(name) % (2**31))
    prices = [100.0]
    sl = BASE_SLOPES.get(name, 0.004)
    phase_len = n // 8
    patterns = ["bull","sideways","bull","bear","sideways","bull","bear","bull"]
    phases = []
    for ph in patterns:
        v = sl if ph == "bull" else (-sl*0.75 if ph == "bear" else 0.0)
        phases.extend([v] * phase_len)
    while len(phases) < n:
        phases.append(sl * 0.3)
    vol = 0.6
    for i in range(n - 1):
        shock = rng.standard_normal() * vol
        vol = np.clip(0.95 * vol + 0.05 * (abs(shock) * 0.5 + 0.3), 0.2, 2.5)
        prices.append(max(1.0, prices[-1] * (1 + phases[i]/100 + shock/100)))
    prices = np.array(prices)
    spread = abs(rng.standard_normal(n)) * prices * 0.002
    return pd.DataFrame({
        "open": prices, "high": prices + spread,
        "low": np.maximum(prices - spread, prices * 0.98),
        "close": prices,
        "volume": rng.integers(5000, 80000, n).astype(float),
    })


PAIRS   = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XAUT/USDT", "LINK/USDT"]
N_15M   = 35_040   # 1 año en 15m
CAPITAL = 1_000.0
MAX_HOLD = 8


def run_all():
    sizer = KellySizer(variant="half_kelly", min_trades=20, max_fraction=0.30)
    router = ScenarioRouter(swing_window=5, verbose=False)

    results  = []
    all_eq   = {}
    all_reg  = {}

    print(f"\n{'='*80}")
    print(f"  SCENARIO ROUTER BACKTEST — 6 pares × 1 año × 15m")
    print(f"  Estrategia activa según régimen de mercado detectado")
    print(f"{'='*80}")

    for pair in PAIRS:
        name = pair.split("/")[0]
        data = _gen(name, N_15M)

        print(f"\n  Par: {pair}")

        # ── Régimen por vela ──────────────────────────────────────────
        regimes = router.detector.detect_all(data)
        reg_count = Counter(regimes)
        total = len(regimes)

        # ── Señales del router ────────────────────────────────────────
        signals = precompute_router_signals_fast(data, router)

        trades, eq = _run_kelly_trades(signals, data, sizer, MAX_HOLD)
        m = _kelly_metrics(trades, eq, periods_per_year=N_15M)

        final_eq = m["equity_curve"][-1] * CAPITAL
        pct      = (final_eq - CAPITAL) / CAPITAL * 100
        t_month  = m["trades"] / 12

        all_eq[pair]  = m["equity_curve"]
        all_reg[pair] = reg_count

        results.append({
            "par":          pair,
            "trades":       m["trades"],
            "trades_month": round(t_month, 1),
            "winrate":      round(m["winrate"] * 100, 1),
            "compuesto":    round(pct, 1),
            "final_eur":    round(final_eq, 1),
            "sharpe":       m["sharpe"],
            "sortino":      m["sortino"],
            "max_dd":       round(m["max_drawdown"] * 100, 1),
        })

        # Distribución de regímenes
        reg_str = "  ".join(
            f"{RegimeDetector.regime_label(r).split()[0]}:{c/total*100:.0f}%"
            for r, c in reg_count.most_common(4)
        )
        print(f"    Trades:{m['trades']:5d} (~{t_month:5.1f}/mes)  "
              f"WR:{m['winrate']*100:5.1f}%  Ret:{pct:+8.1f}%  "
              f"Sharpe:{m['sharpe']:5.2f}  DD:{m['max_drawdown']*100:5.1f}%")
        print(f"    Regímenes: {reg_str}")

    df = pd.DataFrame(results)
    _print_summary(df)
    _save_plots(df, all_eq, all_reg)
    df.to_csv("data/scenario_backtest_results.csv", index=False)
    print("\n  Guardado: data/scenario_backtest_results.csv")
    return df


def _print_summary(df):
    print(f"\n{'='*80}")
    print("  RESUMEN AGREGADO — ScenarioRouter (media 6 pares)")
    print(f"{'='*80}")
    r = df.mean(numeric_only=True)
    print(f"  Trades/mes:  {r['trades_month']:.1f}")
    print(f"  Win Rate:    {r['winrate']:.1f}%")
    print(f"  Retorno:     {r['compuesto']:+.1f}%")
    print(f"  Capital:     {r['final_eur']:.1f}€ (desde {1000}€)")
    print(f"  Sharpe:      {r['sharpe']:.2f}")
    print(f"  Sortino:     {r['sortino']:.2f}")
    print(f"  Max DD:      {r['max_dd']:.1f}%")

    print(f"\n  {'Par':12s}  {'T/mes':>6}  {'WR%':>6}  {'Ret%':>9}  "
          f"{'Final€':>8}  {'Sharpe':>7}  {'DD%':>6}")
    print(f"  {'-'*62}")
    for _, row in df.iterrows():
        print(f"  {row['par']:12s}  {row['trades_month']:6.1f}  "
              f"{row['winrate']:6.1f}%  {row['compuesto']:+9.1f}%  "
              f"{row['final_eur']:8.1f}€  {row['sharpe']:7.2f}  {row['max_dd']:5.1f}%")


def _save_plots(df, all_eq, all_reg):
    colors_reg = {
        "strong_trend_bull":  "#2ca02c",
        "strong_trend_bear":  "#d62728",
        "weak_trend_bull":    "#98df8a",
        "weak_trend_bear":    "#ff9896",
        "ranging":            "#aec7e8",
        "breakout":           "#ffbb78",
        "mean_reversion_bull":"#17becf",
        "mean_reversion_bear":"#9467bd",
        "high_volatility":    "#7f7f7f",
        "insufficient_data":  "#c7c7c7",
    }

    # ── Fig 1: Equity curves ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("ScenarioRouter — Equity Curves por Par (15m, Half-Kelly)",
                 fontsize=14, fontweight="bold")
    for ax, pair in zip(axes.flat, PAIRS):
        eq = all_eq[pair]
        x  = np.linspace(0, 12, len(eq))  # meses
        ax.plot(x, eq, color="#1f77b4", linewidth=1.3)
        ax.fill_between(x, 1.0, eq, where=[e >= 1.0 for e in eq],
                        alpha=0.2, color="green")
        ax.fill_between(x, 1.0, eq, where=[e < 1.0 for e in eq],
                        alpha=0.2, color="red")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(pair, fontweight="bold")
        ax.set_xlabel("Mes")
        ax.set_ylabel("Equity (×inicial)")
        ax.grid(True, alpha=0.3)
        # Anotar retorno final
        final = eq[-1]
        ax.annotate(f"{(final-1)*100:+.1f}%",
                    xy=(12, final), xytext=(-30, 10),
                    textcoords="offset points", fontsize=10,
                    fontweight="bold",
                    color="green" if final >= 1 else "red")
    plt.tight_layout()
    plt.savefig("data/plots/20_scenario_equity.png", dpi=130)
    plt.close()
    print("  Guardado: data/plots/20_scenario_equity.png")

    # ── Fig 2: Distribución de regímenes por par ──────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Distribución de Regímenes de Mercado por Par",
                 fontsize=14, fontweight="bold")
    for ax, pair in zip(axes.flat, PAIRS):
        reg_count = all_reg[pair]
        total     = sum(reg_count.values())
        labels    = [RegimeDetector.regime_label(r).replace(" ", "\n")
                     for r in reg_count.keys()]
        sizes     = [c / total * 100 for c in reg_count.values()]
        c_list    = [colors_reg.get(r, "#cccccc") for r in reg_count.keys()]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=c_list,
            autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 7},
        )
        for at in autotexts:
            at.set_fontsize(7)
        ax.set_title(pair, fontweight="bold")
    plt.tight_layout()
    plt.savefig("data/plots/21_regime_distribution.png", dpi=130)
    plt.close()
    print("  Guardado: data/plots/21_regime_distribution.png")

    # ── Fig 3: Barras comparativas ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("ScenarioRouter vs Baseline — Comparativa", fontsize=13, fontweight="bold")
    pairs = df["par"].tolist()
    x = range(len(pairs))

    for ax, (col, title, color) in zip(axes, [
        ("compuesto", "Retorno Anual (%)", "#2ca02c"),
        ("sharpe",    "Sharpe Ratio",      "#1f77b4"),
        ("max_dd",    "Max Drawdown (%)",  "#d62728"),
    ]):
        vals = df[col].tolist()
        bars = ax.bar(x, vals, color=color, alpha=0.80, edgecolor="white")
        ax.set_xticks(list(x))
        ax.set_xticklabels(pairs, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(max(vals, default=1)) * 0.02,
                    f"{v:.1f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("data/plots/22_scenario_vs_baseline.png", dpi=130)
    plt.close()
    print("  Guardado: data/plots/22_scenario_vs_baseline.png")


if __name__ == "__main__":
    run_all()
