"""
pro_backtest.py

Backtest pro: compara 6 configuraciones con Kelly sizing dinámico.
Datos: BTC, ETH, SOL, BNB en 1h, 1 año.

Estrategias:
  1. Pine Swing (w=50) — mejor resultado anterior
  2. Pine Swing + Half-Kelly sizing
  3. Pine Swing + Quarter-Kelly sizing
  4. MTF 4h+1h + Half-Kelly
  5. Confluence + Half-Kelly
  6. Portfolio: Pine Swing ∥ MTF ∥ Confluence (mejor señal cada vela)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

from backtests.backtester_fast import FastBacktester
from backtests.kelly_backtest import KellyBacktester
from strategies.smc_strategy import SMCStrategy
from strategies.mtf_strategy import MTFStrategy
from strategies.confluence_strategy import ConfluenceStrategy
from strategies.kelly_sizer import KellySizer


# ── Generador de datos mock realistas ─────────────────────────────────────────

def _generate_pair(name, n=8760, seed=None):
    """
    Genera OHLCV de 1 año (8760 velas de 1h) con régimen de tendencias.
    Simula fases bull/bear/lateral alternantes.
    """
    rng = np.random.default_rng(seed or hash(name) % (2**31))
    prices = [100.0]

    phase_len = n // 6
    phases = []
    base_slopes = {"BTC": 0.004, "ETH": 0.005, "SOL": 0.007, "BNB": 0.003,
                   "XAUT": 0.002, "LINK": 0.006}
    slope_base = base_slopes.get(name.split("/")[0], 0.004)

    phase_types = ["bull", "bear", "sideways", "bull", "bear", "bull"]
    for ph in phase_types:
        if ph == "bull":
            phases.extend([slope_base] * phase_len)
        elif ph == "bear":
            phases.extend([-slope_base * 0.8] * phase_len)
        else:
            phases.extend([0.0] * phase_len)

    while len(phases) < n:
        phases.append(slope_base * 0.3)

    vol = 0.8
    for i in range(n - 1):
        shock = rng.standard_normal() * vol
        # Volatility clustering
        vol = 0.95 * vol + 0.05 * (abs(shock) * 0.6 + 0.4)
        vol = np.clip(vol, 0.3, 2.0)
        prices.append(max(1.0, prices[-1] * (1 + phases[i] / 100 + shock / 100)))

    prices = np.array(prices)
    spread = abs(rng.standard_normal(n)) * prices * 0.003
    high   = prices + spread
    low    = prices - spread
    low    = np.maximum(low, prices * 0.97)
    vol_v  = rng.integers(5000, 50000, n).astype(float)

    return pd.DataFrame({
        "open": prices, "high": high, "low": low,
        "close": prices, "volume": vol_v,
    })


# ── Configuraciones de estrategia ─────────────────────────────────────────────

def build_configs():
    sizer_half    = KellySizer(variant="half_kelly",    min_trades=20, max_fraction=0.30)
    sizer_quarter = KellySizer(variant="quarter_kelly", min_trades=20, max_fraction=0.20)
    sizer_fixed   = KellySizer(variant="fixed_pct",     fixed_pct=0.02)

    configs = [
        {
            "name": "Pine Swing w=50\n(Fixed 2%)",
            "short": "PineSwing-Fixed",
            "strategy": SMCStrategy(swing_window=50),
            "sizer":    sizer_fixed,
            "kelly":    False,
        },
        {
            "name": "Pine Swing w=50\n(Half-Kelly)",
            "short": "PineSwing-HK",
            "strategy": SMCStrategy(swing_window=50),
            "sizer":    sizer_half,
            "kelly":    True,
        },
        {
            "name": "Pine Swing w=50\n(Quarter-Kelly)",
            "short": "PineSwing-QK",
            "strategy": SMCStrategy(swing_window=50),
            "sizer":    sizer_quarter,
            "kelly":    True,
        },
        {
            "name": "MTF 4h+1h\n(Half-Kelly)",
            "short": "MTF-HK",
            "strategy": MTFStrategy(high_tf="4h", high_tf_window=10,
                                     low_tf_window=5, use_pd_filter=False),
            "sizer":    sizer_half,
            "kelly":    True,
        },
        {
            "name": "Confluence\n(Half-Kelly)",
            "short": "Confluence-HK",
            "strategy": ConfluenceStrategy(swing_window=10, require_sweep=False,
                                            require_fvg_or_ob=False),
            "sizer":    sizer_half,
            "kelly":    True,
        },
    ]
    return configs


# ── Portfolio: combina señales de varias estrategias ──────────────────────────

def run_portfolio(data, sizer, pairs_data, max_hold=20):
    """
    Portfolio multi-estrategia: corre Pine Swing + MTF + Confluence en paralelo.
    En cada vela toma la señal de mayoría (2+ de 3); si hay empate, hold.
    Retorna métricas del portfolio combinado.
    """
    from backtests.backtester_fast import _precompute_signals
    from backtests.kelly_backtest import _run_kelly_trades, _kelly_metrics

    strategies = [
        SMCStrategy(swing_window=50),
        MTFStrategy(high_tf="4h", high_tf_window=10, low_tf_window=5, use_pd_filter=False),
        ConfluenceStrategy(swing_window=10, require_sweep=False, require_fvg_or_ob=False),
    ]

    # Precomputa señales para cada estrategia
    all_signals = []
    for strat in strategies:
        sigs = _precompute_signals(
            data,
            swing_window=strat.swing_window,
            require_fvg=strat.require_fvg,
            use_choch_filter=strat.use_choch_filter,
        )
        all_signals.append(sigs)

    n = len(data)
    combined = []
    for i in range(n):
        votes = [s[i] for s in all_signals]
        buys  = votes.count("buy")
        sells = votes.count("sell")
        if buys >= 2:
            combined.append("buy")
        elif sells >= 2:
            combined.append("sell")
        else:
            combined.append("hold")

    trades, equity_curve = _run_kelly_trades(combined, data, sizer, max_hold)
    return _kelly_metrics(trades, equity_curve)


# ── Runner principal ──────────────────────────────────────────────────────────

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XAUT/USDT", "LINK/USDT"]
MAX_HOLD = 20
INITIAL_CAPITAL = 1000.0

def run_all():
    configs = build_configs()
    sizer_portfolio = KellySizer(variant="half_kelly", min_trades=20, max_fraction=0.30)

    # Nombre completo incluyendo portfolio
    col_names = [c["short"] for c in configs] + ["Portfolio-HK"]
    results = []

    print(f"\n{'='*78}")
    print(f"  PRO BACKTEST — Kelly Sizing Dinámico")
    print(f"  {len(PAIRS)} pares × 1 año × 1h | Capital inicial: {INITIAL_CAPITAL}€")
    print(f"{'='*78}")

    pair_data = {}
    for pair in PAIRS:
        name = pair.split("/")[0]
        pair_data[pair] = _generate_pair(name, n=8760)

    all_metrics = {c: [] for c in col_names}

    for pair in PAIRS:
        data = pair_data[pair]
        print(f"\n  Par: {pair}")

        for cfg in configs:
            strat  = cfg["strategy"]
            sizer  = cfg["sizer"]
            short  = cfg["short"]

            if cfg["kelly"]:
                bt = KellyBacktester(strat, data, sizer=sizer, max_hold=MAX_HOLD)
            else:
                bt = FastBacktester(strat, data, max_hold=MAX_HOLD)

            m = bt.run(periods_per_year=8760)
            final_eq = m["equity_curve"][-1] * INITIAL_CAPITAL
            pct      = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

            all_metrics[short].append(m)
            results.append({
                "par":      pair,
                "config":   short,
                "trades":   m["trades"],
                "winrate":  round(m["winrate"] * 100, 1),
                "compuesto": round(pct, 1),
                "final_eur": round(final_eq, 1),
                "sharpe":   m["sharpe"],
                "sortino":  m["sortino"],
                "max_dd":   round(m["max_drawdown"] * 100, 1),
                "avg_frac": m.get("avg_fraction", 0.02),
            })
            print(f"    {short:20s} Trades:{m['trades']:4d}  WR:{m['winrate']*100:5.1f}%  "
                  f"Ret:{pct:+8.1f}%  Sharpe:{m['sharpe']:5.2f}  DD:{m['max_drawdown']*100:5.1f}%")

        # Portfolio
        m_port = run_portfolio(data, sizer_portfolio, pair_data)
        final_eq = m_port["equity_curve"][-1] * INITIAL_CAPITAL
        pct      = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        all_metrics["Portfolio-HK"].append(m_port)
        results.append({
            "par":      pair,
            "config":   "Portfolio-HK",
            "trades":   m_port["trades"],
            "winrate":  round(m_port["winrate"] * 100, 1),
            "compuesto": round(pct, 1),
            "final_eur": round(final_eq, 1),
            "sharpe":   m_port["sharpe"],
            "sortino":  m_port["sortino"],
            "max_dd":   round(m_port["max_drawdown"] * 100, 1),
            "avg_frac": m_port.get("avg_fraction", 0.0),
        })
        print(f"    {'Portfolio-HK':20s} Trades:{m_port['trades']:4d}  "
              f"WR:{m_port['winrate']*100:5.1f}%  Ret:{pct:+8.1f}%  "
              f"Sharpe:{m_port['sharpe']:5.2f}  DD:{m_port['max_drawdown']*100:5.1f}%")

    df = pd.DataFrame(results)
    _print_summary(df)
    _save_plots(df, all_metrics, col_names, pair_data)

    out = "data/pro_backtest_results.csv"
    df.to_csv(out, index=False)
    print(f"\n  Resultados guardados en {out}")
    return df


def _print_summary(df):
    print(f"\n{'='*92}")
    print(f"  RESUMEN AGREGADO (media de {len(df['par'].unique())} pares)")
    print(f"{'='*92}")
    print(f"  {'Config':22s}  {'Trades':>7}  {'WR%':>6}  {'Ret%':>9}  "
          f"{'Final€':>8}  {'Sharpe':>7}  {'Sortino':>8}  {'DD%':>7}  {'Frac':>6}")
    print(f"  {'-'*84}")

    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg]
        print(f"  {cfg:22s}  "
              f"{sub['trades'].mean():7.0f}  "
              f"{sub['winrate'].mean():6.1f}%  "
              f"{sub['compuesto'].mean():+9.1f}%  "
              f"{sub['final_eur'].mean():8.1f}€  "
              f"{sub['sharpe'].mean():7.2f}  "
              f"{sub['sortino'].mean():8.2f}  "
              f"{sub['max_dd'].mean():6.1f}%  "
              f"{sub['avg_frac'].mean():.4f}")


def _save_plots(df, all_metrics, col_names, pair_data):
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#b07aa1"]
    pairs  = list(pair_data.keys())

    # ── Figura 1: Equity curves por par ───────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Pro Backtest — Equity Curves por Par (Kelly Sizing)", fontsize=14, fontweight="bold")

    for ax, pair in zip(axes.flat, pairs):
        for j, col in enumerate(col_names):
            idx = PAIRS.index(pair)
            eq  = all_metrics[col][idx]["equity_curve"]
            ax.plot(eq, color=colors[j], linewidth=1.2,
                    label=col.replace("\n", " "), alpha=0.85)
        ax.set_title(pair, fontweight="bold")
        ax.set_ylabel("Equity (×inicial)")
        ax.set_xlabel("Velas (1h)")
        ax.legend(fontsize=6.5, loc="upper left")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/plots/15_pro_equity_curves.png", dpi=130)
    plt.close()
    print("  Guardado: data/plots/15_pro_equity_curves.png")

    # ── Figura 2: Comparativa de métricas agregadas ───────────────────────────
    agg = df.groupby("config").agg({
        "compuesto": "mean",
        "sharpe":    "mean",
        "max_dd":    "mean",
        "winrate":   "mean",
        "avg_frac":  "mean",
    }).reindex(col_names)

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Pro Backtest — Métricas Comparativas (media 4 pares)", fontsize=14, fontweight="bold")

    metrics_plot = [
        (gs[0, 0], "compuesto",  "Retorno Compuesto (%)",  True),
        (gs[0, 1], "sharpe",     "Sharpe Ratio",           False),
        (gs[0, 2], "sortino",    "Sortino Ratio",          False),
        (gs[1, 0], "max_dd",     "Max Drawdown (%)",       False),
        (gs[1, 1], "winrate",    "Win Rate (%)",           False),
        (gs[1, 2], "avg_frac",   "Fracción Kelly Media",   False),
    ]

    # sortino separado
    agg_sortino = df.groupby("config")["sortino"].mean().reindex(col_names)

    for spec, key, title, highlight in metrics_plot:
        ax = fig.add_subplot(spec)
        vals   = agg[key] if key in agg.columns else agg_sortino
        xpos   = range(len(col_names))
        bar_colors = [colors[i] for i in range(len(col_names))]
        bars = ax.bar(xpos, vals, color=bar_colors, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(xpos)
        ax.set_xticklabels(col_names, fontsize=7.5, rotation=30, ha="right")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(vals.max()) * 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    plt.savefig("data/plots/16_pro_metrics.png", dpi=130)
    plt.close()
    print("  Guardado: data/plots/16_pro_metrics.png")


if __name__ == "__main__":
    run_all()
