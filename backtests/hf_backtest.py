"""
hf_backtest.py  — High-Frequency Pro Backtest

Objetivo: maximizar trades/mes manteniendo WR > 55% y Sharpe > 2.

Configuraciones comparadas (todas con Half-Kelly):
  1. 1h w=50  (baseline anterior)
  2. 1h w=5   (más señales en 1h)
  3. 15m w=5  (4× datos, 4× señales)
  4. 15m w=5 + FVG  (FVG como trigger adicional)
  5. 15m w=5 + MTF (1h dirección + 15m entrada)
  6. 15m w=3  (ultrarápido)

6 pares: BTC, ETH, SOL, BNB, XAUT, LINK
Muestra: trades/mes por par + resumen agregado
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from backtests.kelly_backtest import KellyBacktester
from backtests.backtester_fast import FastBacktester, _precompute_signals
from backtests.kelly_backtest import _run_kelly_trades, _kelly_metrics
from strategies.smc_strategy import SMCStrategy
from strategies.mtf_strategy import MTFStrategy
from strategies.kelly_sizer import KellySizer


# ── Generador OHLCV multi-timeframe ──────────────────────────────────────────

BASE_SLOPES = {
    "BTC": 0.004, "ETH": 0.005, "SOL": 0.007,
    "BNB": 0.003, "XAUT": 0.002, "LINK": 0.006,
}

def _generate_pair(name, n, seed=None):
    """
    Genera n velas OHLCV con fases bull/bear/lateral alternantes.
    Volatility clustering incluido.
    """
    rng = np.random.default_rng(seed or hash(name) % (2**31))
    prices = [100.0]
    slope_base = BASE_SLOPES.get(name, 0.004)

    n_phases = 8
    phase_len = n // n_phases
    phase_types = ["bull", "sideways", "bull", "bear",
                   "sideways", "bull", "bear", "bull"]
    phases = []
    for ph in phase_types:
        if ph == "bull":
            phases.extend([slope_base] * phase_len)
        elif ph == "bear":
            phases.extend([-slope_base * 0.75] * phase_len)
        else:
            phases.extend([0.0] * phase_len)
    while len(phases) < n:
        phases.append(slope_base * 0.3)

    vol = 0.6
    for i in range(n - 1):
        shock = rng.standard_normal() * vol
        vol = np.clip(0.95 * vol + 0.05 * (abs(shock) * 0.5 + 0.3), 0.2, 2.5)
        prices.append(max(1.0, prices[-1] * (1 + phases[i] / 100 + shock / 100)))

    prices = np.array(prices)
    spread = abs(rng.standard_normal(n)) * prices * 0.002
    high = prices + spread
    low  = prices - spread
    low  = np.maximum(low, prices * 0.98)
    return pd.DataFrame({
        "open": prices, "high": high, "low": low,
        "close": prices,
        "volume": rng.integers(5000, 80000, n).astype(float),
    })


def _block_resample(df, factor):
    """Agrupa velas de 15m a 1h (factor=4) sin necesitar DatetimeIndex."""
    n = len(df) // factor
    rows = []
    for i in range(n):
        sl = df.iloc[i * factor: (i + 1) * factor]
        rows.append({
            "open":   sl["open"].iloc[0],
            "high":   sl["high"].max(),
            "low":    sl["low"].min(),
            "close":  sl["close"].iloc[-1],
            "volume": sl["volume"].sum(),
        })
    return pd.DataFrame(rows)


# ── Backtest MTF sobre 15m (dirección 1h, entrada 15m) ───────────────────────

def _run_mtf_15m(df_15m, sizer, max_hold, high_tf_window=10, low_tf_window=5):
    """
    Usa las velas 1h (bloque de 4×15m) para detectar dirección,
    luego genera entradas en el timeframe de 15m.
    """
    df_1h = _block_resample(df_15m, 4)

    # Señales de dirección en 1h
    dir_signals = _precompute_signals(
        df_1h,
        swing_window=high_tf_window,
        require_fvg=False,
        use_choch_filter=False,
    )
    # forward-fill dirección a 15m
    dir_expanded = []
    for s in dir_signals:
        dir_expanded.extend([s] * 4)
    while len(dir_expanded) < len(df_15m):
        dir_expanded.append("hold")
    dir_expanded = dir_expanded[:len(df_15m)]

    # Señales de entrada en 15m
    entry_signals = _precompute_signals(
        df_15m,
        swing_window=low_tf_window,
        require_fvg=False,
        use_choch_filter=True,
    )

    # Combinar: sólo entra si la dirección 1h coincide
    combined = []
    for d, e in zip(dir_expanded, entry_signals):
        if e == "buy" and d in ("buy", "hold"):
            combined.append("buy")
        elif e == "sell" and d in ("sell", "hold"):
            combined.append("sell")
        else:
            combined.append("hold")

    trades, eq = _run_kelly_trades(combined, df_15m, sizer, max_hold)
    return _kelly_metrics(trades, eq, periods_per_year=35040)


# ── Configuraciones ───────────────────────────────────────────────────────────

def _configs(sizer_hk):
    return [
        {
            "name":    "1h w=50\n(baseline)",
            "short":   "1h-w50",
            "candles": 8_760,
            "tf":      "1h",
            "strat":   SMCStrategy(swing_window=50),
            "max_hold": 20,
            "ppy":     8_760,
            "fvg":     False,
            "mtf15":   False,
        },
        {
            "name":    "1h w=5\n(más señales)",
            "short":   "1h-w5",
            "candles": 8_760,
            "tf":      "1h",
            "strat":   SMCStrategy(swing_window=5),
            "max_hold": 10,
            "ppy":     8_760,
            "fvg":     False,
            "mtf15":   False,
        },
        {
            "name":    "15m w=5\n(alta freq)",
            "short":   "15m-w5",
            "candles": 35_040,
            "tf":      "15m",
            "strat":   SMCStrategy(swing_window=5),
            "max_hold": 8,
            "ppy":     35_040,
            "fvg":     False,
            "mtf15":   False,
        },
        {
            "name":    "15m w=5+FVG\n(calidad+frecuencia)",
            "short":   "15m-w5-FVG",
            "candles": 35_040,
            "tf":      "15m",
            "strat":   SMCStrategy(swing_window=5, require_fvg=True),
            "max_hold": 8,
            "ppy":     35_040,
            "fvg":     True,
            "mtf15":   False,
        },
        {
            "name":    "15m w=5 MTF\n(1h dir+15m entrada)",
            "short":   "15m-MTF",
            "candles": 35_040,
            "tf":      "15m",
            "strat":   None,   # usa _run_mtf_15m
            "max_hold": 8,
            "ppy":     35_040,
            "fvg":     False,
            "mtf15":   True,
        },
        {
            "name":    "15m w=3\n(ultrarápido)",
            "short":   "15m-w3",
            "candles": 35_040,
            "tf":      "15m",
            "strat":   SMCStrategy(swing_window=3),
            "max_hold": 5,
            "ppy":     35_040,
            "fvg":     False,
            "mtf15":   False,
        },
    ]


# ── Runner ────────────────────────────────────────────────────────────────────

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XAUT/USDT", "LINK/USDT"]
INITIAL_CAPITAL = 1000.0

def run_all():
    sizer_hk = KellySizer(variant="half_kelly", min_trades=20, max_fraction=0.30)
    configs   = _configs(sizer_hk)
    col_names = [c["short"] for c in configs]

    results = []
    all_metrics = {c: [] for c in col_names}

    print(f"\n{'='*86}")
    print(f"  HF PRO BACKTEST — Máximos trades/mes con Kelly Half")
    print(f"  {len(PAIRS)} pares × 1 año | Capital: {INITIAL_CAPITAL}€")
    print(f"{'='*86}")

    for pair in PAIRS:
        name = pair.split("/")[0]
        print(f"\n  Par: {pair}")

        # Genera datos para cada timeframe necesario
        data_cache = {}
        for cfg in configs:
            n = cfg["candles"]
            if n not in data_cache:
                data_cache[n] = _generate_pair(name, n)

        for cfg in configs:
            data   = data_cache[cfg["candles"]]
            short  = cfg["short"]
            mh     = cfg["max_hold"]
            ppy    = cfg["ppy"]

            if cfg["mtf15"]:
                m = _run_mtf_15m(data, sizer_hk, mh)
            else:
                bt = KellyBacktester(cfg["strat"], data, sizer=sizer_hk,
                                     max_hold=mh)
                m  = bt.run(periods_per_year=ppy)

            final_eq      = m["equity_curve"][-1] * INITIAL_CAPITAL
            pct           = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            trades_month  = round(m["trades"] / 12, 1)

            all_metrics[short].append(m)
            results.append({
                "par":           pair,
                "config":        short,
                "tf":            cfg["tf"],
                "trades_total":  m["trades"],
                "trades_month":  trades_month,
                "winrate":       round(m["winrate"] * 100, 1),
                "compuesto":     round(pct, 1),
                "final_eur":     round(final_eq, 1),
                "sharpe":        m["sharpe"],
                "sortino":       m["sortino"],
                "max_dd":        round(m["max_drawdown"] * 100, 1),
                "avg_frac":      m.get("avg_fraction", 0.0),
            })
            print(f"    {short:14s}  Trades:{m['trades']:5d} (~{trades_month:5.1f}/mes)"
                  f"  WR:{m['winrate']*100:5.1f}%  Ret:{pct:+8.1f}%"
                  f"  Sharpe:{m['sharpe']:5.2f}  DD:{m['max_drawdown']*100:5.1f}%")

    df = pd.DataFrame(results)
    _print_summary(df)
    _save_plots(df, all_metrics, col_names)

    out = "data/hf_backtest_results.csv"
    df.to_csv(out, index=False)
    print(f"\n  Resultados guardados en {out}")
    return df


def _print_summary(df):
    pairs_n = df["par"].nunique()
    print(f"\n{'='*100}")
    print(f"  RESUMEN AGREGADO — media de {pairs_n} pares")
    print(f"{'='*100}")
    print(f"  {'Config':14s}  {'TF':>4}  {'T/mes':>6}  {'WR%':>6}  {'Ret%':>9}  "
          f"{'Final€':>8}  {'Sharpe':>7}  {'Sortino':>8}  {'DD%':>7}")
    print(f"  {'-'*88}")

    tf_map = {r["config"]: r["tf"] for _, r in df.iterrows()}
    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg]
        print(f"  {cfg:14s}  {tf_map[cfg]:>4}  "
              f"{sub['trades_month'].mean():6.1f}  "
              f"{sub['winrate'].mean():6.1f}%  "
              f"{sub['compuesto'].mean():+9.1f}%  "
              f"{sub['final_eur'].mean():8.1f}€  "
              f"{sub['sharpe'].mean():7.2f}  "
              f"{sub['sortino'].mean():8.2f}  "
              f"{sub['max_dd'].mean():6.1f}%")


def _save_plots(df, all_metrics, col_names):
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#b07aa1"]
    pairs  = PAIRS

    # ── Fig 1: Trades/mes por par y configuración ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Trades por Mes — por Par y Configuración", fontsize=14, fontweight="bold")

    for ax, pair in zip(axes.flat, pairs):
        sub     = df[df["par"] == pair]
        configs = sub["config"].tolist()
        tmonth  = sub["trades_month"].tolist()
        wr      = sub["winrate"].tolist()
        bars = ax.bar(configs, tmonth, color=colors[:len(configs)], alpha=0.85, edgecolor="white")
        ax2 = ax.twinx()
        ax2.plot(configs, wr, "D--", color="#333", markersize=5, linewidth=1.2, label="WR%")
        ax2.set_ylim(40, 80)
        ax2.set_ylabel("WR%", fontsize=8)
        ax.set_title(pair, fontweight="bold")
        ax.set_ylabel("Trades/mes")
        ax.set_xticklabels(configs, rotation=35, ha="right", fontsize=7.5)
        ax.axhline(20, color="gray", linestyle=":", linewidth=0.8)
        for bar, v in zip(bars, tmonth):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, f"{v:.0f}", ha="center", fontsize=7.5)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/plots/17_trades_per_month.png", dpi=130)
    plt.close()
    print("  Guardado: data/plots/17_trades_per_month.png")

    # ── Fig 2: Retorno vs Trades/mes (scatter) ────────────────────────────────
    agg = df.groupby("config").agg(
        trades_month=("trades_month", "mean"),
        compuesto=("compuesto", "mean"),
        sharpe=("sharpe", "mean"),
        max_dd=("max_dd", "mean"),
        winrate=("winrate", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("HF Backtest — Retorno vs Frecuencia vs Riesgo", fontsize=13, fontweight="bold")

    for i, (ax, (ycol, ylabel)) in enumerate(zip(axes, [
        ("compuesto", "Retorno Anual (%)"),
        ("sharpe",    "Sharpe Ratio"),
        ("max_dd",    "Max Drawdown (%)"),
    ])):
        for j, row in agg.iterrows():
            ax.scatter(row["trades_month"], row[ycol],
                       color=colors[j % len(colors)], s=120, zorder=5)
            ax.annotate(row["config"],
                        (row["trades_month"], row[ycol]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("Trades/mes (media)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.8)

    plt.tight_layout()
    plt.savefig("data/plots/18_freq_vs_return.png", dpi=130)
    plt.close()
    print("  Guardado: data/plots/18_freq_vs_return.png")

    # ── Fig 3: Equity curves para los mejores configs ─────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Equity Curves — Todos los Pares (Half-Kelly)", fontsize=14, fontweight="bold")

    for ax, pair in zip(axes.flat, pairs):
        idx = pairs.index(pair)
        for j, col in enumerate(col_names):
            eq = all_metrics[col][idx]["equity_curve"]
            # Normaliza longitud para comparación visual
            x = np.linspace(0, 1, len(eq))
            ax.plot(x, eq, color=colors[j], linewidth=1.1,
                    label=col, alpha=0.80)
        ax.set_title(pair, fontweight="bold")
        ax.set_ylabel("Equity (×inicial)")
        ax.set_xlabel("Tiempo normalizado")
        ax.legend(fontsize=6.5, loc="upper left")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/plots/19_hf_equity_curves.png", dpi=130)
    plt.close()
    print("  Guardado: data/plots/19_hf_equity_curves.png")


if __name__ == "__main__":
    run_all()
