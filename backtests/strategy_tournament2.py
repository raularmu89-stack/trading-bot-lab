"""
strategy_tournament2.py

Torneo Ronda 2 — 100 nuevas configuraciones.
Incluye los 10 tipos nuevos + variantes finas de los ganadores de la ronda 1
(CCI y Engulfing).

Uso:
    python backtests/strategy_tournament2.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtests.kelly_backtest import _run_kelly_trades, _kelly_metrics
from strategies.kelly_sizer   import KellySizer

from strategies.strategy_zoo  import CCIStrategy, EngulfingStrategy
from strategies.strategy_zoo2 import (
    VWAPStrategy, ParabolicSARStrategy, WilliamsRStrategy,
    KeltnerBreakoutStrategy, MACDHistStrategy, BollingerBandStrategy,
    ROCStrategy, MFIStrategy, HeikinAshiStrategy, ZScoreStrategy,
)


# ── 100 configuraciones para ronda 2 ─────────────────────────────────────────

def build_configs_r2() -> list:
    configs = []

    def add(name, s):
        configs.append((name, s))

    # ── VWAP (10) ────────────────────────────────────────────────────────
    for w, bp, os_, ob in [
        (48, 0.001, 45, 55), (96, 0.002, 45, 55), (96, 0.003, 40, 60),
        (48, 0.002, 40, 60), (24, 0.002, 45, 55), (192, 0.002, 45, 55),
        (96, 0.001, 50, 50), (72, 0.002, 45, 55),  (48, 0.003, 45, 55),
        (96, 0.002, 35, 65),
    ]:
        add(f"VWAP(w{w},bp{bp})", VWAPStrategy(vwap_window=w, bounce_pct=bp,
                                                rsi_os=os_, rsi_ob=ob))

    # ── Parabolic SAR (10) ───────────────────────────────────────────────
    for af, af_max, et in [
        (0.02, 0.2, 50), (0.02, 0.2, 100), (0.01, 0.1, 50),
        (0.03, 0.3, 50), (0.02, 0.2, 30),  (0.02, 0.2, 200),
        (0.01, 0.2, 50), (0.02, 0.1, 50),  (0.04, 0.2, 50),
        (0.02, 0.2, 80),
    ]:
        add(f"SAR(af{af},max{af_max},et{et})",
            ParabolicSARStrategy(af_start=af, af_max=af_max, ema_trend=et))

    # ── Williams %R (10) ─────────────────────────────────────────────────
    for p, os_, et in [
        (14, -80, 50), (14, -75, 50), (14, -80, 100), (21, -80, 50),
        (7, -80, 50),  (14, -85, 50), (14, -80, 30),  (28, -80, 50),
        (14, -70, 50), (10, -80, 50),
    ]:
        add(f"WR(p{p},os{os_},et{et})",
            WilliamsRStrategy(period=p, oversold=os_, ema_trend=et))

    # ── Keltner Breakout (10) ────────────────────────────────────────────
    for ema, mult, cb in [
        (20, 1.5, 1), (20, 2.0, 1), (20, 2.5, 1), (20, 2.0, 2),
        (10, 1.5, 1), (30, 2.0, 1), (20, 1.0, 1), (50, 2.0, 1),
        (20, 2.0, 3), (15, 1.5, 1),
    ]:
        add(f"Keltner(ema{ema},m{mult},cb{cb})",
            KeltnerBreakoutStrategy(ema_period=ema, mult=mult, confirm_bars=cb))

    # ── MACD Histogram Slope (10) ────────────────────────────────────────
    for fast, slow, sig, sb, et in [
        (12, 26, 9, 2, 50), (12, 26, 9, 3, 50), (12, 26, 9, 2, 100),
        (5, 13, 3, 2, 50),  (8, 21, 5, 2, 50),  (12, 26, 9, 4, 50),
        (5, 13, 3, 3, 50),  (12, 26, 9, 2, 30),  (8, 21, 5, 3, 50),
        (12, 26, 9, 2, 200),
    ]:
        add(f"MACDHist(f{fast},s{slow},sb{sb})",
            MACDHistStrategy(fast=fast, slow=slow, signal=sig,
                             slope_bars=sb, ema_trend=et))

    # ── Bollinger Band Touch (10) ────────────────────────────────────────
    for p, std, os_, ob in [
        (20, 2.0, 40, 60), (20, 1.5, 40, 60), (20, 2.5, 35, 65),
        (10, 2.0, 40, 60), (30, 2.0, 40, 60), (20, 2.0, 35, 65),
        (20, 2.0, 45, 55), (14, 2.0, 40, 60), (20, 1.8, 40, 60),
        (20, 2.0, 30, 70),
    ]:
        add(f"BB(p{p},std{std},os{os_})",
            BollingerBandStrategy(period=p, std_mult=std, rsi_os=os_, rsi_ob=ob))

    # ── ROC (10) ─────────────────────────────────────────────────────────
    for p, thr, et, sm in [
        (12, 1.5, 50, 3), (12, 1.0, 50, 3), (12, 2.0, 50, 3),
        (6,  1.5, 50, 3), (24, 1.5, 50, 3), (12, 1.5, 100, 3),
        (12, 1.5, 50, 1), (12, 1.5, 30, 3), (9,  1.5, 50, 3),
        (12, 1.5, 50, 5),
    ]:
        add(f"ROC(p{p},thr{thr},sm{sm})",
            ROCStrategy(roc_period=p, threshold=thr, ema_trend=et, smooth=sm))

    # ── MFI (10) ─────────────────────────────────────────────────────────
    for p, os_, ob, et in [
        (14, 25, 75, 50), (14, 20, 80, 50), (14, 30, 70, 50),
        (7,  25, 75, 50), (21, 25, 75, 50), (14, 25, 75, 100),
        (14, 25, 75, 30), (10, 25, 75, 50), (14, 20, 75, 50),
        (14, 25, 80, 50),
    ]:
        add(f"MFI(p{p},os{os_},et{et})",
            MFIStrategy(period=p, oversold=os_, overbought=ob, ema_trend=et))

    # ── Heikin-Ashi (10) ─────────────────────────────────────────────────
    for cb, et, os_, ob in [
        (2, 50, 45, 55), (3, 50, 45, 55), (2, 100, 45, 55),
        (1, 50, 45, 55), (2, 30,  45, 55), (2, 200, 45, 55),
        (2, 50, 40, 60), (3, 100, 45, 55), (2, 50, 35, 65),
        (4, 50, 45, 55),
    ]:
        add(f"HA(cb{cb},et{et},os{os_})",
            HeikinAshiStrategy(confirm_bars=cb, ema_trend=et,
                               rsi_os=os_, rsi_ob=ob))

    # ── Z-Score (10) ─────────────────────────────────────────────────────
    for p, ze, zx, et in [
        (30, 2.0, 0.5, 100), (20, 2.0, 0.5, 100), (30, 1.5, 0.5, 100),
        (30, 2.5, 0.5, 100), (30, 2.0, 1.0, 100), (50, 2.0, 0.5, 100),
        (30, 2.0, 0.5, 50),  (30, 2.0, 0.3, 100), (20, 1.5, 0.5, 100),
        (30, 2.0, 0.5, 200),
    ]:
        add(f"ZScore(p{p},z{ze},zx{zx})",
            ZScoreStrategy(period=p, z_entry=ze, z_exit=zx, ema_trend=et))

    assert len(configs) == 100, f"Expected 100 configs, got {len(configs)}"
    return configs


# ── Backtest ──────────────────────────────────────────────────────────────────

def _bt(strategy, data, max_hold=8, ppy=8760, kelly="half_kelly"):
    sizer = KellySizer(variant=kelly, min_trades=20)
    try:
        signals = strategy.generate_signals_batch(data)
    except Exception:
        signals = ["hold"] * len(data)
    trades, equity = _run_kelly_trades(signals, data, sizer, max_hold=max_hold)
    return _kelly_metrics(trades, equity, periods_per_year=ppy)


def run_tournament_r2(datasets, max_hold=8, ppy=8760, verbose=True):
    configs  = build_configs_r2()
    n_cfg    = len(configs)
    n_pairs  = len(datasets)
    t0       = time.time()

    if verbose:
        print(f"\n{'='*72}")
        print(f"  TORNEO RONDA 2")
        print(f"  {n_cfg} configs × {n_pairs} pares = {n_cfg * n_pairs} backtests")
        print(f"{'='*72}\n")

    records = []
    for idx, (name, strat) in enumerate(configs):
        sharpes, wrs, rets, tms, dds, tots = [], [], [], [], [], []
        for sym, df in datasets.items():
            try:
                m = _bt(strat, df, max_hold=max_hold, ppy=ppy)
                if m["trades"] > 0:
                    sharpes.append(m["sharpe"])
                    wrs.append(m["winrate"] * 100)
                    rets.append((m["equity_curve"][-1] - 1) * 100)
                    tms.append(m["trades"] / (len(df) / (ppy / 12)))
                    dds.append(m["max_drawdown"] * 100)
                    tots.append(m["trades"])
            except Exception:
                pass
        records.append({
            "name":    name,
            "sharpe":  round(np.mean(sharpes) if sharpes else 0, 3),
            "winrate": round(np.mean(wrs)     if wrs     else 0, 2),
            "return":  round(np.mean(rets)    if rets    else 0, 2),
            "t_month": round(np.mean(tms)     if tms     else 0, 1),
            "max_dd":  round(np.mean(dds)     if dds     else 0, 2),
            "n_trades":sum(tots),
            "n_pairs": len(sharpes),
        })
        if verbose and (idx % 10 == 0 or idx == n_cfg - 1):
            best = max(r["sharpe"] for r in records)
            print(f"  [{(idx+1)/n_cfg*100:5.1f}%] t={time.time()-t0:.1f}s  "
                  f"{idx+1}/{n_cfg}  mejor_sharpe={best:.3f}")

    df = pd.DataFrame(records).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df


# ── Visualización ─────────────────────────────────────────────────────────────

def print_podium_r2(df, top_n=15):
    df_f = df[(df["t_month"] >= 3) & (df["n_trades"] >= 50)].reset_index(drop=True)
    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    print(f"\n{'='*92}")
    print(f"  🏆  RONDA 2 — TOP {min(top_n, len(df_f))} (≥3 t/mes, ≥50 trades)")
    print(f"{'='*92}")
    print(f"  {'#':>3}  {'Estrategia':<36} {'Sharpe':>7} {'WR%':>6} "
          f"{'Ret%':>8} {'T/mes':>6} {'MaxDD':>7} {'Trades':>7}")
    print(f"  {'-'*88}")
    for i, (_, row) in enumerate(df_f.head(top_n).iterrows()):
        m = medals.get(i, "  ")
        print(f"  {m}{i+1:>2}  {row['name']:<36} {row['sharpe']:>7.3f} "
              f"{row['winrate']:>5.1f}%  {row['return']:>+7.2f}%  "
              f"{row['t_month']:>5.0f}  {row['max_dd']:>+6.2f}%  {row['n_trades']:>6}")
    print()
    return df_f


def grand_final(r1_csv, df_r2, datasets, ppy=8760, max_hold=8):
    """
    Gran Final: toma los top 5 de cada ronda y los re-backtestea
    para un ranking definitivo.
    """
    from backtests.strategy_tournament import build_configs as build_r1

    r1 = pd.read_csv(r1_csv)
    r1_f = r1[(r1["t_month"] >= 3) & (r1["n_trades"] >= 50)].head(5)
    r2_f = df_r2[(df_r2["t_month"] >= 3) & (df_r2["n_trades"] >= 50)].head(5)

    # Recuperar instancias de las estrategias ganadoras ronda 1
    r1_configs = {name: s for name, s in build_r1()}
    r2_configs = {name: s for name, s in build_configs_r2()}

    finalists = []
    for _, row in r1_f.iterrows():
        if row["name"] in r1_configs:
            finalists.append((f"[R1] {row['name']}", r1_configs[row["name"]]))
    for _, row in r2_f.iterrows():
        if row["name"] in r2_configs:
            finalists.append((f"[R2] {row['name']}", r2_configs[row["name"]]))

    print(f"\n{'='*92}")
    print(f"  🏆🏆  GRAN FINAL — Top 5 Ronda 1 vs Top 5 Ronda 2")
    print(f"{'='*92}")
    print(f"  {'#':>3}  {'Estrategia':<44} {'Sharpe':>7} {'WR%':>6} "
          f"{'Ret%':>8} {'T/mes':>6} {'MaxDD':>7}")
    print(f"  {'-'*88}")

    medals = {0:"🥇",1:"🥈",2:"🥉",3:" 4",4:" 5",5:" 6",6:" 7",7:" 8",8:" 9",9:"10"}
    results = []
    for name, strat in finalists:
        sharpes, wrs, rets, tms, dds = [], [], [], [], []
        for sym, df in datasets.items():
            try:
                m = _bt(strat, df, max_hold=max_hold, ppy=ppy)
                if m["trades"] > 0:
                    sharpes.append(m["sharpe"])
                    wrs.append(m["winrate"]*100)
                    rets.append((m["equity_curve"][-1]-1)*100)
                    tms.append(m["trades"]/(len(df)/(ppy/12)))
                    dds.append(m["max_drawdown"]*100)
            except Exception:
                pass
        results.append({
            "name": name,
            "sharpe": np.mean(sharpes) if sharpes else 0,
            "wr": np.mean(wrs) if wrs else 0,
            "ret": np.mean(rets) if rets else 0,
            "tm": np.mean(tms) if tms else 0,
            "dd": np.mean(dds) if dds else 0,
        })

    results.sort(key=lambda x: -x["sharpe"])
    for i, r in enumerate(results):
        m = medals.get(i, "  ")
        print(f"  {m}  {r['name']:<44} {r['sharpe']:>7.3f} "
              f"{r['wr']:>5.1f}%  {r['ret']:>+7.2f}%  "
              f"{r['tm']:>5.0f}  {r['dd']:>+6.2f}%")
    print()

    top5 = results[:5]
    print("  🎯 TOP 5 DEFINITIVO:")
    for i, r in enumerate(top5):
        print(f"    {i+1}. {r['name']}")

    return results


def save_plots_r2(df, path="data/plots/27_tournament_r2.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_f = df[(df["t_month"] >= 3) & (df["n_trades"] >= 50)].head(20)
    if df_f.empty:
        df_f = df.head(20)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Tournament Round 2 — Top 20", fontsize=13, fontweight="bold")
    for ax, col, color, title in [
        (axes[0], "sharpe",  "#3fb950", "Sharpe"),
        (axes[1], "winrate", "#58a6ff", "Win Rate %"),
        (axes[2], "return",  "#d29922", "Retorno %"),
    ]:
        vals  = df_f[col].values[::-1]
        names = df_f["name"].values[::-1]
        colors = [color] * len(vals) if col != "return" else \
                 ["#3fb950" if v >= 0 else "#f85149" for v in vals]
        ax.barh(names, vals, color=colors)
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        if col == "winrate":
            ax.axvline(50, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.kucoin_client import KuCoinClient

    INTERVAL = "1hour"
    PPY      = 8_760
    PAIRS    = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "LINK-USDT"]

    print("Descargando datos reales KuCoin 1h...")
    client   = KuCoinClient(verbose=False)
    datasets = {}
    for sym in PAIRS:
        df = client.get_ohlcv(sym, interval=INTERVAL, limit=1500)
        if df is not None and len(df) > 100:
            datasets[sym] = df
            print(f"  {sym}: {len(df)} velas  ({df.index[0].date()} → {df.index[-1].date()})")

    # Ronda 2
    results_r2 = run_tournament_r2(datasets, max_hold=8, ppy=PPY, verbose=True)
    df_f = print_podium_r2(results_r2, top_n=15)
    results_r2.to_csv("data/tournament_r2_results.csv", index=False)
    print(f"  CSV: data/tournament_r2_results.csv")
    save_plots_r2(results_r2)

    # Gran Final: R1 vs R2
    grand_final("data/tournament_results.csv", results_r2, datasets,
                ppy=PPY, max_hold=8)
