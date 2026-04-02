"""
scalp_tournament.py

Torneo de scalping: 100 configuraciones en datos 15min de KuCoin.
Objetivo: maximizar trades/mes manteniendo Sharpe > 0.

Uso:
    python backtests/scalp_tournament.py
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

from strategies.scalp_strategies import (
    EMAScalpStrategy, RSIScalpStrategy, StochScalpStrategy,
    BBScalpStrategy, CCIScalpStrategy, MACDZeroStrategy,
    VolumeBreakStrategy, PriceActionScalpStrategy,
    MomentumScalpStrategy, DualThrustStrategy,
)


# ── 100 configuraciones de scalping ──────────────────────────────────────────

def build_scalp_configs() -> list:
    configs = []
    def add(n, s): configs.append((n, s))

    # ── EMA Scalp (10) ────────────────────────────────────────────────────
    for f, s, t, rt in [
        (3, 8, 21, True), (3, 8, 13, True), (2, 5, 13, True),
        (3, 8, 21, False), (5, 13, 34, True), (2, 5, 21, True),
        (3, 8, 8, False), (4, 9, 21, True), (3, 10, 21, True),
        (2, 7, 21, True),
    ]:
        add(f"EMAScalp(f{f},s{s},t{t})", EMAScalpStrategy(fast=f, slow=s, trend=t, require_trend=rt))

    # ── RSI Scalp (10) ────────────────────────────────────────────────────
    for p, os_, ob, ef, es in [
        (7, 42, 58, 8, 21), (7, 40, 60, 8, 21), (5, 42, 58, 5, 13),
        (7, 45, 55, 8, 21), (9, 42, 58, 8, 21), (7, 42, 58, 5, 13),
        (7, 38, 62, 8, 21), (14,42, 58, 8, 21), (7, 42, 58, 3, 8),
        (7, 42, 58, 8, 34),
    ]:
        add(f"RSIScalp(p{p},os{os_})", RSIScalpStrategy(rsi_period=p, os=os_, ob=ob, ema_fast=ef, ema_slow=es))

    # ── Stochastic Scalp (10) ────────────────────────────────────────────
    for k, d, os_, ob, et in [
        (5, 3, 15, 85, 21), (5, 3, 20, 80, 21), (3, 3, 15, 85, 21),
        (5, 3, 10, 90, 21), (5, 3, 15, 85, 13), (7, 3, 15, 85, 21),
        (5, 3, 15, 85, 34), (5, 5, 15, 85, 21), (9, 3, 15, 85, 21),
        (5, 3, 15, 85, 8),
    ]:
        add(f"StochScalp(k{k},d{d},os{os_})", StochScalpStrategy(k=k, d=d, os=os_, ob=ob, ema_trend=et))

    # ── BB Scalp (10) ────────────────────────────────────────────────────
    for p, std, rp, os_, ob in [
        (10, 1.5, 7, 35, 65), (10, 2.0, 7, 35, 65), (10, 1.5, 5, 35, 65),
        (8,  1.5, 7, 35, 65), (14, 1.5, 7, 35, 65), (10, 1.5, 7, 40, 60),
        (10, 1.5, 7, 30, 70), (10, 1.0, 7, 35, 65), (20, 1.5, 7, 35, 65),
        (10, 1.5, 9, 35, 65),
    ]:
        add(f"BBScalp(p{p},std{std},rsi{rp})", BBScalpStrategy(period=p, std=std, rsi_period=rp, os=os_, ob=ob))

    # ── CCI Scalp (10) ───────────────────────────────────────────────────
    for p, thr, et in [
        (10, 50, 21), (10, 75, 21), (7, 50, 21), (14, 50, 21),
        (10, 50, 13), (10, 50, 34), (10, 50, 8),  (10, 50, 55),
        (10, 40, 21), (10, 60, 21),
    ]:
        add(f"CCIScalp(p{p},thr{thr},et{et})", CCIScalpStrategy(period=p, thr=thr, ema_trend=et))

    # ── MACD Zero Cross (10) ─────────────────────────────────────────────
    for f, s, sig, et in [
        (5, 13, 3, 21), (3, 8, 3, 21), (5, 13, 5, 21), (5, 13, 3, 13),
        (5, 13, 3, 34), (8, 21, 5, 21), (3, 8, 3, 13), (5, 13, 3, 8),
        (5, 13, 3, 55), (5, 13, 3, 5),
    ]:
        add(f"MACDZero(f{f},s{s},t{et})", MACDZeroStrategy(fast=f, slow=s, signal=sig, ema_trend=et))

    # ── Volume Break (10) ────────────────────────────────────────────────
    for vm, bm, et in [
        (1.8, 0.6, 21), (1.5, 0.5, 21), (2.0, 0.6, 21), (1.8, 0.8, 21),
        (1.8, 0.6, 13), (1.8, 0.6, 34), (1.5, 0.4, 21), (2.5, 0.6, 21),
        (1.8, 0.6, 8),  (1.2, 0.5, 21),
    ]:
        add(f"VolBreak(vm{vm},bm{bm},et{et})", VolumeBreakStrategy(vol_mult=vm, body_mult=bm, ema_trend=et))

    # ── Price Action Scalp (10) ──────────────────────────────────────────
    for et, mr in [
        (21, 0.3), (21, 0.2), (21, 0.4), (13, 0.3), (34, 0.3),
        (21, 0.1), (8, 0.3),  (55, 0.3), (21, 0.5), (21, 0.25),
    ]:
        add(f"PAScalp(et{et},mr{mr})", PriceActionScalpStrategy(ema_trend=et, min_range_atr=mr))

    # ── Momentum Scalp (10) ──────────────────────────────────────────────
    for streak, vm, et, mb in [
        (3, 1.2, 21, 0.2), (2, 1.2, 21, 0.2), (4, 1.2, 21, 0.2),
        (3, 1.5, 21, 0.2), (3, 1.0, 21, 0.2), (3, 1.2, 13, 0.2),
        (3, 1.2, 34, 0.2), (3, 1.2, 21, 0.1), (3, 1.2, 21, 0.3),
        (2, 1.0, 13, 0.1),
    ]:
        add(f"MomScalp(s{streak},vm{vm},et{et})",
            MomentumScalpStrategy(streak=streak, vol_mult=vm, ema_trend=et, min_body_atr=mb))

    # ── Dual Thrust (10) ─────────────────────────────────────────────────
    for n, k, et in [
        (20, 0.5, 21), (20, 0.4, 21), (20, 0.6, 21), (10, 0.5, 21),
        (30, 0.5, 21), (20, 0.5, 13), (20, 0.5, 34), (20, 0.3, 21),
        (15, 0.5, 21), (20, 0.5, 8),
    ]:
        add(f"DualThrust(n{n},k{k},et{et})", DualThrustStrategy(n=n, k=k, ema_trend=et))

    assert len(configs) == 100, f"Expected 100, got {len(configs)}"
    return configs


# ── Backtest ──────────────────────────────────────────────────────────────────

def _bt(strategy, data, max_hold=4, ppy=35_040, kelly="quarter_kelly"):
    """15min: max_hold=4 velas = 1h máximo por trade. Kelly quarter para riesgo."""
    sizer = KellySizer(variant=kelly, min_trades=20)
    try:
        signals = strategy.generate_signals_batch(data)
    except Exception:
        signals = ["hold"] * len(data)
    trades, equity = _run_kelly_trades(signals, data, sizer, max_hold=max_hold)
    return _kelly_metrics(trades, equity, periods_per_year=ppy)


def run_scalp_tournament(datasets, max_hold=4, ppy=35_040,
                          kelly="quarter_kelly", verbose=True):
    configs  = build_scalp_configs()
    n_cfg    = len(configs)
    n_pairs  = len(datasets)
    t0       = time.time()

    if verbose:
        print(f"\n{'='*72}")
        print(f"  TORNEO SCALPING — 15min")
        print(f"  {n_cfg} configs × {n_pairs} pares = {n_cfg * n_pairs} backtests")
        print(f"  max_hold={max_hold} velas ({max_hold*15}min max/trade)  kelly={kelly}")
        print(f"{'='*72}\n")

    records = []
    for idx, (name, strat) in enumerate(configs):
        sharpes, wrs, rets, tms, dds, tots = [], [], [], [], [], []
        for sym, df in datasets.items():
            try:
                m = _bt(strat, df, max_hold=max_hold, ppy=ppy, kelly=kelly)
                if m["trades"] > 0:
                    sharpes.append(m["sharpe"])
                    wrs.append(m["winrate"] * 100)
                    rets.append((m["equity_curve"][-1] - 1) * 100)
                    candles_per_month = ppy / 12   # 35040/12 = 2920
                    tms.append(m["trades"] / (len(df) / candles_per_month))
                    dds.append(m["max_drawdown"] * 100)
                    tots.append(m["trades"])
            except Exception:
                pass
        records.append({
            "name":    name,
            "sharpe":  round(np.mean(sharpes) if sharpes else 0, 3),
            "winrate": round(np.mean(wrs)     if wrs     else 0, 2),
            "return":  round(np.mean(rets)    if rets    else 0, 3),
            "t_month": round(np.mean(tms)     if tms     else 0, 0),
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


def print_scalp_podium(df, top_n=20, min_trades_month=50):
    df_f = df[df["t_month"] >= min_trades_month].reset_index(drop=True)
    medals = {0:"🥇", 1:"🥈", 2:"🥉"}
    print(f"\n{'='*96}")
    print(f"  ⚡  SCALPING — TOP {min(top_n, len(df_f))} (≥{min_trades_month} trades/mes)")
    print(f"{'='*96}")
    print(f"  {'#':>3}  {'Estrategia':<38} {'Sharpe':>7} {'WR%':>6} "
          f"{'Ret%':>8} {'T/mes':>7} {'MaxDD':>7} {'Trades':>7}")
    print(f"  {'-'*92}")
    for i, (_, row) in enumerate(df_f.head(top_n).iterrows()):
        m = medals.get(i, "  ")
        print(f"  {m}{i+1:>2}  {row['name']:<38} {row['sharpe']:>7.3f} "
              f"{row['winrate']:>5.1f}%  {row['return']:>+7.3f}%  "
              f"{row['t_month']:>6.0f}  {row['max_dd']:>+6.2f}%  {row['n_trades']:>6}")
    print()

    # Ranking por familia
    df_f2 = df[df["n_trades"] >= 30].copy()
    df_f2["family"] = df_f2["name"].str.extract(r"^([A-Za-z]+)")
    fam = df_f2.groupby("family")[["sharpe","winrate","return","t_month"]].mean()
    fam = fam.sort_values("sharpe", ascending=False)
    print(f"  RANKING POR FAMILIA (todos con ≥30 trades):")
    print(f"  {'Familia':<20} {'Sharpe':>7} {'WR%':>6} {'Ret%':>8} {'T/mes':>7}")
    print(f"  {'-'*52}")
    for nm, row in fam.iterrows():
        print(f"  {nm:<20} {row['sharpe']:>7.3f} {row['winrate']:>5.1f}%  "
              f"{row['return']:>+7.3f}%  {row['t_month']:>6.0f}")
    print()
    return df_f


def save_scalp_plots(df, path="data/plots/28_scalp_tournament.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_f = df[df["t_month"] >= 50].head(20)
    if df_f.empty:
        df_f = df.head(20)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Scalping Tournament — Top 20 (≥50 trades/mes)", fontsize=13, fontweight="bold")

    # Sharpe
    ax = axes[0, 0]
    vals = df_f["sharpe"].values[::-1]
    names = df_f["name"].values[::-1]
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in vals]
    ax.barh(names, vals, color=colors); ax.set_title("Sharpe Ratio"); ax.tick_params(axis="y", labelsize=6)
    ax.axvline(0, color="white", alpha=0.3)

    # WinRate
    ax = axes[0, 1]
    ax.barh(names, df_f["winrate"].values[::-1], color="#58a6ff")
    ax.axvline(50, color="red", ls="--", alpha=0.5); ax.set_title("Win Rate %"); ax.tick_params(axis="y", labelsize=6)

    # Trades/mes
    ax = axes[1, 0]
    ax.barh(names, df_f["t_month"].values[::-1], color="#d29922"); ax.set_title("Trades / mes"); ax.tick_params(axis="y", labelsize=6)

    # Return
    ax = axes[1, 1]
    vals_r = df_f["return"].values[::-1]
    colors_r = ["#3fb950" if v >= 0 else "#f85149" for v in vals_r]
    ax.barh(names, vals_r, color=colors_r); ax.set_title("Retorno %"); ax.tick_params(axis="y", labelsize=6)

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.kucoin_client import KuCoinClient

    INTERVAL = "15min"
    PPY      = 35_040   # 15min candles per year
    MAX_HOLD = 4        # 4 × 15min = 1h máximo
    KELLY    = "quarter_kelly"
    PAIRS    = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "LINK-USDT"]

    print("Descargando datos reales KuCoin 15min...")
    client   = KuCoinClient(verbose=False)
    datasets = {}
    for sym in PAIRS:
        df = client.get_ohlcv(sym, interval=INTERVAL, limit=1500)
        if df is not None and len(df) > 100:
            datasets[sym] = df
            days = len(df) / (4 * 24)
            print(f"  {sym}: {len(df)} velas  ~{days:.0f} días  "
                  f"({df.index[0].date()} → {df.index[-1].date()})")

    results = run_scalp_tournament(datasets, max_hold=MAX_HOLD, ppy=PPY,
                                    kelly=KELLY, verbose=True)
    df_filtered = print_scalp_podium(results, top_n=20, min_trades_month=50)

    results.to_csv("data/scalp_tournament_results.csv", index=False)
    print(f"  CSV: data/scalp_tournament_results.csv")
    save_scalp_plots(results)

    # Top 5 definitivo
    top5 = results[results["t_month"] >= 50].head(5)
    print("  ⚡ TOP 5 SCALPING DEFINITIVO:")
    for i, (_, row) in enumerate(top5.iterrows()):
        print(f"    {i+1}. {row['name']:<40} Sharpe={row['sharpe']:.3f}  "
              f"WR={row['winrate']:.1f}%  T/mes={row['t_month']:.0f}  Ret={row['return']:+.3f}%")
