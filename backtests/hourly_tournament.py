"""
hourly_tournament.py

Torneo a 1H — las mejores estrategias de los torneos anteriores
re-testeadas en timeframe horario con full_kelly.

Objetivo: más trades (10-40/mes) manteniendo rentabilidad neta > fees.

  Timeframe : 1H (8 760 velas/año)
  Kelly     : full_kelly (máximo rendimiento)
  max_hold  : 24 velas = 1 día máximo por trade
  Fees      : 0.2% round-trip (KuCoin taker)
  Datos     : 1 año paginado (6 requests × 1500 velas)

Uso:
    python backtests/hourly_tournament.py
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

from strategies.smc_strategy    import SMCStrategy
from strategies.strategy_zoo2   import (
    ParabolicSARStrategy, WilliamsRStrategy, HeikinAshiStrategy,
    MFIStrategy, KeltnerBreakoutStrategy, VWAPStrategy, ZScoreStrategy,
    BollingerBandStrategy, ROCStrategy, MACDHistStrategy,
)
from strategies.strategy_zoo    import (
    CCIStrategy, EngulfingStrategy, DonchianBreakoutStrategy,
    ADXDIStrategy, SupertrendStrategy, TripleEMAStrategy, EMACrossStrategy,
    StochasticStrategy,
)
from strategies.swing_strategies import (
    ChandelierExitStrategy, TurtleBreakoutStrategy, MultiEMASwingStrategy,
)


# ── Configuración ─────────────────────────────────────────────────────────────

PAIRS    = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BNB-USDT"]
MAX_HOLD = 24      # 24h máx por trade
PPY      = 8760    # horas por año
FEE_RT   = 0.002   # 0.2% round-trip
SIZER    = KellySizer(variant="full_kelly", min_trades=20)


# ── 100 configuraciones para 1H ───────────────────────────────────────────────

def build_configs():
    configs = []

    def add(n, s):
        configs.append((n, s))

    # ── Williams %R (10) — ganador consistente ───────────────────────────
    for p, os_, et in [
        (7,  -80, 50), (7,  -80, 20), (7,  -75, 20),
        (14, -80, 50), (14, -80, 20), (14, -75, 20),
        (14, -85, 50), (21, -80, 50), (21, -80, 20),
        (10, -80, 20),
    ]:
        add(f"WR(p{p},os{os_},et{et})",
            WilliamsRStrategy(period=p, oversold=os_, ema_trend=et))

    # ── Parabolic SAR (10) ───────────────────────────────────────────────
    for af, afmax, et in [
        (0.02, 0.2, 20), (0.02, 0.2, 50),  (0.02, 0.2, 100),
        (0.01, 0.1, 20), (0.01, 0.1, 50),  (0.01, 0.2, 20),
        (0.03, 0.2, 20), (0.02, 0.3, 20),  (0.02, 0.2, 14),
        (0.01, 0.15, 50),
    ]:
        add(f"SAR(af{af},m{afmax},et{et})",
            ParabolicSARStrategy(af_start=af, af_max=afmax, ema_trend=et))

    # ── HeikinAshi (10) ──────────────────────────────────────────────────
    for cb, et, ros, rob in [
        (2, 20, 45, 55), (2, 50, 45, 55), (2, 20, 40, 60),
        (3, 20, 45, 55), (3, 50, 45, 55), (3, 20, 40, 60),
        (2, 14, 45, 55), (2, 20, 35, 65), (3, 14, 45, 55),
        (4, 20, 45, 55),
    ]:
        add(f"HA(cb{cb},et{et},os{ros})",
            HeikinAshiStrategy(confirm_bars=cb, ema_trend=et,
                               rsi_os=ros, rsi_ob=rob))

    # ── Turtle Breakout (10) ─────────────────────────────────────────────
    for en, ex, et in [
        (20, 10, 50),  (20, 10, 100), (20,  5, 50),
        (10,  5, 50),  (10,  5, 20),  (15,  7, 50),
        (30, 15, 50),  (15,  7, 20),  (20, 10, 20),
        (10,  5, 100),
    ]:
        add(f"Turtle(en{en},ex{ex},et{et})",
            TurtleBreakoutStrategy(entry_period=en, exit_period=ex,
                                   ema_trend=et))

    # ── MultiEMA + ADX (8) ───────────────────────────────────────────────
    for e1, e2, e3, adx_t in [
        (8, 21, 50, 20), (8, 21, 50, 25), (13, 34, 89, 20),
        (13, 34, 89, 25),(21, 55, 100, 20),(21, 55, 100, 25),
        (8, 21, 50, 15), (13, 34, 89, 15),
    ]:
        add(f"MultiEMA({e1}/{e2}/{e3},adx>{adx_t})",
            MultiEMASwingStrategy(e1=e1, e2=e2, e3=e3, adx_thresh=adx_t))

    # ── CCI (8) ──────────────────────────────────────────────────────────
    for p, t, et in [
        (14, 100, 20), (14, 100, 50), (20, 100, 20),
        (20, 100, 50), (10, 100, 20), (14, 150, 20),
        (20, 150, 50), (14,  80, 20),
    ]:
        add(f"CCI(p{p},t{t},et{et})",
            CCIStrategy(period=p, threshold=t, ema_trend=et))

    # ── Chandelier Exit (8) ──────────────────────────────────────────────
    for p, m, et in [
        (14, 2.0, 50),  (14, 2.5, 50),  (14, 3.0, 50),
        (22, 2.0, 50),  (22, 3.0, 50),  (14, 2.0, 20),
        (14, 3.0, 20),  (22, 2.0, 20),
    ]:
        add(f"Chandelier(p{p},m{m},et{et})",
            ChandelierExitStrategy(period=p, mult=m, ema_trend=et))

    # ── SMC (6) ──────────────────────────────────────────────────────────
    for sw in [3, 5, 7, 10, 12, 15]:
        add(f"SMC(sw{sw})", SMCStrategy(swing_window=sw))

    # ── Supertrend (6) ───────────────────────────────────────────────────
    for p, m in [
        (7, 2.0), (7, 3.0), (10, 2.0),
        (10, 3.0),(14, 2.0), (14, 3.0),
    ]:
        add(f"Supertrend(p{p},m{m})",
            SupertrendStrategy(atr_period=p, multiplier=m))

    # ── MFI (6) ──────────────────────────────────────────────────────────
    for p, os_, et in [
        (14, 20, 50), (14, 25, 50), (14, 20, 20),
        (14, 25, 20), (21, 25, 50), (7,  20, 20),
    ]:
        add(f"MFI(p{p},os{os_},et{et})",
            MFIStrategy(period=p, oversold=os_, ema_trend=et))

    # ── EMA Cross (6) ────────────────────────────────────────────────────
    for fast, slow, trend in [
        (5, 13, 50), (5, 21, 50), (8, 21, 50),
        (5, 13, 20), (8, 21, 20), (9, 21, 50),
    ]:
        add(f"EMA({fast},{slow},{trend})",
            EMACrossStrategy(fast=fast, slow=slow, trend=trend))

    # ── Stochastic (6) ───────────────────────────────────────────────────
    for k, d, os_, ob in [
        (14, 3, 20, 80), (9, 3, 20, 80),  (5, 3, 20, 80),
        (14, 3, 25, 75), (14, 5, 20, 80), (9, 3, 25, 75),
    ]:
        add(f"Stoch(k{k},d{d},os{os_})",
            StochasticStrategy(k_period=k, d_period=d,
                               oversold=os_, overbought=ob))

    # ── ROC (6) ──────────────────────────────────────────────────────────
    for p, t, et in [
        (10, 0.5, 50), (14, 0.5, 50), (10, 1.0, 50),
        (10, 0.5, 20), (14, 1.0, 50), (20, 1.0, 50),
    ]:
        add(f"ROC(p{p},t{t},et{et})",
            ROCStrategy(roc_period=p, threshold=t, ema_trend=et))

    print(f"Configs generadas: {len(configs)}")
    return configs


# ── Motor de backtest ─────────────────────────────────────────────────────────

def _run_one(name, strat, datasets):
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
        except Exception:
            pass

    if not all_m:
        return None

    def avg(k):
        return float(np.mean([m[k] for m in all_m]))

    total_trades = sum(m["trades"] for m in all_m)
    n_pairs      = len(all_m)
    avg_frac     = avg("avg_fraction")

    gross_returns = [(m["equity_curve"][-1] - 1.0) * 100
                     for m in all_m if m["equity_curve"]]
    gross_pct     = float(np.mean(gross_returns)) if gross_returns else 0.0

    trades_pp    = total_trades / n_pairs if n_pairs else 0
    fee_cost_pct = trades_pp * FEE_RT * avg_frac * 100
    net_pct      = gross_pct - fee_cost_pct

    gross_sharpe = avg("sharpe")
    net_sharpe   = gross_sharpe * (net_pct / gross_pct) if abs(gross_pct) > 1e-6 else 0.0

    # trades/mes — 1H: ~720 velas/mes
    candles_total = sum(len(df) for df in datasets.values())
    months        = candles_total / (720 * n_pairs) if n_pairs else 1
    t_month       = trades_pp / months

    # equity media
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
        "avg_frac_pct": round(avg_frac * 100, 2),
        "eq_curve":     eq_mean,
    }


# ── Presentación ──────────────────────────────────────────────────────────────

def print_podium(rows):
    profitable = sorted([r for r in rows if r and r["net_pct"] > 0],
                        key=lambda x: -x["net_sharpe"])
    losing     = sorted([r for r in rows if r and r["net_pct"] <= 0],
                        key=lambda x: -x["net_pct"])

    print(f"\n{'='*100}")
    print(f"  TORNEO 1H — FULL KELLY — 1 AÑO")
    print(f"  Rentables post-fees: {len(profitable)}/{len(rows)}")
    print(f"{'='*100}")
    print(f"  {'#':>3}  {'Estrategia':<38}  "
          f"{'NetSh':>7}  {'Net%/yr':>8}  {'Gross%':>7}  "
          f"{'Fees%':>6}  {'WR%':>5}  {'PF':>5}  "
          f"{'t/mes':>5}  {'MaxDD%':>7}  {'Calmar':>6}  {'Frac%':>5}")
    print("  " + "-"*98)

    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    for i, r in enumerate(profitable[:20]):
        m = medals.get(i, "  ")
        print(
            f"  {i+1:>3}. {m} {r['name']:<36}  "
            f"{r['net_sharpe']:>7.3f}  "
            f"{r['net_pct']:>+8.2f}%  "
            f"{r['gross_pct']:>+7.2f}%  "
            f"{r['fee_cost_pct']:>6.2f}%  "
            f"{r['winrate']:>5.1f}%  "
            f"{r['profit_factor']:>5.2f}  "
            f"{r['t_month']:>5.1f}  "
            f"{r['max_dd']:>7.1f}%  "
            f"{r['calmar']:>6.2f}  "
            f"{r['avg_frac_pct']:>5.1f}%"
        )

    if losing:
        print(f"\n  ── No rentables ({len(losing)}) ── (top 5 más cercanos)")
        for r in losing[:5]:
            print(f"  ✗  {r['name']:<38}  net {r['net_pct']:>+7.2f}%  "
                  f"(gross {r['gross_pct']:>+6.2f}% - fees {r['fee_cost_pct']:.2f}%)"
                  f"  {r['t_month']:.1f} t/mes")

    print(f"\n{'='*100}")
    if profitable:
        best_sh  = profitable[0]
        best_ret = max(profitable, key=lambda x: x["net_pct"])
        print(f"  ★ MEJOR SHARPE NET  : {best_sh['name']}  →  {best_sh['net_sharpe']:.3f}  "
              f"({best_sh['net_pct']:+.2f}%/año,  {best_sh['t_month']:.1f} t/mes)")
        print(f"  ★ MEJOR RETORNO NET : {best_ret['name']}  →  {best_ret['net_pct']:+.2f}%/año  "
              f"(Sharpe {best_ret['net_sharpe']:.3f},  {best_ret['t_month']:.1f} t/mes)")
    print(f"{'='*100}\n")


def save_plots(rows):
    profitable = sorted([r for r in rows if r and r["net_pct"] > 0],
                        key=lambda x: -x["net_sharpe"])[:15]
    if not profitable:
        return

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    gs  = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)

    def _ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        return ax

    # 1. Equity curves
    ax1 = _ax(gs[0, :])
    palette = plt.cm.plasma(np.linspace(0.1, 0.95, len(profitable)))
    for r, col in zip(profitable, palette):
        eq = np.array(r["eq_curve"])
        x  = np.linspace(0, 12, len(eq))   # meses
        ax1.plot(x, eq, label=r["name"][:28], linewidth=1.4, color=col, alpha=0.9)
    ax1.axhline(1.0, color="#30363d", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Meses", color="#8b949e")
    ax1.set_ylabel("Capital (1.0 = inicio)", color="#8b949e")
    ax1.set_title("Equity Curves — Top 15  |  1H Full Kelly  |  1 Año", color="#e6edf3", fontsize=11)
    ax1.legend(fontsize=6.5, framealpha=0.15, labelcolor="#c9d1d9", loc="upper left", ncol=3)

    # 2. Net% bar chart (horizontal)
    ax2 = _ax(gs[1, 0])
    names  = [r["name"][:24] for r in profitable]
    values = [r["net_pct"] for r in profitable]
    colors = ["#3fb950" if v > 0 else "#f85149" for v in values]
    bars   = ax2.barh(names, values, color=colors, alpha=0.85, edgecolor="#30363d", linewidth=0.5)
    ax2.axvline(0, color="#8b949e", linewidth=0.8)
    ax2.set_xlabel("Net Return % / año", color="#8b949e")
    ax2.set_title("Retorno Neto Anual (post-fees)", color="#e6edf3")
    ax2.tick_params(axis="y", labelsize=7.5, colors="#c9d1d9")
    for bar, v in zip(bars, values):
        ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                 f"{v:+.1f}%", va="center", fontsize=7, color="#c9d1d9")

    # 3. t/mes vs Net% — burbuja = Sharpe
    ax3 = _ax(gs[1, 1])
    all_p = sorted([r for r in rows if r and r["net_pct"] > 0],
                   key=lambda x: -x["net_sharpe"])
    sizes  = [max(30, r["net_sharpe"] * 200) for r in all_p]
    sc = ax3.scatter([r["t_month"] for r in all_p],
                     [r["net_pct"]  for r in all_p],
                     s=sizes, c=[r["net_sharpe"] for r in all_p],
                     cmap="plasma", alpha=0.85,
                     edgecolors="#30363d", linewidths=0.5)
    for r in all_p[:12]:
        ax3.annotate(r["name"][:20], (r["t_month"], r["net_pct"]),
                     fontsize=6, color="#8b949e",
                     xytext=(4, 4), textcoords="offset points")
    ax3.axhline(0, color="#30363d", linestyle="--", linewidth=0.8)
    ax3.set_xlabel("Trades / mes", color="#8b949e")
    ax3.set_ylabel("Net Return % / año", color="#8b949e")
    ax3.set_title("Frecuencia vs Rentabilidad Neta", color="#e6edf3")
    cb = fig.colorbar(sc, ax=ax3, pad=0.02)
    cb.set_label("Net Sharpe", color="#8b949e", fontsize=8)
    cb.ax.tick_params(colors="#8b949e", labelsize=7)

    fig.suptitle(
        "Torneo 1H — Full Kelly — 1 Año  |  KuCoin 5 Pares  |  Fee 0.2% round-trip",
        color="#e6edf3", fontsize=13, y=0.99
    )
    out = "data/hourly_tournament.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Plot: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from data.kucoin_client import KuCoinClient
    client = KuCoinClient()

    print("\n📡 Descargando 1 año de datos 1H (paginando ~6 requests/par)...")
    datasets = {}
    for pair in PAIRS:
        try:
            df = client.get_ohlcv_paginated(pair, interval="1hour", days=365)
            if df is not None and len(df) > 500:
                datasets[pair] = df
                print(f"  ✅ {pair}: {len(df)} velas  "
                      f"[{df.index[0].strftime('%Y-%m-%d')} → "
                      f"{df.index[-1].strftime('%Y-%m-%d')}]")
            else:
                print(f"  ⚠ {pair}: datos insuficientes")
        except Exception as e:
            print(f"  ❌ {pair}: {e}")
        time.sleep(0.3)

    if not datasets:
        print("❌ Sin datos.")
        return

    configs = build_configs()
    print(f"\n  Configs: {len(configs)}  |  Pares: {len(datasets)}  |  Kelly: full_kelly\n")

    rows = []
    t_total = time.time()
    for i, (name, strat) in enumerate(configs, 1):
        row = _run_one(name, strat, datasets)
        if row:
            sign = "✅" if row["net_pct"] > 0 else "❌"
            rows.append(row)
            if i % 10 == 0 or row["net_pct"] > 5:
                print(f"  [{i:3d}/{len(configs)}] {sign} {name:<38}  "
                      f"net {row['net_pct']:>+7.2f}%  "
                      f"sh {row['net_sharpe']:>6.3f}  "
                      f"{row['t_month']:>5.1f} t/mes")

    elapsed = time.time() - t_total
    print(f"\n  Completado en {elapsed:.1f}s")

    # Guardar CSV
    csv_rows = [{k: v for k, v in r.items() if k != "eq_curve"} for r in rows]
    df_out   = pd.DataFrame(csv_rows).sort_values("net_sharpe", ascending=False)
    csv_out  = "data/hourly_tournament_results.csv"
    df_out.to_csv(csv_out, index=False)
    print(f"  💾 CSV: {csv_out}")

    print_podium(rows)
    save_plots(rows)

    return rows


if __name__ == "__main__":
    main()
