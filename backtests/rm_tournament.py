"""
rm_tournament.py

100 configuraciones con SL/TP automático (RiskManager ATR-based).
El RiskManager fuerza un ratio Riesgo/Beneficio mínimo en cada trade,
elevando el profit factor y con ello el Kelly fraction → más retorno.

Estrategias × 5 combinaciones de RiskManager = 100 configs
  RM params: atr_mult ∈ {1.0, 1.5, 2.0} × rr ∈ {2.0, 3.0}

Timeframe : 1H  (8 760 velas/año)
Kelly     : full_kelly, max_fraction=0.40
max_hold  : 24h (el RM cierra antes via SL/TP)
Fees      : 0.2% round-trip

Uso:
    python backtests/rm_tournament.py
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
from strategies.risk_manager   import RiskManager

from strategies.smc_strategy    import SMCStrategy
from strategies.strategy_zoo2   import (
    ParabolicSARStrategy, WilliamsRStrategy, HeikinAshiStrategy,
)
from strategies.strategy_zoo    import CCIStrategy
from strategies.swing_strategies import (
    ChandelierExitStrategy, TurtleBreakoutStrategy, MultiEMASwingStrategy,
)

# ── Configuración ──────────────────────────────────────────────────────────────

PAIRS    = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BNB-USDT"]
MAX_HOLD = 24
PPY      = 8760
FEE_RT   = 0.002
SIZER    = KellySizer(variant="full_kelly", min_trades=20,
                      max_fraction=0.40, min_fraction=0.01)

# 5 perfiles de RiskManager
RM_PROFILES = [
    ("RM1x2",  RiskManager(method="atr", atr_multiplier=1.0, rr_ratio=2.0)),
    ("RM1x3",  RiskManager(method="atr", atr_multiplier=1.0, rr_ratio=3.0)),
    ("RM15x2", RiskManager(method="atr", atr_multiplier=1.5, rr_ratio=2.0)),
    ("RM15x3", RiskManager(method="atr", atr_multiplier=1.5, rr_ratio=3.0)),
    ("RM2x3",  RiskManager(method="atr", atr_multiplier=2.0, rr_ratio=3.0)),
]


# ── 100 configuraciones ────────────────────────────────────────────────────────

def build_configs():
    configs = []

    def add(name, strat, rm):
        configs.append((name, strat, rm))

    # ── SMC (4 variantes × 5 RM = 20) ────────────────────────────────────
    for sw in [3, 5, 7, 10]:
        for rm_name, rm in RM_PROFILES:
            add(f"SMC(sw{sw})+{rm_name}", SMCStrategy(swing_window=sw), rm)

    # ── Williams %R (4 × 5 = 20) ─────────────────────────────────────────
    for p, os_, et in [(7,-80,20), (14,-80,20), (14,-80,50), (21,-80,50)]:
        for rm_name, rm in RM_PROFILES:
            add(f"WR(p{p},et{et})+{rm_name}",
                WilliamsRStrategy(period=p, oversold=os_, ema_trend=et), rm)

    # ── Parabolic SAR (4 × 5 = 20) ───────────────────────────────────────
    for af, afmax, et in [(0.02,0.2,20),(0.02,0.2,50),(0.01,0.1,20),(0.01,0.2,50)]:
        for rm_name, rm in RM_PROFILES:
            add(f"SAR(af{af},et{et})+{rm_name}",
                ParabolicSARStrategy(af_start=af, af_max=afmax, ema_trend=et), rm)

    # ── HeikinAshi (3 × 5 = 15) ──────────────────────────────────────────
    for cb, et in [(2,20), (2,50), (3,20)]:
        for rm_name, rm in RM_PROFILES:
            add(f"HA(cb{cb},et{et})+{rm_name}",
                HeikinAshiStrategy(confirm_bars=cb, ema_trend=et), rm)

    # ── CCI (2 × 5 = 10) ─────────────────────────────────────────────────
    for p, t, et in [(14,100,20),(20,100,50)]:
        for rm_name, rm in RM_PROFILES:
            add(f"CCI(p{p},et{et})+{rm_name}",
                CCIStrategy(period=p, threshold=t, ema_trend=et), rm)

    # ── Chandelier Exit (2 × 5 = 10) ─────────────────────────────────────
    for p, m, et in [(14,2.0,50),(22,3.0,50)]:
        for rm_name, rm in RM_PROFILES:
            add(f"Chandelier(p{p})+{rm_name}",
                ChandelierExitStrategy(period=p, mult=m, ema_trend=et), rm)

    # ── Turtle (1 × 5 = 5) ───────────────────────────────────────────────
    for rm_name, rm in RM_PROFILES:
        add(f"Turtle(en15)+{rm_name}",
            TurtleBreakoutStrategy(entry_period=15, exit_period=7, ema_trend=50), rm)

    assert len(configs) == 100, f"Expected 100, got {len(configs)}"
    return configs


# ── Motor ──────────────────────────────────────────────────────────────────────

def _run_one(name, strat, rm, datasets):
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

            trades, eq = _run_kelly_trades(sigs, df, SIZER,
                                           max_hold=MAX_HOLD,
                                           risk_manager=rm)
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
    gross_pct = float(np.mean(gross_returns)) if gross_returns else 0.0

    trades_pp    = total_trades / n_pairs if n_pairs else 0
    fee_cost_pct = trades_pp * FEE_RT * avg_frac * 100
    net_pct      = gross_pct - fee_cost_pct

    gross_sharpe = avg("sharpe")
    net_sharpe   = (gross_sharpe * net_pct / gross_pct
                    if abs(gross_pct) > 1e-6 else 0.0)

    candles_total = sum(len(df) for df in datasets.values())
    months        = candles_total / (720 * n_pairs) if n_pairs else 1
    t_month       = trades_pp / months

    min_len = min(len(e) for e in all_eq)
    eq_mean = np.mean([e[:min_len] for e in all_eq], axis=0).tolist()

    sl_hits = int(np.mean([m["sl_hits"] for m in all_m]))
    tp_hits = int(np.mean([m["tp_hits"] for m in all_m]))

    return {
        "name":         name,
        "net_sharpe":   round(net_sharpe,   3),
        "gross_sharpe": round(gross_sharpe, 3),
        "net_pct":      round(net_pct,      2),
        "gross_pct":    round(gross_pct,    2),
        "fee_pct":      round(fee_cost_pct, 2),
        "winrate":      round(avg("winrate") * 100, 1),
        "pf":           round(avg("profit_factor"), 2),
        "max_dd":       round(avg("max_drawdown") * 100, 1),
        "t_month":      round(t_month, 1),
        "n_trades":     total_trades,
        "calmar":       round(avg("calmar"), 2),
        "frac_pct":     round(avg_frac * 100, 1),
        "sl_hits":      sl_hits,
        "tp_hits":      tp_hits,
        "eq_curve":     eq_mean,
    }


# ── Presentación ───────────────────────────────────────────────────────────────

def print_top10(rows):
    profitable = sorted([r for r in rows if r and r["net_pct"] > 0],
                        key=lambda x: -x["net_sharpe"])

    print(f"\n{'='*105}")
    print(f"  TOP 10 — SMC+SL/TP — 1H FULL KELLY — 1 AÑO REAL")
    print(f"  RiskManager ATR-based · {len(profitable)}/{len(rows)} configs rentables")
    print(f"{'='*105}")
    print(f"  {'#':>3}  {'Estrategia':<42}  "
          f"{'NetSh':>6}  {'Net%/yr':>8}  {'Gross%':>7}  "
          f"{'WR%':>5}  {'PF':>5}  {'t/mes':>5}  "
          f"{'TP%':>5}  {'SL%':>5}  {'MaxDD':>6}  {'Cal':>5}  {'Fra%':>4}")
    print("  " + "─"*103)

    medals = {0:"🥇", 1:"🥈", 2:"🥉"}
    for i, r in enumerate(profitable[:10]):
        m      = medals.get(i, "  ")
        trades = r["tp_hits"] + r["sl_hits"]
        tp_pct = r["tp_hits"] / trades * 100 if trades else 0
        sl_pct = r["sl_hits"] / trades * 100 if trades else 0
        print(
            f"  {i+1:>3}. {m} {r['name']:<40}  "
            f"{r['net_sharpe']:>6.3f}  "
            f"{r['net_pct']:>+8.2f}%  "
            f"{r['gross_pct']:>+7.2f}%  "
            f"{r['winrate']:>5.1f}%  "
            f"{r['pf']:>5.2f}  "
            f"{r['t_month']:>5.1f}  "
            f"{tp_pct:>5.0f}%  "
            f"{sl_pct:>5.0f}%  "
            f"{r['max_dd']:>+6.1f}%  "
            f"{r['calmar']:>5.2f}  "
            f"{r['frac_pct']:>3.0f}%"
        )

    if profitable:
        best = profitable[0]
        print(f"\n{'='*105}")
        print(f"  ★ GANADORA: {best['name']}")
        print(f"     Net Sharpe : {best['net_sharpe']:.3f}")
        print(f"     Net Return : {best['net_pct']:+.2f}%/año  "
              f"(Gross {best['gross_pct']:+.2f}% − Fees {best['fee_pct']:.2f}%)")
        print(f"     Win Rate   : {best['winrate']:.1f}%   PF: {best['pf']:.2f}   "
              f"Calmar: {best['calmar']:.2f}")
        print(f"     Trades/mes : {best['t_month']:.1f}   "
              f"Frac Kelly: {best['frac_pct']:.0f}%   MaxDD: {best['max_dd']:+.1f}%")

        # Proyección con €100
        pct_mes = (1 + best["net_pct"] / 100) ** (1/12) - 1
        print(f"\n  {'─'*55}")
        print(f"  PROYECCIÓN (reinvirtiendo ganancias):")
        cap = 100.0
        for m_num in [1, 3, 6, 12, 24, 36]:
            val = cap * (1 + best["net_pct"] / 100) ** (m_num / 12)
            print(f"    {m_num:>2} meses  →  €{val:>8.2f}  "
                  f"(+€{val-cap:>7.2f},  {(val/cap-1)*100:>+.1f}%)")
        print(f"\n  Para €100/mes necesitas: "
              f"€{100/pct_mes:>,.0f} de capital")
    print(f"{'='*105}\n")


def save_plot(rows):
    top = sorted([r for r in rows if r and r["net_pct"] > 0],
                 key=lambda x: -x["net_sharpe"])[:10]
    if not top:
        return

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")
    gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)

    def _ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        return ax

    # 1. Equity curves top 10
    ax1 = _ax(gs[0, :])
    pal = plt.cm.plasma(np.linspace(0.1, 0.95, len(top)))
    for r, col in zip(top, pal):
        eq = np.array(r["eq_curve"])
        x  = np.linspace(0, 12, len(eq))
        ax1.plot(x, eq, label=r["name"][:30], lw=1.5, color=col, alpha=0.9)
    ax1.axhline(1.0, color="#30363d", ls="--", lw=0.8)
    ax1.set_xlabel("Meses", color="#8b949e")
    ax1.set_ylabel("Capital (inicio=1.0)", color="#8b949e")
    ax1.set_title("Equity Curves Top 10  |  1H · Full Kelly · RiskManager ATR",
                  color="#e6edf3", fontsize=10)
    ax1.legend(fontsize=6, framealpha=0.15, labelcolor="#c9d1d9",
               loc="upper left", ncol=2)

    # 2. Net% barchart
    ax2 = _ax(gs[1, 0])
    names  = [r["name"][:26] for r in top]
    values = [r["net_pct"] for r in top]
    bars = ax2.barh(names, values, color="#3fb950", alpha=0.85,
                    edgecolor="#30363d", lw=0.5)
    ax2.axvline(0, color="#8b949e", lw=0.8)
    ax2.set_xlabel("Net Return % / año", color="#8b949e")
    ax2.set_title("Retorno Neto Anual (post-fees)", color="#e6edf3")
    ax2.tick_params(axis="y", labelsize=7, colors="#c9d1d9")
    for bar, v in zip(bars, values):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{v:+.1f}%", va="center", fontsize=7, color="#c9d1d9")

    # 3. Scatter t/mes vs Net% — burbuja=Sharpe
    ax3 = _ax(gs[1, 1])
    all_p = sorted([r for r in rows if r and r["net_pct"] > 0],
                   key=lambda x: -x["net_sharpe"])
    sc = ax3.scatter(
        [r["t_month"]   for r in all_p],
        [r["net_pct"]   for r in all_p],
        s=[max(20, r["net_sharpe"] * 150) for r in all_p],
        c=[r["net_sharpe"] for r in all_p],
        cmap="plasma", alpha=0.85, edgecolors="#30363d", lw=0.4
    )
    for r in all_p[:10]:
        ax3.annotate(r["name"][:20], (r["t_month"], r["net_pct"]),
                     fontsize=5.5, color="#8b949e",
                     xytext=(3, 3), textcoords="offset points")
    ax3.axhline(0, color="#30363d", ls="--", lw=0.8)
    ax3.set_xlabel("Trades / mes", color="#8b949e")
    ax3.set_ylabel("Net Return % / año", color="#8b949e")
    ax3.set_title("Frecuencia vs Rentabilidad", color="#e6edf3")
    cb = fig.colorbar(sc, ax=ax3, pad=0.02)
    cb.set_label("Net Sharpe", color="#8b949e", fontsize=7)
    cb.ax.tick_params(colors="#8b949e", labelsize=6)

    fig.suptitle(
        "RM Tournament · 1H · Full Kelly · max_frac=40% · 5 pares KuCoin · 1 año",
        color="#e6edf3", fontsize=11, y=0.99
    )
    out = "data/rm_tournament.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Plot: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    from data.kucoin_client import KuCoinClient
    client = KuCoinClient()

    print("\n📡 Descargando 1 año de datos 1H...")
    datasets = {}
    for pair in PAIRS:
        try:
            df = client.get_ohlcv_paginated(pair, interval="1hour", days=365)
            if df is not None and len(df) > 500:
                datasets[pair] = df
                print(f"  ✅ {pair}: {len(df)} velas")
            else:
                print(f"  ⚠ {pair}: datos insuficientes")
        except Exception as e:
            print(f"  ❌ {pair}: {e}")
        time.sleep(0.3)

    if not datasets:
        print("❌ Sin datos.")
        return

    configs = build_configs()
    print(f"\n  {len(configs)} configs · {len(datasets)} pares · full_kelly max_frac=40%\n")

    rows = []
    t0   = time.time()
    for i, (name, strat, rm) in enumerate(configs, 1):
        row = _run_one(name, strat, rm, datasets)
        if row:
            rows.append(row)
        if i % 25 == 0:
            elapsed = time.time() - t0
            profitable = sum(1 for r in rows if r["net_pct"] > 0)
            print(f"  [{i:3d}/100]  rentables hasta ahora: {profitable}  ⏱ {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\n  Completado en {elapsed:.1f}s")

    # CSV
    csv_rows = [{k:v for k,v in r.items() if k != "eq_curve"} for r in rows]
    df_out   = pd.DataFrame(csv_rows).sort_values("net_sharpe", ascending=False)
    df_out.to_csv("data/rm_tournament_results.csv", index=False)
    print(f"  💾 data/rm_tournament_results.csv")

    print_top10(rows)
    save_plot(rows)
    return rows


if __name__ == "__main__":
    main()
