"""
aggressive_tournament.py

Torneo agresivo: más retorno, más pares, más Kelly.

Mejoras sobre rm_tournament.py:
  1. MultiTFSMC  — señales 1H filtradas por tendencia 4H → WR +5-8 pp
  2. max_fraction = 0.60  — Kelly puede usar hasta 60% del capital
  3. 10 pares           — BTC ETH SOL XRP BNB ADA AVAX DOGE LTC DOT
  4. 100 configs        — MTF-SMC × RM profiles + SMC base de comparación

Timeframe : 1H · 1 año · full_kelly
Fees      : 0.2% round-trip
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
from strategies.smc_strategy   import SMCStrategy
from strategies.mtf_smc        import MultiTFSMC

# ── Config ─────────────────────────────────────────────────────────────────────

PAIRS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BNB-USDT",
    "ADA-USDT", "AVAX-USDT", "DOGE-USDT", "LTC-USDT", "DOT-USDT",
]
MAX_HOLD = 24
PPY      = 8760
FEE_RT   = 0.002

# Tres niveles de agresividad
SIZER_40 = KellySizer(variant="full_kelly", min_trades=20,
                      max_fraction=0.40, min_fraction=0.01)
SIZER_50 = KellySizer(variant="full_kelly", min_trades=20,
                      max_fraction=0.50, min_fraction=0.01)
SIZER_60 = KellySizer(variant="full_kelly", min_trades=20,
                      max_fraction=0.60, min_fraction=0.01)

RM_PROFILES = [
    ("RM1x2",  RiskManager(method="atr", atr_multiplier=1.0, rr_ratio=2.0)),
    ("RM1x3",  RiskManager(method="atr", atr_multiplier=1.0, rr_ratio=3.0)),
    ("RM15x2", RiskManager(method="atr", atr_multiplier=1.5, rr_ratio=2.0)),
    ("RM15x3", RiskManager(method="atr", atr_multiplier=1.5, rr_ratio=3.0)),
    ("RM2x2",  RiskManager(method="atr", atr_multiplier=2.0, rr_ratio=2.0)),
]


# ── 100 configuraciones ────────────────────────────────────────────────────────

def build_configs():
    configs = []  # (name, strat, rm, sizer)

    def add(n, s, rm, sz):
        configs.append((n, s, rm, sz))

    # ── MTF-SMC × 5 RM × 3 max_frac = 60 configs ─────────────────────
    for sw, ema4h in [(3,50),(5,50),(5,100),(7,50),(10,50),(10,100)]:
        for rm_name, rm in RM_PROFILES:
            for frac, sz in [(40, SIZER_40),(50, SIZER_50),(60, SIZER_60)]:
                if len(configs) < 60:   # primero llenamos 60 MTF
                    add(f"MTF(sw{sw},e{ema4h})+{rm_name}+f{frac}",
                        MultiTFSMC(swing_window=sw, trend_ema=ema4h), rm, sz)

    # ── SMC base + RM (comparación) × 5 RM × 2 frac = 40 configs ─────
    for sw in [3, 5, 7, 10]:
        for rm_name, rm in RM_PROFILES:
            for frac, sz in [(50, SIZER_50),(60, SIZER_60)]:
                if len(configs) < 100:
                    add(f"SMC(sw{sw})+{rm_name}+f{frac}",
                        SMCStrategy(swing_window=sw), rm, sz)

    assert len(configs) == 100, f"Expected 100, got {len(configs)}"
    return configs


# ── Motor ──────────────────────────────────────────────────────────────────────

def _run_one(name, strat, rm, sizer, datasets):
    all_m  = []
    all_eq = []

    for pair, df in datasets.items():
        try:
            if isinstance(strat, MultiTFSMC):
                sigs = strat.generate_signals_batch(df)
            elif isinstance(strat, SMCStrategy):
                sigs = _precompute_signals(
                    df,
                    swing_window     = strat.swing_window,
                    require_fvg      = strat.require_fvg,
                    use_choch_filter = strat.use_choch_filter,
                )
            else:
                sigs = strat.generate_signals_batch(df)

            trades, eq = _run_kelly_trades(sigs, df, sizer,
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

    months   = sum(len(df) for df in datasets.values()) / (720 * n_pairs)
    t_month  = trades_pp / months if months else 0

    min_len  = min(len(e) for e in all_eq)
    eq_mean  = np.mean([e[:min_len] for e in all_eq], axis=0).tolist()

    sl_hits = int(np.mean([m["sl_hits"] for m in all_m]))
    tp_hits = int(np.mean([m["tp_hits"] for m in all_m]))

    return {
        "name":       name,
        "net_sharpe": round(net_sharpe,   3),
        "net_pct":    round(net_pct,      2),
        "gross_pct":  round(gross_pct,    2),
        "fee_pct":    round(fee_cost_pct, 2),
        "winrate":    round(avg("winrate") * 100, 1),
        "pf":         round(avg("profit_factor"), 2),
        "max_dd":     round(avg("max_drawdown") * 100, 1),
        "t_month":    round(t_month, 1),
        "n_trades":   total_trades,
        "calmar":     round(avg("calmar"), 2),
        "frac_pct":   round(avg_frac * 100, 1),
        "sl_hits":    sl_hits,
        "tp_hits":    tp_hits,
        "eq_curve":   eq_mean,
    }


# ── Presentación ───────────────────────────────────────────────────────────────

def print_top10(rows):
    profitable = sorted([r for r in rows if r and r["net_pct"] > 0],
                        key=lambda x: -x["net_sharpe"])

    print(f"\n{'='*108}")
    print(f"  TORNEO AGRESIVO — MTF-SMC + RiskManager + full_kelly — 1H 1 AÑO 10 PARES")
    print(f"  Rentables: {len(profitable)}/{len(rows)}  |  max_fraction hasta 60%")
    print(f"{'='*108}")
    hdr = (f"  {'#':>3}  {'Estrategia':<44}  "
           f"{'NetSh':>6}  {'Net%/yr':>8}  {'Gross%':>7}  "
           f"{'WR%':>5}  {'PF':>5}  {'t/mes':>5}  "
           f"{'MaxDD':>6}  {'Cal':>6}  {'Fra%':>4}")
    print(hdr)
    print("  " + "─" * 106)

    medals = {0:"🥇", 1:"🥈", 2:"🥉"}
    for i, r in enumerate(profitable[:10]):
        m = medals.get(i, "  ")
        print(
            f"  {i+1:>3}. {m} {r['name']:<42}  "
            f"{r['net_sharpe']:>6.3f}  "
            f"{r['net_pct']:>+8.2f}%  "
            f"{r['gross_pct']:>+7.2f}%  "
            f"{r['winrate']:>5.1f}%  "
            f"{r['pf']:>5.2f}  "
            f"{r['t_month']:>5.1f}  "
            f"{r['max_dd']:>+6.1f}%  "
            f"{r['calmar']:>6.2f}  "
            f"{r['frac_pct']:>3.0f}%"
        )

    if not profitable:
        print("  ⚠ Ninguna config rentable.")
        return

    best = profitable[0]
    pct_mes = (1 + best["net_pct"] / 100) ** (1/12) - 1

    print(f"\n{'='*108}")
    print(f"  ★  GANADORA: {best['name']}")
    print(f"     Net Sharpe  : {best['net_sharpe']:.3f}")
    print(f"     Net Return  : {best['net_pct']:+.2f}%/año  "
          f"(Gross {best['gross_pct']:+.2f}% − Fees {best['fee_pct']:.2f}%)")
    print(f"     Win Rate    : {best['winrate']:.1f}%  |  PF: {best['pf']:.2f}  "
          f"|  Calmar: {best['calmar']:.2f}  |  MaxDD: {best['max_dd']:+.1f}%")
    print(f"     Trades/mes  : {best['t_month']:.1f}  |  Kelly frac: {best['frac_pct']:.0f}%")

    print(f"\n  PROYECCIÓN (€100 reinvirtiendo):")
    cap = 100.0
    for meses in [1, 3, 6, 12, 24, 36]:
        val    = cap * (1 + best["net_pct"] / 100) ** (meses / 12)
        ganado = val - cap
        print(f"    {meses:>2} meses  →  €{val:>8.2f}  (+€{ganado:>7.2f} / {(val/cap-1)*100:>+.1f}%)")

    print(f"\n  Capital necesario para €X/mes con esta estrategia:")
    for obj in [50, 100, 200, 500, 1000]:
        cap_needed = obj / pct_mes
        print(f"    €{obj:>5}/mes  →  €{cap_needed:>9,.0f}")
    print(f"{'='*108}\n")


def save_plot(rows):
    top = sorted([r for r in rows if r and r["net_pct"] > 0],
                 key=lambda x: -x["net_sharpe"])[:12]
    if not top:
        return

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#0d1117")
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    def _ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        return ax

    pal = plt.cm.plasma(np.linspace(0.05, 0.95, len(top)))

    # 1. Equity curves (span 2 cols)
    ax1 = _ax(gs[0, :2])
    for r, col in zip(top, pal):
        eq = np.array(r["eq_curve"])
        x  = np.linspace(0, 12, len(eq))
        ax1.plot(x, eq, label=r["name"][:32], lw=1.5, color=col, alpha=0.9)
    ax1.axhline(1.0, color="#30363d", ls="--", lw=0.7)
    ax1.set_xlabel("Meses", color="#8b949e")
    ax1.set_ylabel("Capital (1.0 = inicio)", color="#8b949e")
    ax1.set_title("Equity Curves Top 12  |  1H · MTF-SMC · Full Kelly",
                  color="#e6edf3", fontsize=10)
    ax1.legend(fontsize=6, framealpha=0.1, labelcolor="#c9d1d9",
               loc="upper left", ncol=2)

    # 2. Resumen MTF vs SMC base (net%)
    ax_comp = _ax(gs[0, 2])
    all_p = [r for r in rows if r and r["net_pct"] > 0]
    mtf   = [r["net_pct"] for r in all_p if "MTF" in r["name"]]
    smc   = [r["net_pct"] for r in all_p if "SMC" in r["name"] and "MTF" not in r["name"]]
    ax_comp.boxplot([smc, mtf], labels=["SMC base", "MTF-SMC"],
                    patch_artist=True,
                    boxprops=dict(facecolor="#1f6feb", alpha=0.7),
                    medianprops=dict(color="#f0e040", lw=2))
    ax_comp.set_ylabel("Net% / año", color="#8b949e")
    ax_comp.set_title("SMC vs MTF-SMC\nDistribución retornos", color="#e6edf3", fontsize=9)
    ax_comp.tick_params(axis="x", colors="#c9d1d9")

    # 3. Net% barchart
    ax2 = _ax(gs[1, :2])
    names  = [r["name"][:32] for r in top]
    values = [r["net_pct"] for r in top]
    bar_c  = ["#3fb950" if "MTF" in n else "#4fc3f7" for n in names]
    bars   = ax2.barh(names, values, color=bar_c, alpha=0.85,
                      edgecolor="#30363d", lw=0.4)
    ax2.axvline(0, color="#8b949e", lw=0.8)
    ax2.set_xlabel("Net Return % / año", color="#8b949e")
    ax2.set_title("Retorno Neto  (verde=MTF, azul=SMC base)", color="#e6edf3", fontsize=9)
    ax2.tick_params(axis="y", labelsize=7, colors="#c9d1d9")
    for bar, v in zip(bars, values):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{v:+.1f}%", va="center", fontsize=7, color="#c9d1d9")

    # 4. Scatter t/mes vs Net% — burbuja=Sharpe
    ax3 = _ax(gs[1, 2])
    sc = ax3.scatter(
        [r["t_month"]   for r in all_p],
        [r["net_pct"]   for r in all_p],
        s=[max(15, r["net_sharpe"] * 100) for r in all_p],
        c=[r["net_sharpe"] for r in all_p],
        cmap="plasma", alpha=0.85, edgecolors="#30363d", lw=0.3
    )
    ax3.axhline(0, color="#30363d", ls="--", lw=0.7)
    ax3.set_xlabel("Trades / mes", color="#8b949e")
    ax3.set_ylabel("Net Return % / año", color="#8b949e")
    ax3.set_title("Frecuencia vs Rentabilidad", color="#e6edf3", fontsize=9)
    cb = fig.colorbar(sc, ax=ax3, pad=0.02)
    cb.set_label("Net Sharpe", color="#8b949e", fontsize=7)
    cb.ax.tick_params(colors="#8b949e", labelsize=6)

    fig.suptitle(
        "Torneo Agresivo · MTF-SMC · full_kelly max_frac=60% · 10 pares · 1H 1 año",
        color="#e6edf3", fontsize=11, y=0.99
    )
    out = "data/aggressive_tournament.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Plot: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    from data.kucoin_client import KuCoinClient
    client = KuCoinClient()

    print("\n📡 Descargando 1 año 1H para 10 pares...")
    datasets = {}
    for pair in PAIRS:
        try:
            df = client.get_ohlcv_paginated(pair, interval="1hour", days=365)
            if df is not None and len(df) > 500:
                datasets[pair] = df
                print(f"  ✅ {pair}: {len(df)} velas")
            else:
                print(f"  ⚠ {pair}: insuficiente")
        except Exception as e:
            print(f"  ❌ {pair}: {e}")
        time.sleep(0.25)

    if not datasets:
        print("❌ Sin datos.")
        return

    configs = build_configs()
    n_pairs = len(datasets)
    print(f"\n  {len(configs)} configs · {n_pairs} pares activos\n")

    rows = []
    t0   = time.time()
    for i, (name, strat, rm, sizer) in enumerate(configs, 1):
        row = _run_one(name, strat, rm, sizer, datasets)
        if row:
            rows.append(row)
        if i % 20 == 0:
            elapsed    = time.time() - t0
            profitable = sum(1 for r in rows if r["net_pct"] > 0)
            best_so_far = max((r["net_pct"] for r in rows if r["net_pct"] > 0), default=0)
            print(f"  [{i:3d}/100]  rentables: {profitable}  "
                  f"mejor hasta ahora: {best_so_far:+.2f}%  ⏱ {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\n  Completado en {elapsed:.1f}s")

    # CSV
    csv_rows = [{k:v for k,v in r.items() if k != "eq_curve"} for r in rows]
    df_out   = pd.DataFrame(csv_rows).sort_values("net_sharpe", ascending=False)
    df_out.to_csv("data/aggressive_tournament_results.csv", index=False)
    print(f"  💾 data/aggressive_tournament_results.csv")

    print_top10(rows)
    save_plot(rows)
    return rows


if __name__ == "__main__":
    main()
