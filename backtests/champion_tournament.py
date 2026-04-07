"""
champion_tournament.py

TORNEO CAMPEÓN — optimización del ganador + 6 nuevas estrategias.

Fase 1: Grid search sobre TRAIL·REG[light]·AdaptMTF
  - trailing_atr_mult : 1.0, 1.5, 2.0, 2.5, 3.0
  - partial_ratio     : 0.33, 0.50, 0.67
  - adx_min           : 13, 16, 20

Fase 2: 6 nuevas estrategias (SuperTrend, VWAP, HullMA, Keltner, PinBar, RSIDivergence)
  × REG[light] + TRAIL optimizado

Timeframe : 1H · 1 año · full_kelly
Pares     : 10
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from backtests.kelly_backtest_v2     import run_kelly_v2, kelly_metrics_v2
from strategies.kelly_sizer          import KellySizer
from strategies.risk_manager         import RiskManager
from strategies.mtf_smc              import MultiTFSMC
from strategies.advanced_strategies  import AdaptiveMTFStrategy
from strategies.regime_filter        import RegimeFilteredStrategy
from strategies.price_action_strategies import (
    SuperTrendStrategy, VWAPStrategy, HullMAStrategy,
    KeltnerBreakoutStrategy, PinBarStrategy, RSIDivergenceStrategy,
)

# ── Config ────────────────────────────────────────────────────────────────────

PAIRS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BNB-USDT",
    "ADA-USDT", "AVAX-USDT", "DOGE-USDT", "LTC-USDT", "DOT-USDT",
]
MAX_HOLD = 24
PPY      = 8760
FEE_RT   = 0.002

SIZER = KellySizer(variant="full_kelly", min_trades=20,
                   max_fraction=0.60, min_fraction=0.01)
RM    = RiskManager(method="atr", atr_multiplier=1.0, rr_ratio=2.0)

# Anterior campeona (referencia)
PREV_CHAMP = "TRAIL·REG[light]·AdaptMTF  (Net +105.94%/yr  Sh 5.941  WR 63.4%)"


# ── Demo data ─────────────────────────────────────────────────────────────────

def _gen_demo(symbol: str, n: int = 8760) -> pd.DataFrame:
    seeds = {p: i for i, p in enumerate(PAIRS)}
    rng = np.random.default_rng(seeds.get(symbol, 0))
    base = {"BTC-USDT": 40000, "ETH-USDT": 2000,
            "SOL-USDT": 80,    "BNB-USDT": 300}.get(symbol, 10.0)
    p = [base]
    for _ in range(n - 1):
        p.append(max(1e-4, p[-1] + 0.3 + rng.standard_normal() * p[-1] * 0.008))
    p = np.array(p)
    o = np.concatenate([[p[0]], p[:-1]])
    s = np.abs(rng.standard_normal(n)) * p * 0.003
    ts = pd.to_datetime([int(time.time()) - (n-i)*3600 for i in range(n)],
                        unit="s", utc=True)
    return pd.DataFrame({"open": o, "high": np.maximum(o, p)+s,
                         "low": np.minimum(o, p)-s, "close": p,
                         "volume": rng.integers(500, 20000, n).astype(float)},
                        index=ts)


# ── Motor ─────────────────────────────────────────────────────────────────────

def _run(name, strat, datasets, trailing_atr=2.0, partial_ratio=0.50):
    all_m, all_eq = [], []
    for df in datasets.values():
        try:
            sigs = strat.generate_signals_batch(df)
            trades, eq = run_kelly_v2(
                sigs, df, SIZER,
                max_hold=MAX_HOLD, risk_manager=RM,
                trailing_atr_mult=trailing_atr,
                partial_tp=True, partial_ratio=partial_ratio,
            )
            m = kelly_metrics_v2(trades, eq, periods_per_year=PPY)
            all_m.append(m); all_eq.append(eq)
        except Exception:
            pass

    if not all_m:
        return None

    def avg(k): return float(np.mean([m[k] for m in all_m]))

    n_pairs      = len(all_m)
    total_trades = sum(m["trades"] for m in all_m)
    avg_frac     = avg("avg_fraction")
    gross_returns = [(m["equity_curve"][-1] - 1.0)*100
                     for m in all_m if m["equity_curve"]]
    gross_pct    = float(np.mean(gross_returns)) if gross_returns else 0.0
    fee_pct      = (total_trades / n_pairs) * FEE_RT * avg_frac * 100
    net_pct      = gross_pct - fee_pct
    gross_sh     = avg("sharpe")
    net_sh       = gross_sh * net_pct / gross_pct if abs(gross_pct) > 1e-6 else 0.0
    months       = sum(len(df) for df in datasets.values()) / (720 * n_pairs)
    t_month      = (total_trades / n_pairs) / months if months else 0
    min_len      = min(len(e) for e in all_eq)
    eq_mean      = np.mean([e[:min_len] for e in all_eq], axis=0).tolist()

    return {
        "name": name, "net_sharpe": round(net_sh, 3),
        "net_pct": round(net_pct, 2), "gross_pct": round(gross_pct, 2),
        "fee_pct": round(fee_pct, 2), "winrate": round(avg("winrate")*100, 1),
        "pf": round(avg("profit_factor"), 2),
        "max_dd": round(avg("max_drawdown")*100, 1),
        "t_month": round(t_month, 1), "calmar": round(avg("calmar"), 2),
        "frac_pct": round(avg_frac*100, 1), "eq_curve": eq_mean,
        "partial_tp_hits": int(np.mean([m.get("partial_tp_hits",0) for m in all_m])),
        "trail_hits":      int(np.mean([m.get("trail_hits",0) for m in all_m])),
    }


# ── Fase 1: Optimización del campeón ─────────────────────────────────────────

def run_optimization(datasets):
    print("\n" + "─"*60)
    print("  FASE 1 — Optimización TRAIL·REG·AdaptMTF")
    print("─"*60)

    results = []
    base    = AdaptiveMTFStrategy()

    trailing_vals  = [1.0, 1.5, 2.0, 2.5, 3.0]
    partial_vals   = [0.33, 0.50, 0.67]
    adx_vals       = [13, 16, 20]

    total = len(trailing_vals) * len(partial_vals) * len(adx_vals)
    done  = 0

    for trail in trailing_vals:
        for partial in partial_vals:
            for adx in adx_vals:
                strat = RegimeFilteredStrategy(
                    base, adx_min=adx, di_align=True,
                    atr_min_pct=0.2, atr_max_pct=6.0,
                )
                name = f"OPT·trail{trail}·p{partial}·adx{adx}"
                r = _run(name, strat, datasets,
                         trailing_atr=trail, partial_ratio=partial)
                results.append(r)
                done += 1
                if done % 10 == 0:
                    print(f"  [{done}/{total}] completados...")

    valid = sorted([r for r in results if r and r["net_pct"] > 0],
                   key=lambda x: -x["net_sharpe"])

    print(f"\n  Top 5 configuraciones optimizadas:")
    print(f"  {'#':>3}  {'Config':<38}  {'NetSh':>6}  {'Net%':>8}  "
          f"{'WR%':>5}  {'PF':>5}  {'t/mes':>5}")
    print("  " + "─"*75)
    for i, r in enumerate(valid[:5]):
        print(f"  {i+1:>3}. {r['name']:<38}  "
              f"{r['net_sharpe']:>6.3f}  {r['net_pct']:>+8.2f}%  "
              f"{r['winrate']:>5.1f}%  {r['pf']:>5.2f}  {r['t_month']:>5.1f}")

    best_opt = valid[0] if valid else None
    return results, best_opt


# ── Fase 2: Nuevas estrategias ────────────────────────────────────────────────

def run_new_strategies(datasets, best_trail=2.0, best_partial=0.50, best_adx=16):
    print("\n" + "─"*60)
    print("  FASE 2 — 6 Nuevas Estrategias con Trail+Régimen")
    print("─"*60)

    new_strats = [
        ("SuperTrend(10,3)",    SuperTrendStrategy(atr_period=10, multiplier=3.0)),
        ("SuperTrend(7,2.5)",   SuperTrendStrategy(atr_period=7,  multiplier=2.5)),
        ("VWAP(24,1.5)",        VWAPStrategy(session_len=24, band_mult=1.5)),
        ("HullMA(20,55)",       HullMAStrategy(fast=20, slow=55)),
        ("HullMA(10,30)",       HullMAStrategy(fast=10, slow=30)),
        ("Keltner(20,1.5)",     KeltnerBreakoutStrategy(ema_period=20, mult=1.5)),
        ("Keltner(20,2.0)",     KeltnerBreakoutStrategy(ema_period=20, mult=2.0)),
        ("PinBar(50,2.0)",      PinBarStrategy(ema_trend=50, wick_ratio=2.0)),
        ("PinBar(100,2.5)",     PinBarStrategy(ema_trend=100, wick_ratio=2.5)),
        ("RSIDiv(14,20)",       RSIDivergenceStrategy(rsi_period=14, lookback=20)),
        ("RSIDiv(14,30)",       RSIDivergenceStrategy(rsi_period=14, lookback=30)),
        # Combos ganadores de torneos anteriores también en v2
        ("MTF_sw3(v2)",         MultiTFSMC(swing_window=3, trend_ema=50)),
        ("AdaptMTF(v2-base)",   AdaptiveMTFStrategy()),
    ]

    results = []
    for strat_name, base_strat in new_strats:
        # Sin régimen (base)
        r0 = _run(f"BASE·{strat_name}", base_strat, datasets,
                  trailing_atr=best_trail, partial_ratio=best_partial)
        results.append(r0)

        # Con régimen optimizado
        filtered = RegimeFilteredStrategy(
            base_strat, adx_min=best_adx, di_align=True,
            atr_min_pct=0.2, atr_max_pct=6.0,
        )
        r1 = _run(f"REG·{strat_name}", filtered, datasets,
                  trailing_atr=best_trail, partial_ratio=best_partial)
        results.append(r1)

        status = f"{r1['net_pct']:>+7.2f}%  Sh {r1['net_sharpe']:.3f}" \
                 if r1 else "sin datos"
        print(f"  {strat_name:<25}  {status}")

    return results


# ── Presentación final ────────────────────────────────────────────────────────

def print_champion(all_results):
    valid  = [r for r in all_results if r]
    profit = sorted([r for r in valid if r["net_pct"] > 0],
                    key=lambda x: -x["net_sharpe"])

    print(f"\n{'='*118}")
    print(f"  TORNEO CAMPEÓN — {len(valid)} configs | Rentables: {len(profit)}")
    print(f"  Referencia anterior: {PREV_CHAMP}")
    print(f"{'='*118}")
    print(f"  {'#':>3}  {'Estrategia':<44}  "
          f"{'NetSh':>6}  {'Net%/yr':>8}  {'WR%':>5}  "
          f"{'PF':>5}  {'t/mes':>5}  {'MaxDD':>6}  {'Cal':>6}")
    print("  " + "─"*116)

    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    for i, r in enumerate(profit[:15]):
        m = medals.get(i, "  ")
        print(
            f"  {i+1:>3}. {m} {r['name']:<42}  "
            f"{r['net_sharpe']:>6.3f}  {r['net_pct']:>+8.2f}%  "
            f"{r['winrate']:>5.1f}%  {r['pf']:>5.2f}  "
            f"{r['t_month']:>5.1f}  {r['max_dd']:>+6.1f}%  "
            f"{r['calmar']:>6.2f}"
        )

    if not profit:
        return

    best    = profit[0]
    pct_yr  = best["net_pct"]
    pct_mo  = (1 + pct_yr/100)**(1/12) - 1

    improved = pct_yr > 105.94
    tag      = f"  ↑ MEJORA vs anterior (+{pct_yr-105.94:.2f}pp)" \
               if improved else f"  → Anterior sigue siendo referencia"

    print(f"\n{'='*118}")
    print(f"  NUEVA CAMPEONA: {best['name']}")
    print(f"  Net Sharpe: {best['net_sharpe']:.3f}  |  "
          f"Net Return: {pct_yr:+.2f}%/año  |  "
          f"WR: {best['winrate']:.1f}%  |  PF: {best['pf']:.2f}  |  "
          f"MaxDD: {best['max_dd']:+.1f}%  |  Calmar: {best['calmar']:.2f}")
    print(tag)

    print(f"\n  ESCALADA MENSUAL (tasa {pct_mo*100:.2f}%/mes):")
    print(f"  {'─'*100}")
    print(f"  {'Capital':<12}  " +
          "  ".join(f"{'mes '+str(m):>9}" for m in [1,3,6,12,18,24,36]))
    print(f"  {'─'*100}")
    for cap in [100, 500, 1_000, 5_000, 10_000, 50_000]:
        row = f"  €{cap:<11,.0f}"
        for m in [1, 3, 6, 12, 18, 24, 36]:
            val = cap * (1 + pct_yr/100)**(m/12)
            row += f"  +€{val-cap:>7,.0f}"
        print(row)
    print(f"  {'─'*100}")

    print(f"\n  Para €100/mes necesitas: €{100/pct_mo:>8,.0f} de capital")
    print(f"  Para €500/mes necesitas: €{500/pct_mo:>8,.0f} de capital")
    print(f"{'='*118}\n")

    return profit


def save_plot(all_results, output="data/champion_tournament.png"):
    valid  = [r for r in all_results if r]
    profit = sorted([r for r in valid if r["net_pct"] > 0],
                    key=lambda x: -x["net_sharpe"])
    if not profit:
        return

    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor("#0d1117")
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    def _ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        return ax

    pal = plt.cm.plasma(np.linspace(0.05, 0.95, min(15, len(profit))))

    # 1. Equity curves
    ax1 = _ax(gs[0, :2])
    for i, r in enumerate(profit[:15]):
        eq = r.get("eq_curve", [])
        if eq:
            xs = np.linspace(0, 12, len(eq))
            ax1.plot(xs, [(v-1)*100 for v in eq],
                     color=pal[i], lw=1.4, alpha=0.75,
                     label=r["name"][:28] if i < 5 else "")
    ax1.axhline(0, color="#8b949e", lw=0.8, ls="--")
    ax1.legend(fontsize=7, framealpha=0.3, labelcolor="#c9d1d9")
    ax1.set_title("Top 15 — Equity Curves", color="#f0f6fc", fontsize=10)
    ax1.set_xlabel("Meses", color="#8b949e")
    ax1.set_ylabel("Retorno (%)", color="#8b949e")

    # 2. Sharpe vs WR
    ax2 = _ax(gs[0, 2])
    xs2 = [r["winrate"]    for r in profit[:30]]
    ys2 = [r["net_sharpe"] for r in profit[:30]]
    sc  = ax2.scatter(xs2, ys2,
                      c=np.arange(len(xs2)), cmap="plasma", s=55, alpha=0.85)
    ax2.axvline(50, color="#f85149", lw=0.8, ls="--", alpha=0.5)
    ax2.set_title("Win Rate vs Net Sharpe", color="#f0f6fc", fontsize=10)
    ax2.set_xlabel("Win Rate %", color="#8b949e")
    ax2.set_ylabel("Net Sharpe", color="#8b949e")

    # 3. Bar top 15
    ax3 = _ax(gs[1, :2])
    top15   = profit[:15]
    names3  = [r["name"][:35] for r in top15]
    rets3   = [r["net_pct"]   for r in top15]
    ax3.barh(range(len(names3)), rets3, color=pal[:len(names3)], alpha=0.85)
    ax3.set_yticks(range(len(names3)))
    ax3.set_yticklabels(names3, fontsize=7)
    ax3.set_xlabel("Net Return %/año", color="#8b949e")
    ax3.set_title("Top 15 — Net Return", color="#f0f6fc", fontsize=10)
    ax3.invert_yaxis()
    ax3.axvline(105.94, color="#56d364", lw=1.5, ls="--",
                alpha=0.8, label="Ant. campeona 105.94%")
    ax3.legend(fontsize=7, framealpha=0.3, labelcolor="#c9d1d9")

    # 4. Scalability
    ax4 = _ax(gs[1, 2])
    best   = profit[0]
    pct_yr = best["net_pct"]
    caps   = [500, 1_000, 5_000, 10_000, 50_000]
    gains  = [cap * (1 + pct_yr/100) - cap for cap in caps]
    cols4  = plt.cm.viridis(np.linspace(0.3, 0.9, len(caps)))
    ax4.bar([f"€{c:,}" for c in caps], gains, color=cols4, alpha=0.9)
    ax4.set_title(f"Ganancia anual\n{best['name'][:30]}", color="#f0f6fc", fontsize=9)
    ax4.set_xlabel("Capital", color="#8b949e")
    ax4.set_ylabel("€/año", color="#8b949e")
    plt.setp(ax4.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    fig.suptitle("TORNEO CAMPEÓN — Optimización + 6 nuevas estrategias  |  1H 1 año 10 pares",
                 color="#f0f6fc", fontsize=12, y=1.01)

    os.makedirs("data", exist_ok=True)
    plt.savefig(output, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Gráfico: {output}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  TORNEO CAMPEÓN — cargando datos 1H 1 año...")
    print("="*60)

    datasets = {}
    for pair in PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        datasets[pair] = _gen_demo(pair, n=8760)
        print("OK")

    # Fase 1 — optimización
    opt_results, best_opt = run_optimization(datasets)

    # Extraer mejores parámetros
    best_trail   = 2.0
    best_partial = 0.50
    best_adx     = 16
    if best_opt:
        parts = best_opt["name"].split("·")
        for p in parts:
            if p.startswith("trail"):  best_trail   = float(p[5:])
            if p.startswith("p"):      best_partial = float(p[1:])
            if p.startswith("adx"):    best_adx     = int(p[3:])

    print(f"\n  Params óptimos → trail={best_trail}  partial={best_partial}  ADX>{best_adx}")

    # Fase 2 — nuevas estrategias
    new_results = run_new_strategies(datasets, best_trail, best_partial, best_adx)

    all_results = opt_results + new_results

    # CSV
    os.makedirs("data", exist_ok=True)
    df_out = pd.DataFrame([r for r in all_results if r])
    df_out = df_out.drop(columns=["eq_curve"], errors="ignore")
    df_out.to_csv("data/champion_results.csv", index=False)
    print(f"\n  CSV: data/champion_results.csv")

    profit = print_champion(all_results)
    save_plot(all_results)

    return all_results


if __name__ == "__main__":
    main()
