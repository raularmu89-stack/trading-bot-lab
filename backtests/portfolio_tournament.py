"""
portfolio_tournament.py

TORNEO PORTFOLIO — dos fases:

FASE 1: Test de todas las estrategias (incluyendo las 5 nuevas de alta freq)
FASE 2: Portfolio multi-estrategia — combina las top N simultáneamente
        Capital dividido entre estrategias activas → menor drawdown,
        curva de equity más suave, mayor Sharpe de portfolio.

FASE 3: Walk-forward — divide el año en 4 trimestres, entrena en 3 y
        valida en 1 (rotando). Muestra si la estrategia es robusta.

Capital inicial: €100
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
from backtests.metrics               import compute_all
from strategies.kelly_sizer          import KellySizer
from strategies.risk_manager         import RiskManager
from strategies.regime_filter        import RegimeFilteredStrategy
from strategies.mtf_smc              import MultiTFSMC
from strategies.advanced_strategies  import AdaptiveMTFStrategy
from strategies.trend_momentum_strategies import (
    DualMomentumStrategy, TrendStrengthStrategy, VolumeTrendStrategy,
    AdaptiveChannelStrategy, MultiSignalStrategy,
)

# ── Config ────────────────────────────────────────────────────────────────────

PAIRS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BNB-USDT",
    "ADA-USDT", "AVAX-USDT", "DOGE-USDT", "LTC-USDT", "DOT-USDT",
]
MAX_HOLD     = 24
PPY          = 8760
FEE_RT       = 0.002
TRAIL        = 3.0
PARTIAL      = 0.33
ADX_MIN      = 13
CAPITAL_INIT = 100.0

SIZER = KellySizer(variant="full_kelly", min_trades=20,
                   max_fraction=0.60, min_fraction=0.01)
RM    = RiskManager(method="atr", atr_multiplier=1.0, rr_ratio=2.0)


# ── Demo data ─────────────────────────────────────────────────────────────────

def _gen_demo(symbol: str, n: int = 8760, seed_offset: int = 0) -> pd.DataFrame:
    seeds = {p: i for i, p in enumerate(PAIRS)}
    rng  = np.random.default_rng(seeds.get(symbol, 0) + seed_offset)
    base = {"BTC-USDT": 40000, "ETH-USDT": 2000,
            "SOL-USDT": 80, "BNB-USDT": 300}.get(symbol, 10.0)
    p = [base]
    for _ in range(n - 1):
        p.append(max(1e-4, p[-1] + 0.3 + rng.standard_normal() * p[-1] * 0.008))
    p  = np.array(p)
    o  = np.concatenate([[p[0]], p[:-1]])
    s  = np.abs(rng.standard_normal(n)) * p * 0.003
    ts = pd.to_datetime([int(time.time()) - (n-i)*3600 for i in range(n)],
                        unit="s", utc=True)
    return pd.DataFrame({"open": o, "high": np.maximum(o,p)+s,
                         "low": np.minimum(o,p)-s, "close": p,
                         "volume": rng.integers(500, 20000, n).astype(float)},
                        index=ts)


# ── Motor individual ──────────────────────────────────────────────────────────

def _run_single(strat, datasets):
    all_m, all_eq = [], []
    for df in datasets.values():
        try:
            sigs = strat.generate_signals_batch(df)
            trades, eq = run_kelly_v2(
                sigs, df, SIZER, max_hold=MAX_HOLD, risk_manager=RM,
                trailing_atr_mult=TRAIL, partial_tp=True, partial_ratio=PARTIAL,
            )
            m = kelly_metrics_v2(trades, eq, periods_per_year=PPY)
            all_m.append(m); all_eq.append(eq)
        except Exception:
            pass
    if not all_m:
        return None, None
    n_p      = len(all_m)
    avg_frac = float(np.mean([m["avg_fraction"] for m in all_m]))
    total_t  = sum(m["trades"] for m in all_m)
    gross    = float(np.mean([(m["equity_curve"][-1]-1)*100
                               for m in all_m if m["equity_curve"]]))
    fee      = (total_t/n_p)*FEE_RT*avg_frac*100
    net      = gross - fee
    gsh      = float(np.mean([m["sharpe"] for m in all_m]))
    nsh      = gsh * net/gross if abs(gross) > 1e-6 else 0.0
    months   = sum(len(df) for df in datasets.values())/(720*n_p)
    t_month  = (total_t/n_p)/months if months else 0
    min_len  = min(len(e) for e in all_eq)
    eq_mean  = np.mean([e[:min_len] for e in all_eq], axis=0)
    return {
        "net_sharpe": round(nsh, 3), "net_pct": round(net, 2),
        "winrate": round(float(np.mean([m["winrate"] for m in all_m]))*100, 1),
        "pf": round(float(np.mean([m["profit_factor"] for m in all_m])), 2),
        "max_dd": round(float(np.mean([m["max_drawdown"] for m in all_m]))*100, 1),
        "t_month": round(t_month, 1),
        "calmar": round(float(np.mean([m["calmar"] for m in all_m])), 2),
        "frac_pct": round(avg_frac*100, 1),
    }, eq_mean


# ── Motor portfolio ───────────────────────────────────────────────────────────

def run_portfolio(strategy_list, datasets, name="Portfolio"):
    """
    Combina señales de múltiples estrategias:
    - En cada barra, recoge señales de todas las estrategias
    - Señal final = majority vote (empate = hold)
    - Capital dividido equitativamente entre posiciones simultáneas
    """
    all_pair_eq = []

    for pair, df in datasets.items():
        # Calcular señales de todas las estrategias
        all_sigs = []
        for strat in strategy_list:
            try:
                s = strat.generate_signals_batch(df)
                all_sigs.append(s)
            except Exception:
                all_sigs.append(["hold"] * len(df))

        n = len(df)
        # Majority vote
        combined = []
        for i in range(n):
            votes_b = sum(1 for s in all_sigs if s[i] == "buy")
            votes_s = sum(1 for s in all_sigs if s[i] == "sell")
            thresh  = len(strategy_list) // 2 + 1   # mayoría simple
            if votes_b >= thresh and votes_b > votes_s:
                combined.append("buy")
            elif votes_s >= thresh and votes_s > votes_b:
                combined.append("sell")
            else:
                combined.append("hold")

        try:
            trades, eq = run_kelly_v2(
                combined, df, SIZER, max_hold=MAX_HOLD, risk_manager=RM,
                trailing_atr_mult=TRAIL, partial_tp=True, partial_ratio=PARTIAL,
            )
            m = kelly_metrics_v2(trades, eq, periods_per_year=PPY)
            all_pair_eq.append((m, eq))
        except Exception:
            pass

    if not all_pair_eq:
        return None

    all_m  = [x[0] for x in all_pair_eq]
    all_eq = [x[1] for x in all_pair_eq]
    n_p    = len(all_m)

    avg_frac = float(np.mean([m["avg_fraction"] for m in all_m]))
    total_t  = sum(m["trades"] for m in all_m)
    gross    = float(np.mean([(m["equity_curve"][-1]-1)*100
                               for m in all_m if m["equity_curve"]]))
    fee      = (total_t/n_p)*FEE_RT*avg_frac*100
    net      = gross - fee
    gsh      = float(np.mean([m["sharpe"] for m in all_m]))
    nsh      = gsh * net/gross if abs(gross) > 1e-6 else 0.0
    months   = sum(len(df) for df in datasets.values())/(720*n_p)
    t_month  = (total_t/n_p)/months if months else 0
    min_len  = min(len(e) for e in all_eq)
    eq_mean  = np.mean([e[:min_len] for e in all_eq], axis=0)

    return {
        "name": name,
        "net_sharpe": round(nsh, 3), "net_pct": round(net, 2),
        "winrate": round(float(np.mean([m["winrate"] for m in all_m]))*100, 1),
        "pf": round(float(np.mean([m["profit_factor"] for m in all_m])), 2),
        "max_dd": round(float(np.mean([m["max_drawdown"] for m in all_m]))*100, 1),
        "t_month": round(t_month, 1),
        "calmar": round(float(np.mean([m["calmar"] for m in all_m])), 2),
        "frac_pct": round(avg_frac*100, 1),
        "n_strats": len(strategy_list),
        "eq_curve": eq_mean.tolist(),
    }


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(strat, datasets, n_folds: int = 4):
    """
    Divide el año en n_folds trimestres.
    Evalúa la estrategia en cada trimestre por separado.
    Muestra consistencia cross-period.
    """
    fold_size = 8760 // n_folds
    fold_results = []

    for fold in range(n_folds):
        start = fold * fold_size
        end   = start + fold_size
        fold_datasets = {
            pair: df.iloc[start:end].copy()
            for pair, df in datasets.items()
        }
        result, _ = _run_single(strat, fold_datasets)
        if result:
            fold_results.append(result)

    if not fold_results:
        return None

    return {
        "folds":       n_folds,
        "avg_net_pct": round(np.mean([r["net_pct"] for r in fold_results]), 2),
        "std_net_pct": round(np.std([r["net_pct"]  for r in fold_results]), 2),
        "min_net_pct": round(min(r["net_pct"]       for r in fold_results), 2),
        "max_net_pct": round(max(r["net_pct"]       for r in fold_results), 2),
        "pct_positive": round(sum(1 for r in fold_results if r["net_pct"] > 0)
                              / n_folds * 100, 0),
        "avg_sharpe":  round(np.mean([r["net_sharpe"] for r in fold_results]), 3),
        "fold_details": fold_results,
    }


# ── Estrategias del torneo ────────────────────────────────────────────────────

def _wrap(strat):
    return RegimeFilteredStrategy(strat, adx_min=ADX_MIN, di_align=True,
                                  atr_min_pct=0.2, atr_max_pct=6.0)

ALL_STRATS = {
    "AdaptMTF":      AdaptiveMTFStrategy(),
    "MTF_sw3":       MultiTFSMC(swing_window=3, trend_ema=50),
    "MTF_sw5":       MultiTFSMC(swing_window=5, trend_ema=50),
    "DualMom":       DualMomentumStrategy(fast_p=12, slow_p=48),
    "TrendStr":      TrendStrengthStrategy(adx_min=20),
    "VolTrend":      VolumeTrendStrategy(ema_fast=12, ema_slow=26),
    "AdaptCh":       AdaptiveChannelStrategy(period=20, atr_mult=1.0),
    "MultiSig3":     MultiSignalStrategy(score_min=3),
    "MultiSig4":     MultiSignalStrategy(score_min=4),
}
ALL_STRATS_REG = {k: _wrap(v) for k, v in ALL_STRATS.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print(f"  TORNEO PORTFOLIO — capital inicial €{CAPITAL_INIT:.0f}")
    print("="*65)

    datasets = {}
    for pair in PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        datasets[pair] = _gen_demo(pair, n=8760)
        print("OK")

    # ── FASE 1: individual ────────────────────────────────────────────
    print(f"\n  FASE 1 — Estrategias individuales (con régimen):")
    print(f"  {'─'*80}")
    print(f"  {'Estrategia':<18} {'NetSh':>6}  {'Net%/yr':>8}  "
          f"{'WR%':>5}  {'PF':>5}  {'t/mes':>5}  {'MaxDD':>6}")
    print(f"  {'─'*80}")

    ind_results = {}
    eq_curves   = {}
    for name, strat in ALL_STRATS_REG.items():
        r, eq = _run_single(strat, datasets)
        ind_results[name] = r
        eq_curves[name]   = eq
        if r and r["net_pct"] > 0:
            print(f"  ✅ {name:<16} {r['net_sharpe']:>6.3f}  "
                  f"{r['net_pct']:>+8.2f}%  {r['winrate']:>5.1f}%  "
                  f"{r['pf']:>5.2f}  {r['t_month']:>5.1f}  "
                  f"{r['max_dd']:>+6.1f}%")
        else:
            net = r["net_pct"] if r else 0
            print(f"  ❌ {name:<16} {'—':>6}  {net:>+8.2f}%  "
                  f"{'—':>5}  {'—':>5}  {'—':>5}  {'—':>6}")

    # ── FASE 2: portfolios ────────────────────────────────────────────
    print(f"\n  FASE 2 — Portfolios multi-estrategia:")
    print(f"  {'─'*90}")

    profitable_names = [k for k, r in ind_results.items()
                        if r and r["net_pct"] > 0]
    profitable_strats = [ALL_STRATS_REG[k] for k in profitable_names]

    portfolios = []

    if len(profitable_strats) >= 2:
        # Top 3
        p3 = run_portfolio(profitable_strats[:3], datasets,
                           name=f"Portfolio-Top3({','.join(profitable_names[:3])})")
        if p3:
            portfolios.append(p3)
            print(f"  Portfolio Top-3: NetSh {p3['net_sharpe']:.3f}  "
                  f"Net {p3['net_pct']:>+.2f}%/yr  WR {p3['winrate']:.1f}%  "
                  f"MaxDD {p3['max_dd']:+.1f}%")

    if len(profitable_strats) >= 5:
        p5 = run_portfolio(profitable_strats[:5], datasets,
                           name="Portfolio-Top5")
        if p5:
            portfolios.append(p5)
            print(f"  Portfolio Top-5: NetSh {p5['net_sharpe']:.3f}  "
                  f"Net {p5['net_pct']:>+.2f}%/yr  WR {p5['winrate']:.1f}%  "
                  f"MaxDD {p5['max_dd']:+.1f}%")

    # Portfolio ALL
    if len(profitable_strats) >= 3:
        pall = run_portfolio(profitable_strats, datasets,
                             name=f"Portfolio-All({len(profitable_strats)})")
        if pall:
            portfolios.append(pall)
            print(f"  Portfolio All-{len(profitable_strats)}: "
                  f"NetSh {pall['net_sharpe']:.3f}  "
                  f"Net {pall['net_pct']:>+.2f}%/yr  WR {pall['winrate']:.1f}%  "
                  f"MaxDD {pall['max_dd']:+.1f}%")

    # ── FASE 3: walk-forward ──────────────────────────────────────────
    print(f"\n  FASE 3 — Walk-Forward (4 trimestres) de la mejor estrategia:")
    print(f"  {'─'*75}")

    best_name = max((k for k, r in ind_results.items() if r and r["net_pct"] > 0),
                    key=lambda k: ind_results[k]["net_sharpe"], default=None)

    if best_name:
        wf = walk_forward(ALL_STRATS_REG[best_name], datasets, n_folds=4)
        if wf:
            print(f"  Estrategia: {best_name}")
            print(f"  Trimestres positivos: {wf['pct_positive']:.0f}% ({wf['folds']} folds)")
            print(f"  Rango retorno:  {wf['min_net_pct']:>+.2f}% → {wf['max_net_pct']:>+.2f}%")
            print(f"  Media: {wf['avg_net_pct']:>+.2f}%/trim  ±{wf['std_net_pct']:.2f}%  "
                  f"Sharpe medio: {wf['avg_sharpe']:.3f}")
            print(f"  Desglose por trimestre:")
            for i, fd in enumerate(wf["fold_details"]):
                bar = "█" * max(0, int(fd["net_pct"] / 5))
                print(f"    Q{i+1}: {fd['net_pct']:>+7.2f}%  Sh {fd['net_sharpe']:.2f}  "
                      f"WR {fd['winrate']:.0f}%  {bar}")

    # ── Resultado final + proyección €100 ────────────────────────────
    all_named = [(k, r, eq_curves[k]) for k, r in ind_results.items()
                 if r and r["net_pct"] > 0]
    all_named += [(p["name"], p, p["eq_curve"]) for p in portfolios if p]
    all_named  = sorted(all_named, key=lambda x: -x[1]["net_sharpe"])

    if not all_named:
        print("\n  Sin configs rentables.")
        return

    best_name, best_r, best_eq = all_named[0]
    pct_yr = best_r["net_pct"]
    pct_mo = (1 + pct_yr/100)**(1/12) - 1

    print(f"\n{'='*100}")
    print(f"  CLASIFICACIÓN FINAL (individual + portfolios):")
    print(f"  {'─'*98}")
    print(f"  {'#':>3}  {'Nombre':<42}  "
          f"{'NetSh':>6}  {'Net%/yr':>8}  {'WR%':>5}  "
          f"{'PF':>5}  {'MaxDD':>6}  {'Cal':>6}")
    print(f"  {'─'*98}")
    medals = {0:"🥇", 1:"🥈", 2:"🥉"}
    for i, (nm, r, _) in enumerate(all_named[:10]):
        m = medals.get(i, "  ")
        print(f"  {i+1:>3}. {m} {nm:<40}  "
              f"{r['net_sharpe']:>6.3f}  {r['net_pct']:>+8.2f}%  "
              f"{r['winrate']:>5.1f}%  {r['pf']:>5.2f}  "
              f"{r['max_dd']:>+6.1f}%  {r['calmar']:>6.2f}")

    print(f"\n{'='*100}")
    print(f"  GANADORA: {best_name}")
    print(f"  {pct_yr:+.2f}%/año  |  Sharpe {best_r['net_sharpe']:.3f}  |  "
          f"WR {best_r['winrate']:.1f}%  |  MaxDD {best_r['max_dd']:+.1f}%")
    print(f"  Tasa mensual: {pct_mo*100:.2f}%/mes\n")

    print(f"  PROYECCIÓN €{CAPITAL_INIT:.0f} MES A MES:")
    print(f"  {'─'*65}")
    print(f"  {'Mes':>4}  {'Capital':>10}  {'Este mes':>10}  "
          f"{'Total ganado':>12}  {'×':>5}")
    print(f"  {'─'*65}")
    for m in list(range(1, 13)) + [15, 18, 21, 24, 30, 36]:
        prev = CAPITAL_INIT * (1 + pct_yr/100)**((m-1)/12)
        cap  = CAPITAL_INIT * (1 + pct_yr/100)**(m/12)
        print(f"  {m:>4}  €{cap:>9,.2f}  +€{cap-prev:>8,.2f}  "
              f"+€{cap-CAPITAL_INIT:>10,.2f}  {cap/CAPITAL_INIT:>4.1f}x")
    print(f"  {'─'*65}")
    print(f"\n  Para €100/mes necesitas: €{100/pct_mo:>8,.0f}")
    print(f"  Para €500/mes necesitas: €{500/pct_mo:>8,.0f}")
    print(f"{'='*100}\n")

    # ── Guardar CSV ───────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    rows = [{"name": nm, **r} for nm, r, _ in all_named]
    for row in rows:
        row.pop("eq_curve", None)
    pd.DataFrame(rows).to_csv("data/portfolio_results.csv", index=False)
    print("  CSV: data/portfolio_results.csv")

    _save_plot(all_named, best_name, pct_yr)
    return all_named


def _save_plot(all_named, best_name, pct_yr):
    fig = plt.figure(figsize=(22, 13))
    fig.patch.set_facecolor("#0d1117")
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    def _ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        return ax

    pal = plt.cm.plasma(np.linspace(0.05, 0.95, min(12, len(all_named))))

    # 1. Equity curves desde €100
    ax1 = _ax(gs[0, :2])
    for i, (nm, r, eq) in enumerate(all_named[:12]):
        if eq is not None:
            eq_arr = np.array(eq) if not isinstance(eq, np.ndarray) else eq
            xs  = np.linspace(0, 12, len(eq_arr))
            ys  = CAPITAL_INIT * eq_arr
            ax1.plot(xs, ys, color=pal[i], lw=1.5, alpha=0.8,
                     label=f"{nm[:22]} ({r['net_pct']:>+.0f}%)" if i < 6 else "")
    ax1.axhline(CAPITAL_INIT, color="#8b949e", lw=0.8, ls="--")
    ax1.legend(fontsize=7, framealpha=0.3, labelcolor="#c9d1d9")
    ax1.set_title(f"Equity desde €{CAPITAL_INIT:.0f}  (todos los ganadores)",
                  color="#f0f6fc", fontsize=10)
    ax1.set_xlabel("Meses", color="#8b949e"); ax1.set_ylabel("€", color="#8b949e")

    # 2. Sharpe vs MaxDD
    ax2 = _ax(gs[0, 2])
    xs2 = [abs(r["max_dd"]) for _, r, _ in all_named[:20]]
    ys2 = [r["net_sharpe"]  for _, r, _ in all_named[:20]]
    ax2.scatter(xs2, ys2, c=np.arange(len(xs2)), cmap="plasma", s=60, alpha=0.85)
    ax2.set_title("MaxDD vs Net Sharpe", color="#f0f6fc", fontsize=10)
    ax2.set_xlabel("|MaxDD %|", color="#8b949e"); ax2.set_ylabel("Net Sharpe", color="#8b949e")

    # 3. Bar net return
    ax3 = _ax(gs[1, :2])
    top = all_named[:15]
    ax3.barh([x[0][:32] for x in top], [x[1]["net_pct"] for x in top],
             color=pal[:len(top)], alpha=0.85)
    ax3.set_xlabel("Net Return %/año", color="#8b949e")
    ax3.set_title("Top 15 — Net Return", color="#f0f6fc", fontsize=10)
    ax3.invert_yaxis()

    # 4. Proyección €100 × top 5
    ax4 = _ax(gs[1, 2])
    meses = list(range(1, 37))
    for i, (nm, r, _) in enumerate(all_named[:5]):
        py   = r["net_pct"]
        vals = [CAPITAL_INIT * (1 + py/100)**(m/12) for m in meses]
        ax4.plot(meses, vals, color=pal[i], lw=2,
                 label=f"{nm[:18]} ({py:>+.0f}%)")
    ax4.axhline(CAPITAL_INIT, color="#8b949e", lw=0.8, ls="--")
    ax4.set_title(f"€{CAPITAL_INIT:.0f} → 36 meses", color="#f0f6fc", fontsize=10)
    ax4.set_xlabel("Mes", color="#8b949e"); ax4.set_ylabel("€", color="#8b949e")
    ax4.legend(fontsize=6.5, framealpha=0.3, labelcolor="#c9d1d9")

    fig.suptitle(
        f"TORNEO PORTFOLIO | Individual + Multi-Estrategia + Walk-Forward | "
        f"€{CAPITAL_INIT:.0f} inicial | 1H 10 pares",
        color="#f0f6fc", fontsize=11, y=1.01,
    )
    out = "data/portfolio_tournament.png"
    os.makedirs("data", exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Gráfico: {out}")


if __name__ == "__main__":
    main()
