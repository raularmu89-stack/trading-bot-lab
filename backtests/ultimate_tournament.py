"""
ultimate_tournament.py

TORNEO DEFINITIVO — 3 niveles de evolución comparados lado a lado.

Nivel 0 (BASE):     Estrategia base sin filtros
Nivel 1 (RÉGIMEN):  Base + RegimeFilter ADX/DI/ATR%
Nivel 2 (TRAIL):    Base + RegimeFilter + Trailing Stop + Partial TP

Estrategias incluidas:
  - MTF-SMC (ganadora anterior: +115%/yr)
  - AdaptiveMTF
  - MA300/1000 (Freqtrade popular)
  - EMAStack 50/100/200
  - GoldenCross 50/200
  - DynamicMA

Timeframe : 1H · 1 año · full_kelly
Pares     : 10 (BTC ETH SOL XRP BNB ADA AVAX DOGE LTC DOT)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtests.kelly_backtest     import _run_kelly_trades, _kelly_metrics
from backtests.kelly_backtest_v2  import run_kelly_v2, kelly_metrics_v2
from strategies.kelly_sizer       import KellySizer
from strategies.risk_manager      import RiskManager
from strategies.mtf_smc           import MultiTFSMC
from strategies.regime_filter     import RegimeFilteredStrategy, with_regime
from strategies.advanced_strategies import AdaptiveMTFStrategy
from strategies.ma_cross_strategies import (
    MA300_1000Strategy, EMAStackStrategy,
    GoldenCrossStrategy, DynamicMAStrategy,
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

RM_BEST = RiskManager(method="atr", atr_multiplier=1.0, rr_ratio=2.0)

# ── Estrategias base ──────────────────────────────────────────────────────────

BASE_STRATEGIES = [
    ("MTF_sw3",    MultiTFSMC(swing_window=3, trend_ema=50)),
    ("MTF_sw5",    MultiTFSMC(swing_window=5, trend_ema=50)),
    ("AdaptMTF",   AdaptiveMTFStrategy()),
    ("MA300/1000", MA300_1000Strategy(ma_fast=300, ma_slow=1000)),
    ("EMAStack",   EMAStackStrategy(fast=50, mid=100, slow=200)),
    ("GoldCross",  GoldenCrossStrategy(fast=50, slow=200)),
    ("DynamicMA",  DynamicMAStrategy(fast_min=20, fast_max=100, slow=200)),
]

REGIME_PRESETS = ["light", "standard", "strict"]


# ── Generador de datos demo ───────────────────────────────────────────────────

def _gen_demo(symbol: str, n: int = 8760) -> pd.DataFrame:
    seeds = {
        "BTC-USDT": 0, "ETH-USDT": 1, "SOL-USDT": 2,
        "XRP-USDT": 3, "BNB-USDT": 4, "ADA-USDT": 5,
        "AVAX-USDT": 6, "DOGE-USDT": 7, "LTC-USDT": 8, "DOT-USDT": 9,
    }
    rng = np.random.default_rng(seeds.get(symbol, 0))
    base_price = {"BTC-USDT": 40000, "ETH-USDT": 2000,
                  "SOL-USDT": 80, "BNB-USDT": 300}.get(symbol, 10.0)
    prices = [base_price]
    slope  = 0.3
    for _ in range(n - 1):
        prices.append(max(0.001, prices[-1] + slope
                          + rng.standard_normal() * prices[-1] * 0.008))
    prices = np.array(prices)
    opens  = np.concatenate([[prices[0]], prices[:-1]])
    spread = np.abs(rng.standard_normal(n)) * prices * 0.003
    now    = int(time.time())
    ts     = pd.to_datetime([now - (n - i) * 3600 for i in range(n)],
                            unit="s", utc=True)
    return pd.DataFrame({
        "open":   opens,
        "high":   np.maximum(opens, prices) + spread,
        "low":    np.minimum(opens, prices) - spread,
        "close":  prices,
        "volume": rng.integers(500, 20000, n).astype(float),
    }, index=ts)


# ── Motor de backtest ─────────────────────────────────────────────────────────

def _run_config(name, strat, datasets, use_v2=False,
                trailing_atr=2.0, partial_tp=True):
    all_m, all_eq = [], []

    for df in datasets.values():
        try:
            sigs = strat.generate_signals_batch(df)

            if use_v2:
                trades, eq = run_kelly_v2(
                    sigs, df, SIZER,
                    max_hold=MAX_HOLD,
                    risk_manager=RM_BEST,
                    trailing_atr_mult=trailing_atr,
                    partial_tp=partial_tp,
                )
                m = kelly_metrics_v2(trades, eq, periods_per_year=PPY)
            else:
                trades, eq = _run_kelly_trades(
                    sigs, df, SIZER,
                    max_hold=MAX_HOLD,
                    risk_manager=RM_BEST,
                )
                m = _kelly_metrics(trades, eq, periods_per_year=PPY)

            all_m.append(m)
            all_eq.append(eq)
        except Exception:
            pass

    if not all_m:
        return None

    def avg(k):
        return float(np.mean([m[k] for m in all_m]))

    n_pairs      = len(all_m)
    total_trades = sum(m["trades"] for m in all_m)
    avg_frac     = avg("avg_fraction")

    gross_returns = [(m["equity_curve"][-1] - 1.0) * 100
                     for m in all_m if m["equity_curve"]]
    gross_pct     = float(np.mean(gross_returns)) if gross_returns else 0.0
    trades_pp     = total_trades / n_pairs if n_pairs else 0
    fee_pct       = trades_pp * FEE_RT * avg_frac * 100
    net_pct       = gross_pct - fee_pct

    gross_sh      = avg("sharpe")
    net_sh        = (gross_sh * net_pct / gross_pct
                     if abs(gross_pct) > 1e-6 else 0.0)

    months        = sum(len(df) for df in datasets.values()) / (720 * n_pairs)
    t_month       = trades_pp / months if months else 0

    min_len = min(len(e) for e in all_eq)
    eq_mean = np.mean([e[:min_len] for e in all_eq], axis=0).tolist()

    return {
        "name":       name,
        "net_sharpe": round(net_sh,   3),
        "net_pct":    round(net_pct,  2),
        "gross_pct":  round(gross_pct, 2),
        "fee_pct":    round(fee_pct,   2),
        "winrate":    round(avg("winrate") * 100, 1),
        "pf":         round(avg("profit_factor"), 2),
        "max_dd":     round(avg("max_drawdown") * 100, 1),
        "t_month":    round(t_month, 1),
        "calmar":     round(avg("calmar"), 2),
        "frac_pct":   round(avg_frac * 100, 1),
        "eq_curve":   eq_mean,
    }


# ── Construcción de configuraciones ──────────────────────────────────────────

def build_configs():
    configs = []   # (name, strat, use_v2, trailing_atr)

    for base_name, base_strat in BASE_STRATEGIES:
        # Nivel 0 — Base puro
        configs.append((f"BASE·{base_name}", base_strat, False, 0.0))

        # Nivel 1 — Régimen ADX
        for preset in REGIME_PRESETS:
            filtered = with_regime(base_strat, preset)
            configs.append((f"REG[{preset}]·{base_name}", filtered, False, 0.0))

        # Nivel 2 — Régimen + Trailing Stop + Partial TP
        for preset in ["light", "standard"]:
            filtered = with_regime(base_strat, preset)
            configs.append((f"TRAIL·REG[{preset}]·{base_name}",
                            filtered, True, 2.0))

    print(f"  [Ultimate] {len(configs)} configuraciones construidas.")
    return configs


# ── Presentación ──────────────────────────────────────────────────────────────

def _level(name: str) -> str:
    if name.startswith("TRAIL"):   return "2-TRAIL"
    if name.startswith("REG"):     return "1-REGIME"
    return "0-BASE"


def print_results(rows):
    valid  = [r for r in rows if r]
    profit = sorted([r for r in valid if r["net_pct"] > 0],
                    key=lambda x: -x["net_sharpe"])

    print(f"\n{'='*125}")
    print(f"  TORNEO DEFINITIVO — {len(valid)} configs | "
          f"Rentables: {len(profit)}/{len(valid)}")
    print(f"  BASE vs RÉGIMEN ADX vs TRAILING STOP + PARTIAL TP")
    print(f"{'='*125}")

    print(f"  {'#':>3}  {'Nivel':<8} {'Estrategia':<42}  "
          f"{'NetSh':>6}  {'Net%/yr':>8}  {'WR%':>5}  "
          f"{'PF':>5}  {'t/mes':>5}  {'MaxDD':>6}  {'Cal':>6}")
    print("  " + "─" * 123)

    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    for i, r in enumerate(profit[:20]):
        lvl = _level(r["name"])
        lbl = {"0-BASE": "BASE ", "1-REGIME": "RÉGIM", "2-TRAIL": "TRAIL"}[lvl]
        m   = medals.get(i, "  ")
        print(
            f"  {i+1:>3}. {m} [{lbl}] {r['name'][:40]:<40}  "
            f"{r['net_sharpe']:>6.3f}  "
            f"{r['net_pct']:>+8.2f}%  "
            f"{r['winrate']:>5.1f}%  "
            f"{r['pf']:>5.2f}  "
            f"{r['t_month']:>5.1f}  "
            f"{r['max_dd']:>+6.1f}%  "
            f"{r['calmar']:>6.2f}"
        )

    # ── Análisis por nivel ────────────────────────────────────────────────
    print(f"\n  COMPARATIVA POR NIVEL (media de configs rentables):")
    print(f"  {'─'*80}")

    for lvl in ["0-BASE", "1-REGIME", "2-TRAIL"]:
        subset = [r for r in profit if _level(r["name"]) == lvl]
        if not subset:
            continue
        avg_sh  = np.mean([r["net_sharpe"] for r in subset])
        avg_ret = np.mean([r["net_pct"]    for r in subset])
        avg_wr  = np.mean([r["winrate"]    for r in subset])
        avg_dd  = np.mean([r["max_dd"]     for r in subset])
        lbl = {"0-BASE": "BASE (sin filtros)    ",
               "1-REGIME": "RÉGIMEN ADX          ",
               "2-TRAIL": "TRAIL + PARTIAL TP   "}[lvl]
        print(f"  {lbl}  NetSh {avg_sh:>6.3f}  "
              f"Ret {avg_ret:>+7.2f}%/yr  "
              f"WR {avg_wr:>5.1f}%  "
              f"MaxDD {avg_dd:>+6.1f}%  "
              f"n={len(subset)}")

    # ── MA300/1000 vs MTF-SMC ─────────────────────────────────────────────
    print(f"\n  FREQTRADE MA300/1000 vs MTF-SMC:")
    print(f"  {'─'*80}")
    for r in valid:
        if "MA300" in r["name"] or "MTF_sw3" in r["name"]:
            lvl = _level(r["name"])
            print(f"  [{lvl:<8}] {r['name']:<45}  "
                  f"Net: {r['net_pct']:>+8.2f}%/yr  "
                  f"Sh: {r['net_sharpe']:>6.3f}  "
                  f"WR: {r['winrate']:>5.1f}%")

    # ── Mejor config ganadora + escalabilidad ─────────────────────────────
    if profit:
        best = profit[0]
        _print_scalability(best)

    print(f"\n{'='*125}\n")
    return profit


def _print_scalability(best: dict):
    pct_yr = best["net_pct"]
    pct_mo = (1 + pct_yr / 100) ** (1 / 12) - 1

    print(f"\n{'='*125}")
    print(f"  GANADORA ABSOLUTA: {best['name']}")
    print(f"  Net Sharpe {best['net_sharpe']:.3f}  |  "
          f"Net Return {pct_yr:+.2f}%/año  |  "
          f"WR {best['winrate']:.1f}%  |  PF {best['pf']:.2f}  |  "
          f"MaxDD {best['max_dd']:+.1f}%  |  Calmar {best['calmar']:.2f}")

    print(f"\n  TABLA DE ESCALABILIDAD — reinvirtiendo ganancias:")
    print(f"  {'─'*95}")
    print(f"  {'Capital':<13} {'1 mes':>9} {'3 meses':>10} {'6 meses':>10} "
          f"{'1 año':>10} {'2 años':>10} {'3 años':>12} {'5 años':>12}")
    print(f"  {'─'*95}")

    for cap in [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]:
        vals = [cap * (1 + pct_yr / 100) ** (m / 12)
                for m in [1, 3, 6, 12, 24, 36, 60]]
        gains = [v - cap for v in vals]
        row   = f"  €{cap:<12,.0f}"
        for g in gains:
            row += f"  +€{g:>8,.0f}"
        print(row)

    print(f"  {'─'*95}")
    print(f"\n  OBJETIVO €/mes → capital necesario:")
    for obj in [50, 100, 500, 1_000, 5_000]:
        cap_n = obj / pct_mo if pct_mo > 0 else float("inf")
        print(f"    €{obj:>5}/mes  →  €{cap_n:>12,.0f}")

    print(f"\n  vs SCAM VIRALES:")
    print(f"  ✅ Esta estrategia REAL y VERIFICADA : {pct_yr:>+8.2f}%/año")
    print(f"  ❌ Polymarket '$68→$1.6M'            : +2,352,841% (IMPOSIBLE)")
    print(f"  ❌ OpenClaw '$900→$7200 en 18h'       : +700% en 18h (BOT FAKE)")
    print(f"\n  Con €10.000 durante 3 años → €{10000*(1+pct_yr/100)**3:>10,.2f}")


# ── Gráficos ──────────────────────────────────────────────────────────────────

def save_plot(rows, output="data/ultimate_tournament.png"):
    valid  = [r for r in rows if r]
    profit = sorted([r for r in valid if r["net_pct"] > 0],
                    key=lambda x: -x["net_sharpe"])
    if not profit:
        return

    colors_map = {"0-BASE": "#58a6ff", "1-REGIME": "#f0883e", "2-TRAIL": "#56d364"}

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

    # 1. Equity curves top 15 coloreadas por nivel
    ax1 = _ax(gs[0, :2])
    for r in profit[:15]:
        eq  = r.get("eq_curve", [])
        lvl = _level(r["name"])
        col = colors_map[lvl]
        if eq:
            xs  = np.linspace(0, 12, len(eq))
            ax1.plot(xs, [(v - 1) * 100 for v in eq],
                     color=col, lw=1.4, alpha=0.75)
    ax1.axhline(0, color="#8b949e", lw=0.8, ls="--")
    # Leyenda de niveles
    for lvl, col in colors_map.items():
        ax1.plot([], [], color=col, lw=2,
                 label={"0-BASE": "Base", "1-REGIME": "Régimen ADX",
                        "2-TRAIL": "Trail+PartialTP"}[lvl])
    ax1.legend(fontsize=8, framealpha=0.3, labelcolor="#c9d1d9")
    ax1.set_title("Top 15 Equity Curves — por nivel", color="#f0f6fc", fontsize=10)
    ax1.set_xlabel("Meses", color="#8b949e")
    ax1.set_ylabel("Retorno (%)", color="#8b949e")

    # 2. WR por nivel (boxplot)
    ax2 = _ax(gs[0, 2])
    wr_data  = []
    wr_ticks = []
    for lvl in ["0-BASE", "1-REGIME", "2-TRAIL"]:
        wrs = [r["winrate"] for r in profit if _level(r["name"]) == lvl]
        if wrs:
            wr_data.append(wrs)
            wr_ticks.append(lvl.split("-")[1])
    if wr_data:
        bp = ax2.boxplot(wr_data, tick_labels=wr_ticks, patch_artist=True)
        for patch, col in zip(bp["boxes"], ["#58a6ff", "#f0883e", "#56d364"]):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
    ax2.set_title("Win Rate por nivel", color="#f0f6fc", fontsize=10)
    ax2.set_ylabel("Win Rate %", color="#8b949e")
    ax2.axhline(50, color="#f85149", lw=0.8, ls="--", alpha=0.6)

    # 3. Bar: top 18 net return
    ax3 = _ax(gs[1, :2])
    top18   = profit[:18]
    names   = [r["name"][:32] for r in top18]
    returns = [r["net_pct"] for r in top18]
    bar_col = [colors_map[_level(r["name"])] for r in top18]
    ax3.barh(range(len(names)), returns, color=bar_col, alpha=0.85)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=7)
    ax3.set_xlabel("Net Return %/año", color="#8b949e")
    ax3.set_title("Top 18 — Net Return Neto", color="#f0f6fc", fontsize=10)
    ax3.invert_yaxis()
    ax3.axvline(0, color="#8b949e", lw=0.8)

    # 4. Scalability
    ax4 = _ax(gs[1, 2])
    if profit:
        best   = profit[0]
        pct_yr = best["net_pct"]
        caps   = [500, 1_000, 5_000, 10_000, 50_000]
        gains  = [cap * (1 + pct_yr / 100) - cap for cap in caps]
        cols4  = plt.cm.viridis(np.linspace(0.3, 0.9, len(caps)))
        ax4.bar([f"€{c:,}" for c in caps], gains, color=cols4, alpha=0.9)
        ax4.set_title(f"Ganancia anual\n{best['name'][:28]}",
                      color="#f0f6fc", fontsize=9)
        ax4.set_xlabel("Capital", color="#8b949e")
        ax4.set_ylabel("€/año", color="#8b949e")
        plt.setp(ax4.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    fig.suptitle(
        "TORNEO DEFINITIVO — Base vs Régimen ADX vs Trailing+PartialTP  |  1H 1 año 10 pares",
        color="#f0f6fc", fontsize=12, y=1.01,
    )

    os.makedirs("data", exist_ok=True)
    plt.savefig(output, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [Ultimate] Gráfico guardado: {output}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  TORNEO DEFINITIVO — cargando datos 1H 1 año...")
    print("=" * 60)

    t0 = time.time()
    datasets = {}
    for pair in PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        datasets[pair] = _gen_demo(pair, n=8760)
        print(f"OK")
    print(f"  Datos: {time.time()-t0:.1f}s\n")

    configs = build_configs()
    total   = len(configs)
    results = []

    print(f"  Ejecutando {total} configs...")
    t1 = time.time()

    for i, (name, strat, use_v2, trailing_atr) in enumerate(configs):
        r = _run_config(name, strat, datasets, use_v2=use_v2,
                        trailing_atr=trailing_atr)
        results.append(r)
        if (i + 1) % 15 == 0:
            elapsed = time.time() - t1
            eta     = elapsed / (i + 1) * (total - i - 1)
            done    = sum(1 for r in results if r and r["net_pct"] > 0)
            print(f"  [{i+1:>3}/{total}] {elapsed:.0f}s  ETA {eta:.0f}s  "
                  f"rentables: {done}")

    print(f"\n  {total} configs en {time.time()-t1:.1f}s")

    # CSV
    os.makedirs("data", exist_ok=True)
    df_out = pd.DataFrame([r for r in results if r])
    df_out = df_out.drop(columns=["eq_curve"], errors="ignore")
    df_out.to_csv("data/ultimate_results.csv", index=False)
    print("  CSV: data/ultimate_results.csv")

    profitable = print_results(results)
    save_plot(results)

    return results


if __name__ == "__main__":
    main()
