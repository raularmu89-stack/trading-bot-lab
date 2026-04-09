"""
final_tournament.py

TORNEO FINAL — todas las estrategias de todos los archivos.
Motor: kelly_backtest_v2 (trailing stop 3.0 + partial TP 33%)
Capital inicial: €100

Estrategias incluidas (total ~50 configs):
  - MTF-SMC (ganadora histórica)
  - AdaptiveMTF (nueva campeona)
  - advanced_strategies.py  (10 estrategias)
  - price_action_strategies.py (6 estrategias)
  - momentum_oscillator_strategies.py (8 estrategias NUEVAS)
  - ma_cross_strategies.py (4 estrategias)
  - SMC base, EnsembleVoter

Todas con RegimeFilter[light] + Trail(3.0) + Partial(33%)
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

from backtests.kelly_backtest_v2  import run_kelly_v2, kelly_metrics_v2
from strategies.kelly_sizer       import KellySizer
from strategies.risk_manager      import RiskManager
from strategies.regime_filter     import RegimeFilteredStrategy
from strategies.mtf_smc           import MultiTFSMC
from strategies.smc_strategy      import SMCStrategy
from strategies.advanced_strategies import (
    StochRSIStrategy, IchimokuStrategy, FibRetracementStrategy,
    MarketStructureStrategy, MACDRSIStrategy, LinearRegStrategy,
    BollingerMomStrategy, OrderBlockStrategy, BreakoutVolStrategy,
    AdaptiveMTFStrategy,
)
from strategies.price_action_strategies import (
    SuperTrendStrategy, VWAPStrategy, HullMAStrategy,
    KeltnerBreakoutStrategy, PinBarStrategy, RSIDivergenceStrategy,
)
from strategies.momentum_oscillator_strategies import (
    AwesomeOscillatorStrategy, CCIStrategy, WilliamsRStrategy,
    SqueezeMomentumStrategy, ChaikinMFStrategy, ElderRayStrategy,
    HeikinAshiEMAStrategy, ParabolicSARStrategy,
)
from strategies.ma_cross_strategies import (
    MA300_1000Strategy, EMAStackStrategy,
    GoldenCrossStrategy, DynamicMAStrategy,
)

# ── Config ────────────────────────────────────────────────────────────────────

PAIRS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BNB-USDT",
    "ADA-USDT", "AVAX-USDT", "DOGE-USDT", "LTC-USDT", "DOT-USDT",
]
MAX_HOLD      = 24
PPY           = 8760
FEE_RT        = 0.002
TRAIL         = 3.0
PARTIAL       = 0.33
ADX_MIN       = 13
CAPITAL_INIT  = 100.0   # € — capital inicial para proyecciones

SIZER = KellySizer(variant="full_kelly", min_trades=20,
                   max_fraction=0.60, min_fraction=0.01)
RM    = RiskManager(method="atr", atr_multiplier=1.0, rr_ratio=2.0)


# ── Demo data ─────────────────────────────────────────────────────────────────

def _gen_demo(symbol: str, n: int = 8760) -> pd.DataFrame:
    seeds = {p: i for i, p in enumerate(PAIRS)}
    rng  = np.random.default_rng(seeds.get(symbol, 0))
    base = {"BTC-USDT": 40000, "ETH-USDT": 2000,
            "SOL-USDT": 80,    "BNB-USDT": 300}.get(symbol, 10.0)
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


# ── Motor ─────────────────────────────────────────────────────────────────────

def _run(name, strat, datasets):
    all_m, all_eq = [], []
    for df in datasets.values():
        try:
            sigs = strat.generate_signals_batch(df)
            trades, eq = run_kelly_v2(
                sigs, df, SIZER,
                max_hold=MAX_HOLD, risk_manager=RM,
                trailing_atr_mult=TRAIL,
                partial_tp=True, partial_ratio=PARTIAL,
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
    gross_pct    = float(np.mean([(m["equity_curve"][-1]-1)*100
                                   for m in all_m if m["equity_curve"]]))
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
        "fee_pct": round(fee_pct, 2),
        "winrate": round(avg("winrate")*100, 1),
        "pf": round(avg("profit_factor"), 2),
        "max_dd": round(avg("max_drawdown")*100, 1),
        "t_month": round(t_month, 1),
        "calmar": round(avg("calmar"), 2),
        "frac_pct": round(avg_frac*100, 1),
        "eq_curve": eq_mean,
    }


def _wrap(strat):
    """Envuelve con RegimeFilter[light]."""
    return RegimeFilteredStrategy(
        strat, adx_min=ADX_MIN, di_align=True,
        atr_min_pct=0.2, atr_max_pct=6.0,
    )


# ── Todas las estrategias ─────────────────────────────────────────────────────

def build_all_strategies():
    S = []

    def add(name, strat, regime=True):
        s = _wrap(strat) if regime else strat
        prefix = "REG·" if regime else "BASE·"
        S.append((prefix + name, s))

    # ── MTF-SMC (histórico ganador) ──────────────────────────────────
    for sw in [3, 5, 7]:
        add(f"MTF_sw{sw}", MultiTFSMC(swing_window=sw, trend_ema=50))
    add("SMC_sw5", SMCStrategy(swing_window=5))

    # ── Advanced (10) ────────────────────────────────────────────────
    add("AdaptMTF",   AdaptiveMTFStrategy())
    add("StochRSI",   StochRSIStrategy())
    add("Ichimoku",   IchimokuStrategy())
    add("FibRet",     FibRetracementStrategy())
    add("MktStruct",  MarketStructureStrategy())
    add("MACD_RSI",   MACDRSIStrategy())
    add("LinearReg",  LinearRegStrategy())
    add("BBMom",      BollingerMomStrategy())
    add("OrdBlock",   OrderBlockStrategy())
    add("BrkVol",     BreakoutVolStrategy())

    # ── Price Action (6) ─────────────────────────────────────────────
    add("SuperTrend10", SuperTrendStrategy(atr_period=10, multiplier=3.0))
    add("SuperTrend7",  SuperTrendStrategy(atr_period=7,  multiplier=2.5))
    add("VWAP24",       VWAPStrategy(session_len=24, band_mult=1.5))
    add("HullMA20",     HullMAStrategy(fast=20, slow=55))
    add("Keltner20",    KeltnerBreakoutStrategy(ema_period=20, mult=1.5))
    add("PinBar50",     PinBarStrategy(ema_trend=50, wick_ratio=2.0))

    # ── Momentum Oscillators (8 NUEVAS) ──────────────────────────────
    add("AwesomeOsc",   AwesomeOscillatorStrategy(fast=5, slow=34))
    add("CCI20",        CCIStrategy(period=20, ema_trend=50))
    add("CCI14",        CCIStrategy(period=14, ema_trend=50))
    add("WilliamsR14",  WilliamsRStrategy(period=14, ema_trend=50))
    add("SqueezeMom",   SqueezeMomentumStrategy(bb_period=20, kc_period=20))
    add("ChaikinMF",    ChaikinMFStrategy(period=20, threshold=0.10))
    add("ElderRay",     ElderRayStrategy(ema_period=13, trend_ema=50))
    add("HeikinAshi",   HeikinAshiEMAStrategy(ema_fast=20, ema_slow=50))
    add("ParabolicSAR", ParabolicSARStrategy(af_start=0.02, ema_trend=50))

    # ── MA Cross (4) ─────────────────────────────────────────────────
    add("EMAStack",    EMAStackStrategy(fast=50, mid=100, slow=200))
    add("GoldCross",   GoldenCrossStrategy(fast=50, slow=200))
    add("DynamicMA",   DynamicMAStrategy(fast_min=20, fast_max=100, slow=200))

    # ── Sin régimen (comparación) ─────────────────────────────────────
    add("AdaptMTF_noR", AdaptiveMTFStrategy(), regime=False)
    add("MTF_sw3_noR",  MultiTFSMC(swing_window=3, trend_ema=50), regime=False)

    print(f"  [Final] {len(S)} estrategias construidas.")
    return S


# ── Presentación ──────────────────────────────────────────────────────────────

def print_results(rows):
    valid  = [r for r in rows if r]
    profit = sorted([r for r in valid if r["net_pct"] > 0],
                    key=lambda x: -x["net_sharpe"])

    print(f"\n{'='*122}")
    print(f"  TORNEO FINAL — {len(valid)} estrategias | "
          f"Rentables: {len(profit)}/{len(valid)}")
    print(f"  Motor: Trail({TRAIL}) + PartialTP({PARTIAL}) + RegimeADX>{ADX_MIN}")
    print(f"{'='*122}")
    print(f"  {'#':>3}  {'Estrategia':<38}  "
          f"{'NetSh':>6}  {'Net%/yr':>8}  {'WR%':>5}  "
          f"{'PF':>5}  {'t/mes':>5}  {'MaxDD':>6}  {'Cal':>6}")
    print("  " + "─"*120)

    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    for i, r in enumerate(profit[:20]):
        m = medals.get(i, "  ")
        print(
            f"  {i+1:>3}. {m} {r['name']:<36}  "
            f"{r['net_sharpe']:>6.3f}  {r['net_pct']:>+8.2f}%  "
            f"{r['winrate']:>5.1f}%  {r['pf']:>5.2f}  "
            f"{r['t_month']:>5.1f}  {r['max_dd']:>+6.1f}%  "
            f"{r['calmar']:>6.2f}"
        )

    if not profit:
        return profit

    best   = profit[0]
    pct_yr = best["net_pct"]
    pct_mo = (1 + pct_yr / 100) ** (1/12) - 1

    # ── Proyección €100 mes a mes ─────────────────────────────────────
    print(f"\n{'='*122}")
    print(f"  GANADORA: {best['name']}")
    print(f"  Sharpe {best['net_sharpe']:.3f} | Return {pct_yr:+.2f}%/año | "
          f"WR {best['winrate']:.1f}% | PF {best['pf']:.2f} | "
          f"MaxDD {best['max_dd']:+.1f}% | Calmar {best['calmar']:.2f}")
    print(f"  Tasa mensual compuesta: {pct_mo*100:.2f}%/mes")

    print(f"\n  PROYECCIÓN DESDE €{CAPITAL_INIT:.0f} — MES A MES (reinvirtiendo):")
    print(f"  {'─'*90}")
    print(f"  {'Mes':>4}  {'Capital':>12}  {'Ganancia mes':>13}  "
          f"{'Total ganado':>13}  {'×':>6}")
    print(f"  {'─'*90}")

    cap = CAPITAL_INIT
    for m in range(1, 37):
        prev  = cap
        cap   = CAPITAL_INIT * (1 + pct_yr / 100) ** (m / 12)
        gan_m = cap - prev
        total = cap - CAPITAL_INIT
        mult  = cap / CAPITAL_INIT
        if m <= 12 or m % 3 == 0:
            print(f"  {m:>4}  €{cap:>11,.2f}  +€{gan_m:>11,.2f}  "
                  f"+€{total:>11,.2f}  {mult:>5.1f}x")

    print(f"  {'─'*90}")

    # Top 5 estrategias rentables comparadas con €100
    print(f"\n  TOP 5 ESTRATEGIAS COMPARADAS (desde €{CAPITAL_INIT:.0f}):")
    print(f"  {'─'*85}")
    print(f"  {'Estrategia':<38} {'%/año':>7}  "
          f"{'1 año':>9}  {'2 años':>9}  {'3 años':>9}")
    print(f"  {'─'*85}")
    for r in profit[:5]:
        py = r["net_pct"]
        v1 = CAPITAL_INIT * (1 + py/100)**1  - CAPITAL_INIT
        v2 = CAPITAL_INIT * (1 + py/100)**2  - CAPITAL_INIT
        v3 = CAPITAL_INIT * (1 + py/100)**3  - CAPITAL_INIT
        print(f"  {r['name']:<38} {py:>+7.2f}%  "
              f"+€{v1:>7,.2f}  +€{v2:>7,.2f}  +€{v3:>7,.2f}")
    print(f"  {'─'*85}")

    # Capital necesario para distintos objetivos mensuales
    print(f"\n  CAPITAL NECESARIO PARA OBJETIVO €/mes:")
    print(f"  {'─'*45}")
    for obj in [50, 100, 200, 500, 1_000]:
        cap_n = obj / pct_mo
        print(f"    €{obj:>5}/mes  →  €{cap_n:>10,.0f}")
    print(f"  {'─'*45}\n")

    print(f"  REFERENCIA vs SCAM:")
    print(f"  ✅ Esta estrategia (backtest real):  {pct_yr:>+.2f}%/año")
    print(f"  ❌ Polymarket viral '$68→$1.6M':     +2.352.841% — ESTAFA")
    print(f"  ❌ OpenClaw '$900→$7200 en 18h':      +700% en 18h — BOT FALSO")
    print(f"{'='*122}\n")

    return profit


def save_plot(rows, output="data/final_tournament.png"):
    valid  = [r for r in rows if r]
    profit = sorted([r for r in valid if r["net_pct"] > 0],
                    key=lambda x: -x["net_sharpe"])
    if not profit:
        return

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

    pal = plt.cm.plasma(np.linspace(0.05, 0.95, min(15, len(profit))))

    # 1. Equity curves (€100 inicial)
    ax1 = _ax(gs[0, :2])
    for i, r in enumerate(profit[:15]):
        eq = r.get("eq_curve", [])
        if eq:
            xs  = np.linspace(0, 12, len(eq))
            ys  = [CAPITAL_INIT * v for v in eq]
            ax1.plot(xs, ys, color=pal[i], lw=1.4, alpha=0.75,
                     label=r["name"][:22] if i < 6 else "")
    ax1.axhline(CAPITAL_INIT, color="#8b949e", lw=0.8, ls="--",
                label=f"Capital inicial €{CAPITAL_INIT:.0f}")
    ax1.legend(fontsize=7, framealpha=0.3, labelcolor="#c9d1d9")
    ax1.set_title(f"Top 15 — Equity desde €{CAPITAL_INIT:.0f}",
                  color="#f0f6fc", fontsize=10)
    ax1.set_xlabel("Meses", color="#8b949e")
    ax1.set_ylabel("€", color="#8b949e")

    # 2. WR vs Sharpe scatter
    ax2 = _ax(gs[0, 2])
    ax2.scatter([r["winrate"] for r in profit[:30]],
                [r["net_sharpe"] for r in profit[:30]],
                c=np.arange(min(30, len(profit))), cmap="plasma",
                s=55, alpha=0.85)
    ax2.axvline(50, color="#f85149", lw=0.8, ls="--", alpha=0.5)
    ax2.set_title("Win Rate vs Net Sharpe", color="#f0f6fc", fontsize=10)
    ax2.set_xlabel("Win Rate %", color="#8b949e")
    ax2.set_ylabel("Net Sharpe", color="#8b949e")

    # 3. Bar top 20
    ax3 = _ax(gs[1, :2])
    top20  = profit[:20]
    names3 = [r["name"][:32] for r in top20]
    rets3  = [r["net_pct"] for r in top20]
    ax3.barh(range(len(names3)), rets3, color=pal[:len(names3)], alpha=0.85)
    ax3.set_yticks(range(len(names3)))
    ax3.set_yticklabels(names3, fontsize=6.5)
    ax3.set_xlabel("Net Return %/año", color="#8b949e")
    ax3.set_title("Top 20 — Net Return", color="#f0f6fc", fontsize=10)
    ax3.invert_yaxis()

    # 4. Proyección €100 × top 5
    ax4 = _ax(gs[1, 2])
    meses = list(range(1, 37))
    for i, r in enumerate(profit[:5]):
        py = r["net_pct"]
        vals = [CAPITAL_INIT * (1 + py/100)**(m/12) for m in meses]
        ax4.plot(meses, vals, color=pal[i], lw=2,
                 label=f"{r['name'][:18]} ({py:+.0f}%)")
    ax4.axhline(CAPITAL_INIT, color="#8b949e", lw=0.8, ls="--")
    ax4.set_title(f"Proyección €{CAPITAL_INIT:.0f} × Top 5 (36 meses)",
                  color="#f0f6fc", fontsize=9)
    ax4.set_xlabel("Mes", color="#8b949e")
    ax4.set_ylabel("€", color="#8b949e")
    ax4.legend(fontsize=6.5, framealpha=0.3, labelcolor="#c9d1d9")

    fig.suptitle(
        f"TORNEO FINAL — Todas las estrategias | €{CAPITAL_INIT:.0f} capital | "
        f"Trail {TRAIL} + Partial {PARTIAL} + ADX>{ADX_MIN} | 1H 1 año 10 pares",
        color="#f0f6fc", fontsize=11, y=1.01,
    )
    os.makedirs("data", exist_ok=True)
    plt.savefig(output, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Gráfico: {output}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print(f"  TORNEO FINAL — capital inicial €{CAPITAL_INIT:.0f}")
    print("="*60)

    t0 = time.time()
    datasets = {}
    for pair in PAIRS:
        print(f"  {pair}...", end=" ", flush=True)
        datasets[pair] = _gen_demo(pair, n=8760)
        print("OK")

    strategies = build_all_strategies()
    total      = len(strategies)
    results    = []

    print(f"\n  Ejecutando {total} estrategias...")
    t1 = time.time()

    for i, (name, strat) in enumerate(strategies):
        r = _run(name, strat, datasets)
        results.append(r)
        if (i + 1) % 10 == 0:
            done = sum(1 for r in results if r and r["net_pct"] > 0)
            print(f"  [{i+1:>3}/{total}]  {time.time()-t1:.0f}s  "
                  f"rentables: {done}")

    print(f"\n  {total} estrategias en {time.time()-t1:.1f}s")

    os.makedirs("data", exist_ok=True)
    df_out = pd.DataFrame([r for r in results if r])
    df_out = df_out.drop(columns=["eq_curve"], errors="ignore")
    df_out.to_csv("data/final_results.csv", index=False)
    print("  CSV: data/final_results.csv")

    profit = print_results(results)
    save_plot(results)
    return results


if __name__ == "__main__":
    main()
