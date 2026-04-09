"""
max_profit_backtest.py

BACKTEST MÁXIMO PROFIT — Todas las estrategias ganadoras + config agresiva.

Toma los mejores resultados de TODOS los torneos anteriores:
  - AdaptMTF (Sharpe 8.003, +153%/yr)
  - MTF_sw3, sw5, sw7, sw10
  - Ensemble de top 3 con voto mayoritario (2/3)
  - Ensemble top 5 con voto mayoritario (3/5)

Grid search de parámetros agresivos:
  - Kelly max_fraction: 0.60, 0.70, 0.80
  - Trailing ATR mult: 3.0, 3.5, 4.0
  - Partial ratio: 0.33, 0.25, 0.20

Capital inicial: €100
Proyección mes a mes hasta el máximo retorno alcanzable.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from backtests.kelly_backtest_v2   import run_kelly_v2, kelly_metrics_v2
from strategies.kelly_sizer        import KellySizer
from strategies.risk_manager       import RiskManager
from strategies.regime_filter      import RegimeFilteredStrategy
from strategies.mtf_smc            import MultiTFSMC
from strategies.advanced_strategies import AdaptiveMTFStrategy

# ── Configuración base ────────────────────────────────────────────────────────

PAIRS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BNB-USDT",
    "ADA-USDT", "AVAX-USDT", "DOGE-USDT", "LTC-USDT", "DOT-USDT",
]

MAX_HOLD     = 24
PPY          = 8760          # periodos/año (velas 1H)
FEE_RT       = 0.002         # 0.20% round-trip KuCoin
ADX_MIN      = 13            # filtro régimen light (mejor probado)
CAPITAL_INIT = 100.0         # € capital inicial

RM = RiskManager(method="atr", atr_multiplier=1.0, rr_ratio=2.0)

# ── Grid de parámetros agresivos ──────────────────────────────────────────────

KELLY_CAPS   = [0.60, 0.70, 0.80]
TRAIL_MULTS  = [3.0, 3.5, 4.0]
PARTIAL_RATS = [0.33, 0.25, 0.20]


# ── Generador de datos sintéticos ─────────────────────────────────────────────

def _gen_demo(symbol: str, n: int = 8760) -> pd.DataFrame:
    seeds = {p: i for i, p in enumerate(PAIRS)}
    rng   = np.random.default_rng(seeds.get(symbol, 0))
    base  = {"BTC-USDT": 40000, "ETH-USDT": 2000,
             "SOL-USDT": 80,    "BNB-USDT": 300}.get(symbol, 10.0)
    p = [base]
    for _ in range(n - 1):
        p.append(max(1e-4, p[-1] + 0.3 + rng.standard_normal() * p[-1] * 0.008))
    p  = np.array(p)
    o  = np.concatenate([[p[0]], p[:-1]])
    s  = np.abs(rng.standard_normal(n)) * p * 0.003
    ts = pd.to_datetime([int(time.time()) - (n-i)*3600 for i in range(n)],
                        unit="s", utc=True)
    return pd.DataFrame(
        {"open": o, "high": np.maximum(o,p)+s,
         "low": np.minimum(o,p)-s, "close": p,
         "volume": rng.integers(500, 20000, n).astype(float)},
        index=ts
    )


def _wrap_regime(strat):
    """Aplica RegimeFilter light (ADX>13)."""
    return RegimeFilteredStrategy(
        strat, adx_min=ADX_MIN, di_align=True,
        atr_min_pct=0.2, atr_max_pct=6.0,
    )


# ── Motor de backtest con señales pre-computadas ──────────────────────────────

def _run_precomputed(sigs_by_pair, datasets, kelly_cap, trail, partial):
    """Corre backtest sobre todos los pares con señales ya computadas."""
    sizer = KellySizer(variant="full_kelly", min_trades=20,
                       max_fraction=kelly_cap, min_fraction=0.01)
    all_m, all_eq = [], []

    for pair, df in datasets.items():
        sigs = sigs_by_pair.get(pair)
        if not sigs:
            continue
        try:
            trades, eq = run_kelly_v2(
                sigs, df, sizer,
                max_hold=MAX_HOLD, risk_manager=RM,
                trailing_atr_mult=trail,
                partial_tp=True, partial_ratio=partial,
            )
            m = kelly_metrics_v2(trades, eq, periods_per_year=PPY)
            all_m.append(m)
            all_eq.append(eq)
        except Exception:
            pass

    if not all_m:
        return None

    n_pairs      = len(all_m)
    total_trades = sum(m["trades"] for m in all_m)
    avg_frac     = float(np.mean([m["avg_fraction"] for m in all_m]))
    gross_pct    = float(np.mean([(m["equity_curve"][-1]-1)*100
                                   for m in all_m if m["equity_curve"]]))
    fee_pct      = (total_trades / n_pairs) * FEE_RT * avg_frac * 100
    net_pct      = gross_pct - fee_pct
    gross_sh     = float(np.mean([m["sharpe"] for m in all_m]))
    net_sh       = gross_sh * net_pct / gross_pct if abs(gross_pct) > 1e-6 else 0.0
    months       = sum(len(df) for df in datasets.values()) / (720 * n_pairs)
    t_month      = (total_trades / n_pairs) / months if months else 0
    min_len      = min(len(e) for e in all_eq)
    eq_mean      = np.mean([e[:min_len] for e in all_eq], axis=0).tolist()

    return {
        "net_pct":    round(net_pct, 2),
        "gross_pct":  round(gross_pct, 2),
        "fee_pct":    round(fee_pct, 2),
        "net_sharpe": round(net_sh, 3),
        "winrate":    round(float(np.mean([m["winrate"] for m in all_m]))*100, 1),
        "pf":         round(float(np.mean([m["profit_factor"] for m in all_m])), 2),
        "max_dd":     round(float(np.mean([m["max_drawdown"] for m in all_m]))*100, 1),
        "t_month":    round(t_month, 1),
        "calmar":     round(float(np.mean([m["calmar"] for m in all_m])), 2),
        "eq_curve":   eq_mean,
        "kelly_cap":  kelly_cap,
        "trail":      trail,
        "partial":    partial,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MAX PROFIT BACKTEST — Estrategias Ganadoras + Config Agresiva")
    print("=" * 70)

    # Generar datos una sola vez
    print("\nGenerando datos sintéticos...")
    datasets = {p: _gen_demo(p) for p in PAIRS}
    print(f"  {len(PAIRS)} pares x 8760h = {len(PAIRS)*8760:,} velas totales")

    # ── Pre-computar señales (1 vez por estrategia/par) ───────────────────
    print("\nPre-computando señales...")
    strat_defs = {
        "AdaptMTF":    _wrap_regime(AdaptiveMTFStrategy()),
        "MTF_sw3_e50": _wrap_regime(MultiTFSMC(swing_window=3,  trend_ema=50)),
        "MTF_sw5_e50": _wrap_regime(MultiTFSMC(swing_window=5,  trend_ema=50)),
        "MTF_sw7_e50": _wrap_regime(MultiTFSMC(swing_window=7,  trend_ema=50)),
        "MTF_sw10_e50":_wrap_regime(MultiTFSMC(swing_window=10, trend_ema=50)),
    }

    precomputed = {}   # {strat_name: {pair: signals}}
    for sname, strat in strat_defs.items():
        precomputed[sname] = {}
        for pair, df in datasets.items():
            try:
                precomputed[sname][pair] = strat.generate_signals_batch(df)
            except Exception:
                precomputed[sname][pair] = ["hold"] * len(df)
        print(f"  {sname} ✓")

    # Ensemble signals (voto mayoritario desde señales pre-computadas)
    def _ensemble_sigs(pair, strat_names, threshold):
        sigs_list = [precomputed[s][pair] for s in strat_names
                     if pair in precomputed.get(s, {})]
        if not sigs_list:
            return []
        n = min(len(s) for s in sigs_list)
        result = []
        for i in range(n):
            votes_buy  = sum(1 for s in sigs_list if s[i] == "buy")
            votes_sell = sum(1 for s in sigs_list if s[i] == "sell")
            if votes_buy  >= threshold:
                result.append("buy")
            elif votes_sell >= threshold:
                result.append("sell")
            else:
                result.append("hold")
        return result

    ensemble_defs = {
        "Ensemble3(2/3)": (["AdaptMTF", "MTF_sw3_e50", "MTF_sw5_e50"], 2),
        "Ensemble5(3/5)": (["AdaptMTF", "MTF_sw3_e50", "MTF_sw5_e50",
                            "MTF_sw7_e50", "MTF_sw10_e50"], 3),
    }
    for ename, (names, thresh) in ensemble_defs.items():
        precomputed[ename] = {
            pair: _ensemble_sigs(pair, names, thresh) for pair in datasets
        }
        print(f"  {ename} ✓")

    all_strat_names = list(strat_defs.keys()) + list(ensemble_defs.keys())
    print(f"\n  {len(all_strat_names)} estrategias listas")

    # ── FASE 1: Grid search de parámetros ────────────────────────────────
    n_combos = len(KELLY_CAPS) * len(TRAIL_MULTS) * len(PARTIAL_RATS)
    total    = len(all_strat_names) * n_combos
    print(f"\n[FASE 1] Grid search: Kelly cap × Trail × Partial")
    print(f"  Estrategias: {len(all_strat_names)}")
    print(f"  Configs: {len(KELLY_CAPS)}×{len(TRAIL_MULTS)}×{len(PARTIAL_RATS)} = "
          f"{n_combos} combos por estrategia")
    print(f"  Total runs: {total}\n")

    results = []
    done    = 0

    for sname in all_strat_names:
        sigs_by_pair = precomputed[sname]
        for kc, tr, pr in itertools.product(KELLY_CAPS, TRAIL_MULTS, PARTIAL_RATS):
            done += 1
            m = _run_precomputed(sigs_by_pair, datasets, kc, tr, pr)
            if m:
                m["name"]     = f"{sname}|K{kc}|T{tr}|P{pr}"
                m["strategy"] = sname
                results.append(m)
            if done % 20 == 0 or done == total:
                print(f"  [{done:3d}/{total}] {done/total*100:4.0f}% completado", end="\r")

    print(f"\n  {len(results)} configuraciones completadas")

    # ── FASE 2: Resultados y ranking ─────────────────────────────────────
    df_res = pd.DataFrame(results).sort_values("net_pct", ascending=False)

    print("\n" + "=" * 70)
    print("  TOP 20 — RANKING POR RETORNO NETO")
    print("=" * 70)
    cols = ["name", "net_pct", "net_sharpe", "winrate", "pf",
            "max_dd", "calmar", "kelly_cap", "trail", "partial"]
    print(df_res[cols].head(20).to_string(index=False))

    # ── FASE 3: Mejor config por estrategia ──────────────────────────────
    print("\n" + "=" * 70)
    print("  MEJOR CONFIG POR ESTRATEGIA")
    print("=" * 70)
    best_per = df_res.groupby("strategy").first().reset_index()
    best_per = best_per.sort_values("net_pct", ascending=False)
    print(best_per[["strategy", "net_pct", "net_sharpe",
                    "winrate", "pf", "max_dd",
                    "kelly_cap", "trail", "partial"]].to_string(index=False))

    # ── FASE 4: Ganador absoluto ──────────────────────────────────────────
    winner = df_res.iloc[0]
    print("\n" + "=" * 70)
    print(f"  GANADOR ABSOLUTO: {winner['name']}")
    print("=" * 70)
    print(f"  Retorno neto anual : {winner['net_pct']:+.2f}%")
    print(f"  Sharpe             : {winner['net_sharpe']:.3f}")
    print(f"  Win rate           : {winner['winrate']:.1f}%")
    print(f"  Profit factor      : {winner['pf']:.2f}")
    print(f"  Max drawdown       : {winner['max_dd']:.1f}%")
    print(f"  Calmar ratio       : {winner['calmar']:.2f}")
    print(f"  Trades/mes         : {winner['t_month']:.1f}")
    print(f"  Kelly cap          : {winner['kelly_cap']:.0%}")
    print(f"  Trailing ATR mult  : {winner['trail']}")
    print(f"  Partial ratio      : {winner['partial']:.0%}")

    # ── FASE 5: Proyección €100 mes a mes ────────────────────────────────
    monthly_ret = (1 + winner["net_pct"] / 100) ** (1/12) - 1
    print(f"\n  Retorno mensual equiv.: {monthly_ret*100:.2f}%")

    print("\n" + "=" * 70)
    print(f"  PROYECCIÓN CAPITAL €{CAPITAL_INIT:.0f} — MES A MES (config ganadora)")
    print("=" * 70)
    print(f"  {'Mes':>4}  {'Capital €':>12}  {'Ganancia €':>12}  {'Mult':>6}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*6}")

    capital    = CAPITAL_INIT
    milestones = {200: False, 500: False, 1000: False, 5000: False,
                  10000: False, 50000: False, 100000: False}

    for mes in range(1, 61):
        capital  = capital * (1 + monthly_ret)
        ganancia = capital - CAPITAL_INIT
        mult     = capital / CAPITAL_INIT

        marker = ""
        for m, hit in milestones.items():
            if not hit and capital >= m:
                milestones[m] = True
                marker = f"  <- €{m:,}"

        if mes <= 36 or mes % 12 == 0:
            print(f"  {mes:4d}  {capital:12.2f}  {ganancia:12.2f}  {mult:6.2f}x{marker}")

    print(f"\n  Año 1 (mes 12): €{CAPITAL_INIT*(1+monthly_ret)**12:.2f}")
    print(f"  Año 2 (mes 24): €{CAPITAL_INIT*(1+monthly_ret)**24:.2f}")
    print(f"  Año 3 (mes 36): €{CAPITAL_INIT*(1+monthly_ret)**36:.2f}")
    print(f"  Año 5 (mes 60): €{CAPITAL_INIT*(1+monthly_ret)**60:.2f}")

    # ── FASE 6: Guardar resultados ────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    df_save = df_res.drop(columns=["eq_curve"], errors="ignore")
    df_save.to_csv("data/max_profit_results.csv", index=False)
    print(f"\n  Resultados guardados: data/max_profit_results.csv")

    # ── FASE 7: Gráfica comparativa ───────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Max Profit Backtest — Análisis Completo", fontsize=14, fontweight="bold")

    # 1. Mejor retorno por estrategia
    ax = axes[0, 0]
    bp = best_per.sort_values("net_pct")
    colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in bp["net_pct"]]
    ax.barh(bp["strategy"], bp["net_pct"], color=colors)
    ax.set_xlabel("Retorno neto anual (%)")
    ax.set_title("Mejor Retorno por Estrategia")
    ax.axvline(x=0, color="black", linewidth=0.5)

    # 2. Scatter Sharpe vs Retorno
    ax = axes[0, 1]
    sc = ax.scatter(df_res["net_pct"], df_res["net_sharpe"],
                    c=df_res["winrate"], cmap="RdYlGn",
                    alpha=0.5, s=25, vmin=55, vmax=70)
    plt.colorbar(sc, ax=ax, label="Win Rate (%)")
    ax.set_xlabel("Retorno neto anual (%)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Sharpe vs Retorno (color=WR%)")
    ax.scatter([winner["net_pct"]], [winner["net_sharpe"]],
               color="red", s=200, zorder=5, marker="*", label="Ganador")
    ax.legend(fontsize=8)

    # 3. Equity curve del ganador
    ax = axes[1, 0]
    if winner.get("eq_curve"):
        eq = winner["eq_curve"]
        ax.plot(np.array(eq) * CAPITAL_INIT, color="#2ecc71", linewidth=1.5)
        ax.fill_between(range(len(eq)), CAPITAL_INIT,
                        np.array(eq)*CAPITAL_INIT, alpha=0.3, color="#2ecc71")
        ax.set_xlabel("Horas")
        ax.set_ylabel("Capital (€)")
        ax.set_title(f"Equity Curve — {winner['strategy']}\n"
                     f"K{winner['kelly_cap']:.0%} T{winner['trail']} P{winner['partial']:.0%}")
        ax.axhline(y=CAPITAL_INIT, color="gray", linestyle="--", alpha=0.5)

    # 4. Impacto Kelly cap vs retorno
    ax = axes[1, 1]
    for kc in KELLY_CAPS:
        subset = df_res[df_res["kelly_cap"] == kc].sort_values("net_pct", ascending=False)
        ax.plot(range(len(subset)), subset["net_pct"].values,
                marker="o", markersize=3, label=f"Kelly {kc:.0%}", linewidth=1.5)
    ax.set_xlabel("Config rank")
    ax.set_ylabel("Retorno neto (%)")
    ax.set_title("Impacto Kelly Cap en Retorno")
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig("data/max_profit_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Gráfica guardada: data/max_profit_analysis.png")

    # ── FASE 8: Resumen ejecutivo ─────────────────────────────────────────
    n_pos = (df_res["net_pct"] > 0).sum()
    print("\n" + "=" * 70)
    print("  RESUMEN EJECUTIVO")
    print("=" * 70)
    print(f"  Configuraciones probadas    : {len(df_res)}")
    print(f"  Configuraciones rentables   : {n_pos} ({n_pos/len(df_res)*100:.0f}%)")
    print(f"  Retorno máximo encontrado   : {df_res['net_pct'].max():+.2f}%/yr")
    print(f"  Retorno promedio (rentables): "
          f"{df_res[df_res['net_pct']>0]['net_pct'].mean():+.2f}%/yr")
    print(f"  Sharpe máximo              : {df_res['net_sharpe'].max():.3f}")

    print(f"\n  Mejor Kelly cap:")
    for kc, ret in df_res.groupby("kelly_cap")["net_pct"].max().sort_values(ascending=False).items():
        print(f"    {kc:.0%} → {ret:+.2f}%/yr")
    print(f"\n  Mejor Trailing mult:")
    for tr, ret in df_res.groupby("trail")["net_pct"].max().sort_values(ascending=False).items():
        print(f"    {tr}× → {ret:+.2f}%/yr")
    print(f"\n  Mejor Partial ratio:")
    for pr, ret in df_res.groupby("partial")["net_pct"].max().sort_values(ascending=False).items():
        print(f"    {pr:.0%} → {ret:+.2f}%/yr")

    return df_res


if __name__ == "__main__":
    main()
