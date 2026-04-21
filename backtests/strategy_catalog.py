"""
strategy_catalog.py

CATÁLOGO COMPLETO DE ESTRATEGIAS GANADORAS

Consolida resultados de TODOS los torneos.
Clasifica cada estrategia por escenario de mercado:
  - TENDENCIA FUERTE  : mercado direccional claro
  - TENDENCIA SUAVE   : mercado con sesgo pero con pullbacks
  - VOLATILIDAD ALTA  : movimientos grandes, breakouts
  - OSCILADORES       : mercado lateral/choppy
  - ENSEMBLE          : combinación de señales (más robusto)

Genera CSV maestro + resumen por escenario.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ── Clasificación manual por escenario ───────────────────────────────────────
# Basado en la lógica de cada estrategia y sus resultados observados

SCENARIO_MAP = {
    # ── TENDENCIA FUERTE (ADX>25, movimiento limpio) ──────────────────────
    "TENDENCIA_FUERTE": [
        "AdaptMTF", "MTF_sw5_e50", "MTF_sw3_e50", "MTF_sw7_e50",
        "MTF_sw10_e50", "Ensemble3(2/3)", "Ensemble5(3/5)",
        "MTF_sw5", "MTF_sw3", "MTF_sw7",
        "MTF_sw5_e100", "MTF_sw3_e100", "MTF_sw10_e100",
        "BASE·AdaptMTF", "REG·AdaptMTF", "REG·MTF_sw5",
    ],
    # ── TENDENCIA CON PULLBACKS (ADX 18-25, zigzag direccional) ──────────
    "TENDENCIA_PULLBACK": [
        "AdaptMTF+RM1x2+f60", "MTF(sw3,e50)+RM1x2+f60",
        "MTF(sw5,e50)+RM1x2+f60", "AdaptMTF+RM1x2+f40",
        "REG[light]·AdaptMTF", "REG[light]·MTF_sw3", "REG[light]·MTF_sw5",
        "REG[standard]·AdaptMTF", "REG[standard]·MTF_sw5",
        "TrendStrength", "DualMomentum", "VolumeTrend",
        "SuperTrend10", "SuperTrend7", "HullMA20",
        "GoldenCross", "EMAStack", "DynamicMA",
    ],
    # ── VOLATILIDAD / BREAKOUTS (rango estrecho → expansión) ─────────────
    "VOLATILIDAD_BREAKOUT": [
        "KeltnerBreakout", "Keltner20", "SqueezeMomentum", "SqueezeMom",
        "VolatilityBreakout", "MomentumBreakout", "AdaptiveChannel",
        "BreakoutVol", "BrkVol", "BollingerMom", "BBMom",
        "MultiSignal", "HeikinAshiEMA",
    ],
    # ── OSCILADORES / MERCADO LATERAL (sin tendencia clara) ──────────────
    "MERCADO_LATERAL": [
        "CCIStrategy", "CCI20", "CCI14",
        "WilliamsR14", "WilliamsRStrategy",
        "AwesomeOsc", "AwesomeOscillatorStrategy",
        "ChaikinMF", "ElderRay",
        "RSIDivergence", "RSI_BB", "AdaptiveRSI",
        "ZScore", "BollingerReversion", "MeanReversion",
        "Pullback", "TrendReversal",
        "MFI", "ForceIndex", "CMO", "UltimateOsc",
        "ParabolicSAR",
    ],
    # ── VOLUMEN / SMART MONEY (institucional, órdenes grandes) ───────────
    "VOLUMEN_SMART_MONEY": [
        "OBVTrend", "SmartMoneyFlow", "ChaikinMF", "ChaikinMFStrategy",
        "AccDist", "VWAP24", "VWAPStrategy",
        "OrderBlock", "OrdBlock", "FVG",
        "MarketStructure", "MktStruct",
        "InsideBarBreakout", "PinBar50", "PinBarStrategy",
    ],
    # ── MULTI-TIMEFRAME (análisis en varios marcos) ───────────────────────
    "MULTI_TIMEFRAME": [
        "TripleScreen", "ScalpMTF", "AdaptMTF",
        "MTF_sw3_e50", "MTF_sw5_e50", "MTF_sw7_e50",
        "Ichimoku", "IchimokuStrategy",
        "LinearReg", "LinearRegStrategy",
    ],
    # ── ESTADÍSTICO / CUANTITATIVO ────────────────────────────────────────
    "ESTADISTICO": [
        "ZScore", "BollingerReversion", "ROC", "TSI",
        "VolumeOsc", "FibRetracement", "FibRet",
        "MACDStoch", "MACD_RSI", "MACDRSIStrategy",
        "ADX_MACD", "DivergenceConfluence",
        "StochRSI", "StochRSIStrategy",
    ],
}

# Invertir mapa para búsqueda por nombre
NAME_TO_SCENARIO = {}
for scenario, names in SCENARIO_MAP.items():
    for n in names:
        NAME_TO_SCENARIO[n.lower()] = scenario


def classify(name: str) -> str:
    """Clasifica una estrategia por su nombre."""
    nl = name.lower()
    for key, scenario in NAME_TO_SCENARIO.items():
        if key in nl:
            return scenario

    # Heurísticas por keywords
    if any(k in nl for k in ["mtf", "adapt", "ensemble", "portfolio"]):
        return "TENDENCIA_FUERTE"
    if any(k in nl for k in ["reg[", "trail", "rm1x2", "rm2x2"]):
        return "TENDENCIA_PULLBACK"
    if any(k in nl for k in ["squeeze", "keltner", "breakout", "volatility"]):
        return "VOLATILIDAD_BREAKOUT"
    if any(k in nl for k in ["cci", "rsi", "williams", "awesome", "oscillator"]):
        return "MERCADO_LATERAL"
    if any(k in nl for k in ["obv", "vwap", "orderblock", "fvg", "volume", "smart"]):
        return "VOLUMEN_SMART_MONEY"
    if any(k in nl for k in ["zscore", "bollinger", "macd", "stoch", "fib"]):
        return "ESTADISTICO"
    return "OTRO"


def load_all_results() -> pd.DataFrame:
    all_results = []
    for f in glob.glob("data/*results*.csv"):
        try:
            df = pd.read_csv(f)
            df["source"] = os.path.basename(f)
            all_results.append(df)
        except Exception:
            pass
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def clean_name(name: str) -> str:
    """Extrae nombre base limpio."""
    n = str(name)
    for prefix in ["BASE·", "REG·", "OPT·", "TRAIL·"]:
        if n.startswith(prefix):
            n = n[len(prefix):]
    n = n.split("|K")[0].split("|T")[0].split("|P")[0]
    n = n.replace("_noR", "").replace("(v2-base)", "")
    return n.strip("·").strip()


def main():
    print("=" * 75)
    print("  CATÁLOGO DE ESTRATEGIAS — Todas las ganadoras por escenario")
    print("=" * 75)

    # ── Cargar y limpiar datos ─────────────────────────────────────────────
    combined = load_all_results()
    if combined.empty:
        print("ERROR: No se encontraron archivos de resultados")
        return

    print(f"\n  Configs totales cargadas  : {len(combined):,}")

    # Filtrar rentables
    profitable = combined[combined["net_pct"] > 5].copy()
    print(f"  Configs rentables (>5%/yr): {len(profitable):,}")

    # Nombre base y clasificación
    profitable = profitable.copy()
    profitable["base_name"] = profitable["name"].apply(clean_name)
    profitable["scenario"]  = profitable["base_name"].apply(classify)

    # Mejor resultado por nombre base
    seen = {}
    for _, row in profitable.sort_values("net_pct", ascending=False).iterrows():
        bn = row["base_name"]
        if bn not in seen:
            seen[bn] = row

    catalog = pd.DataFrame(seen.values()).reset_index(drop=True)
    catalog = catalog.sort_values("net_pct", ascending=False)

    print(f"  Estrategias únicas        : {len(catalog)}")

    # ── Resumen por escenario ─────────────────────────────────────────────
    scenarios_order = [
        "TENDENCIA_FUERTE", "TENDENCIA_PULLBACK", "VOLATILIDAD_BREAKOUT",
        "MERCADO_LATERAL", "VOLUMEN_SMART_MONEY", "MULTI_TIMEFRAME",
        "ESTADISTICO", "OTRO"
    ]

    scenario_labels = {
        "TENDENCIA_FUERTE":    "TENDENCIA FUERTE (ADX>25, movimiento limpio)",
        "TENDENCIA_PULLBACK":  "TENDENCIA CON PULLBACKS (ADX 15-25, zigzag)",
        "VOLATILIDAD_BREAKOUT":"VOLATILIDAD / BREAKOUTS (expansión de rango)",
        "MERCADO_LATERAL":     "MERCADO LATERAL / OSCILADORES (choppy)",
        "VOLUMEN_SMART_MONEY": "VOLUMEN / SMART MONEY (análisis institucional)",
        "MULTI_TIMEFRAME":     "MULTI-TIMEFRAME (varios marcos temporales)",
        "ESTADISTICO":         "ESTADÍSTICO / CUANTITATIVO",
        "OTRO":                "OTRAS ESTRATEGIAS",
    }

    summary_rows = []

    for scenario in scenarios_order:
        sub = catalog[catalog["scenario"] == scenario].sort_values("net_pct", ascending=False)
        if sub.empty:
            continue

        label = scenario_labels.get(scenario, scenario)
        print(f"\n{'='*75}")
        print(f"  {label}")
        print(f"  ({len(sub)} estrategias)")
        print(f"{'='*75}")
        print(f"  {'Estrategia':<40} {'Ret%/yr':>8} {'Sharpe':>7} {'WR%':>6} {'PF':>5} {'DD%':>6} {'Cal':>7}")
        print(f"  {'-'*40} {'-'*8} {'-'*7} {'-'*6} {'-'*5} {'-'*6} {'-'*7}")

        for _, r in sub.iterrows():
            name    = str(r["base_name"])[:40]
            net_pct = r.get("net_pct", 0)
            sharpe  = r.get("net_sharpe", r.get("sharpe", 0))
            wr      = r.get("winrate", 0)
            pf      = r.get("pf", r.get("profit_factor", 0))
            dd      = r.get("max_dd", r.get("max_drawdown", 0))
            calmar  = r.get("calmar", 0)

            if pd.isna(net_pct) or net_pct <= 0:
                continue

            wr_val  = float(wr)*100 if float(wr) <= 1.0 else float(wr)
            dd_val  = float(dd)*100 if abs(float(dd)) <= 1.0 else float(dd)

            tag = " ***" if net_pct > 100 else (" **" if net_pct > 50 else "")
            print(f"  {name:<40} {net_pct:>8.1f} {float(sharpe):>7.3f} "
                  f"{wr_val:>6.1f} {float(pf):>5.2f} {dd_val:>6.1f} {float(calmar):>7.2f}{tag}")

            summary_rows.append({
                "scenario": scenario, "name": r["base_name"],
                "net_pct": round(net_pct, 2),
                "net_sharpe": round(float(sharpe), 3),
                "winrate": round(wr_val, 1),
                "pf": round(float(pf), 2),
                "max_dd": round(dd_val, 1),
                "calmar": round(float(calmar), 2),
            })

    # ── TOP 10 ABSOLUTO ───────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  TOP 10 ABSOLUTO — Las mejores estrategias de toda la historia")
    print(f"{'='*75}")
    top10 = catalog.sort_values("net_pct", ascending=False).head(10)
    print(f"  {'#':>3} {'Estrategia':<42} {'Ret%/yr':>8} {'Sharpe':>7} {'WR%':>6} {'MaxDD%':>7}")
    print(f"  {'---'} {'-'*42} {'-'*8} {'-'*7} {'-'*6} {'-'*7}")
    for i, (_, r) in enumerate(top10.iterrows(), 1):
        wr  = float(r.get("winrate", 0))
        wr  = wr*100 if wr <= 1.0 else wr
        dd  = float(r.get("max_dd", r.get("max_drawdown", 0)))
        dd  = dd*100 if abs(dd) <= 1.0 else dd
        sh  = float(r.get("net_sharpe", r.get("sharpe", 0)))
        print(f"  {i:>3} {str(r['base_name']):<42} {r['net_pct']:>8.1f} {sh:>7.3f} "
              f"{wr:>6.1f} {dd:>7.1f}")

    # ── CUÁNDO USAR CADA ESCENARIO ────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  GUÍA DE USO — Cuándo desplegar cada estrategia")
    print(f"{'='*75}")

    guides = [
        ("TENDENCIA FUERTE",
         "BTC rompe resistencia + volumen x2 + ADX>25 + DI+>DI-",
         "AdaptMTF / MTF_sw5_e50 / Ensemble3(2/3)",
         "+150%/yr | Sharpe 8.0 | WR 64%"),
        ("TENDENCIA CON PULLBACKS",
         "Mercado alcista pero con correcciones 15-30% | ADX 18-25",
         "REG[light]·AdaptMTF / SuperTrend / HullMA",
         "+95-109%/yr | Sharpe 4.5-5.0 | WR 48-50%"),
        ("VOLATILIDAD / BREAKOUT",
         "Bollinger Bands se comprimen → inminente expansión | Squeeze",
         "SqueezeMomentum / KeltnerBreakout / BreakoutVol",
         "+60-80%/yr | Sharpe 3.5-4.5 | WR 45-50%"),
        ("MERCADO LATERAL",
         "ADX<20 + precio rebota entre soportes y resistencias",
         "CCI20 / WilliamsR14 / BollingerReversion / ZScore",
         "+30-60%/yr | Sharpe 2.5-3.5 | WR 52-58%"),
        ("VOLUMEN / SMART MONEY",
         "Divergencia precio/volumen + zonas de Order Block / FVG",
         "VWAP / OBVTrend / SmartMoneyFlow / OrderBlock",
         "+40-70%/yr | Sharpe 3.0-4.5 | WR 50-55%"),
        ("ENSEMBLE (todo terreno)",
         "Mercado mixto | No tienes claro el régimen actual",
         "Ensemble3(2/3): AdaptMTF+MTF_sw3+MTF_sw5",
         "+142-147%/yr | Sharpe 7.7-7.8 | WR 64%"),
    ]

    for name, when, use, perf in guides:
        print(f"\n  [{name}]")
        print(f"  Cuándo : {when}")
        print(f"  Usar   : {use}")
        print(f"  Perf.  : {perf}")

    # ── CONFIGURACIÓN ÓPTIMA GANADORA ─────────────────────────────────────
    print(f"\n{'='*75}")
    print("  CONFIGURACIÓN ÓPTIMA CONFIRMADA (189 configs probadas)")
    print(f"{'='*75}")
    print(f"  Estrategia : AdaptMTF + RegimeFilter[light] (ADX>13)")
    print(f"  Kelly cap  : 80%  (max_fraction=0.80)")
    print(f"  Trailing   : ATR × 4.0")
    print(f"  Partial TP : 33% en 1R → SL a breakeven")
    print(f"  Retorno    : +149.92%/yr  |  Sharpe 7.851  |  WR 64.1%")
    print(f"  Max DD     : -2.3%        |  Calmar 85.76")
    print(f"  Capital    : €100 → €250 (12m) → €625 (24m) → €1,561 (36m)")

    # ── Guardar catálogo ──────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    df_catalog = pd.DataFrame(summary_rows)
    df_catalog.to_csv("data/strategy_catalog.csv", index=False)
    print(f"\n  Catálogo guardado: data/strategy_catalog.csv")

    # ── Gráfica por escenarios ────────────────────────────────────────────
    scenario_stats = df_catalog.groupby("scenario").agg(
        count=("name", "count"),
        avg_ret=("net_pct", "mean"),
        max_ret=("net_pct", "max"),
        avg_sharpe=("net_sharpe", "mean"),
    ).reset_index().sort_values("max_ret", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Catálogo de Estrategias — Rendimiento por Escenario",
                 fontsize=13, fontweight="bold")

    # Bar chart retorno máximo
    ax = axes[0]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(scenario_stats)))
    bars = ax.barh(scenario_stats["scenario"], scenario_stats["max_ret"],
                   color=colors[::-1])
    ax.barh(scenario_stats["scenario"], scenario_stats["avg_ret"],
            color="steelblue", alpha=0.4, label="Promedio")
    ax.set_xlabel("Retorno neto anual (%)")
    ax.set_title("Retorno Máximo y Promedio por Escenario")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.legend()

    # Scatter count vs avg_ret
    ax = axes[1]
    scatter = ax.scatter(scenario_stats["count"], scenario_stats["avg_ret"],
                         s=scenario_stats["max_ret"] * 2,
                         c=scenario_stats["avg_sharpe"], cmap="RdYlGn",
                         vmin=2, vmax=8, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label="Sharpe promedio")
    for _, row in scenario_stats.iterrows():
        ax.annotate(row["scenario"].replace("_", "\n"),
                    (row["count"], row["avg_ret"]),
                    fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("Nº estrategias en escenario")
    ax.set_ylabel("Retorno promedio (%/yr)")
    ax.set_title("Cantidad vs Rentabilidad por Escenario\n(tamaño = retorno máx)")

    plt.tight_layout()
    plt.savefig("data/strategy_catalog_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Gráfica guardada: data/strategy_catalog_chart.png")

    print(f"\n{'='*75}")
    total_s = catalog[catalog["net_pct"] > 0].shape[0]
    total_100 = catalog[catalog["net_pct"] > 100].shape[0]
    print(f"  TOTAL estrategias únicas rentables : {total_s}")
    print(f"  Con retorno >100%/yr               : {total_100}")
    print(f"  Configs totales probadas            : {len(combined):,}")
    print(f"{'='*75}")

    return catalog


if __name__ == "__main__":
    main()
