"""
fee_aware_tournament.py

Torneo Fee-Aware — 120 configuraciones en datos 4H.

Solo estrategias de swing que sobrevivan a las comisiones reales de KuCoin:
  Taker: 0.1% entrada + 0.1% salida = 0.2% round-trip (0.002 fracción)

Métricas clave:
  gross_pct   = retorno bruto sobre capital (sin comisiones)
  fee_cost_pct= trades * 0.002 * avg_fraction * 100
  net_pct     = gross_pct - fee_cost_pct
  net_sharpe  = Sharpe ajustado: gross_sharpe * (net_pct / gross_pct)

Solo aparecen en el ranking estrategias con net_pct > 0 y >= 5 trades totales.

Uso:
    python backtests/fee_aware_tournament.py
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
from backtests.backtester_fast import _precompute_signals
from strategies.kelly_sizer   import KellySizer

# Estrategias ganadoras de torneos anteriores
from strategies.smc_strategy  import SMCStrategy
from strategies.strategy_zoo2 import (
    ParabolicSARStrategy, WilliamsRStrategy, HeikinAshiStrategy,
    KeltnerBreakoutStrategy, VWAPStrategy, ZScoreStrategy, ROCStrategy,
    BollingerBandStrategy,
)
from strategies.strategy_zoo  import (
    CCIStrategy, EngulfingStrategy, DonchianBreakoutStrategy,
    ADXDIStrategy, SupertrendStrategy, TripleEMAStrategy,
)
# Estrategias swing nuevas
from strategies.swing_strategies import (
    ChandelierExitStrategy, TurtleBreakoutStrategy, MultiEMASwingStrategy,
)


# ── Constantes ────────────────────────────────────────────────────────────────

KUCOIN_TAKER_FEE = 0.001   # 0.1% por lado
FEE_ROUNDTRIP    = KUCOIN_TAKER_FEE * 2   # 0.2% total

PAIRS_4H = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BNB-USDT"]
MAX_HOLD  = 48      # 48 velas × 4h = 8 días máximo por trade
PPY       = 2190    # 4h candles per year (365 × 6)
SIZER     = KellySizer(variant="half_kelly", min_trades=20)


# ── 120 configuraciones fee-aware ─────────────────────────────────────────────

def build_configs() -> list:
    configs = []

    def add(name, strat):
        configs.append((name, strat))

    # ── SMC variants (15) ────────────────────────────────────────────────
    for sw, hold in [
        (3, 24), (3, 36), (3, 48), (3, 72),
        (5, 24), (5, 36), (5, 48), (5, 72),
        (7, 24), (7, 36), (7, 48),
        (10, 24), (10, 36), (10, 48), (10, 72),
    ]:
        add(f"SMC(sw{sw},h{hold})", SMCStrategy(swing_window=sw))

    # ── HeikinAshi variants (10) ─────────────────────────────────────────
    for cb, et, rsi_os, rsi_ob in [
        (2,  50,  45, 55), (2,  100, 45, 55), (2,  200, 45, 55),
        (3,  50,  45, 55), (3,  100, 45, 55), (3,  200, 45, 55),
        (4,  100, 45, 55), (4,  200, 45, 55),
        (2,  100, 40, 60), (3,  100, 40, 60),
    ]:
        add(f"HA(cb{cb},et{et},os{rsi_os})",
            HeikinAshiStrategy(confirm_bars=cb, ema_trend=et, rsi_os=rsi_os, rsi_ob=rsi_ob))

    # ── Parabolic SAR variants (10) ──────────────────────────────────────
    for af, afmax, et in [
        (0.01, 0.1, 50),  (0.01, 0.1, 100),  (0.01, 0.2, 50),
        (0.02, 0.2, 50),  (0.02, 0.2, 100),  (0.02, 0.2, 200),
        (0.02, 0.3, 100), (0.03, 0.2, 100),
        (0.01, 0.15, 100),(0.02, 0.25, 100),
    ]:
        add(f"SAR(af{af},m{afmax},et{et})",
            ParabolicSARStrategy(af_start=af, af_max=afmax, ema_trend=et))

    # ── Williams %R variants (10) ────────────────────────────────────────
    for p, os_, et in [
        (14, -80, 50),  (14, -80, 100),  (14, -85, 50),
        (21, -80, 50),  (21, -80, 100),  (21, -85, 100),
        (28, -80, 50),  (28, -80, 100),  (28, -85, 100),
        (14, -75, 100),
    ]:
        add(f"WR(p{p},os{os_},et{et})",
            WilliamsRStrategy(period=p, oversold=os_, ema_trend=et))

    # ── Chandelier Exit variants (12) ────────────────────────────────────
    for p, m, et in [
        (14, 2.0, 50),  (14, 2.5, 50),  (14, 3.0, 50),
        (14, 3.0, 100), (14, 3.0, 200),
        (22, 2.0, 100), (22, 3.0, 100), (22, 3.0, 200),
        (30, 3.0, 100), (30, 3.0, 200),
        (14, 2.0, 100), (22, 2.5, 100),
    ]:
        add(f"Chandelier(p{p},m{m},et{et})",
            ChandelierExitStrategy(period=p, mult=m, ema_trend=et))

    # ── Turtle Breakout variants (10) ────────────────────────────────────
    for en, ex, et in [
        (20, 10, 100), (20, 10, 200), (20, 5,  100),
        (30, 15, 100), (30, 10, 100), (30, 15, 200),
        (40, 20, 100), (40, 20, 200),
        (15, 7,  100), (25, 12, 100),
    ]:
        add(f"Turtle(en{en},ex{ex},et{et})",
            TurtleBreakoutStrategy(entry_period=en, exit_period=ex, ema_trend=et))

    # ── MultiEMA Swing + ADX (10) ─────────────────────────────────────────
    for e1, e2, e3, adx_t in [
        (21, 55, 100, 25), (21, 55, 100, 30), (21, 55, 200, 25),
        (34, 89, 200, 25), (34, 89, 200, 30),
        (21, 50, 100, 20), (50, 100, 200, 25), (50, 100, 200, 30),
        (21, 55, 100, 20), (34, 89, 200, 20),
    ]:
        add(f"MultiEMA(e1={e1},e2={e2},e3={e3},adx>{adx_t})",
            MultiEMASwingStrategy(e1=e1, e2=e2, e3=e3, adx_thresh=adx_t))

    # ── CCI swing (8) ────────────────────────────────────────────────────
    for p, t, et in [
        (20, 100, 50), (30, 100, 50), (20, 150, 50),
        (20, 100, 100),(30, 150, 100),(40, 100, 100),
        (20, 100, 200),(30, 100, 200),
    ]:
        add(f"CCI(p{p},t{t},et{et})",
            CCIStrategy(period=p, threshold=t, ema_trend=et))

    # ── Donchian Swing (8) ────────────────────────────────────────────────
    for p, ep in [
        (20, 10), (20, 5), (30, 15), (30, 10),
        (40, 20), (50, 25),(20, 7),  (40, 15),
    ]:
        add(f"Donchian(p{p},ep{ep})",
            DonchianBreakoutStrategy(period=p, exit_period=ep))

    # ── Supertrend Swing (7) ─────────────────────────────────────────────
    for p, m in [
        (10, 2.0), (10, 3.0), (14, 2.0),
        (14, 3.0), (14, 3.5), (20, 3.0), (20, 4.0),
    ]:
        add(f"Supertrend(p{p},m{m})",
            SupertrendStrategy(atr_period=p, multiplier=m))

    # ── ADX/DI Swing (7) ─────────────────────────────────────────────────
    for p, t, ts in [
        (14, 20, 30), (14, 25, 35), (14, 25, 40),
        (14, 30, 40), (21, 20, 30), (21, 25, 35),
        (14, 20, 25),
    ]:
        add(f"ADX(p{p},t{t},ts{ts})",
            ADXDIStrategy(period=p, adx_threshold=t, adx_strong=ts))

    # ── ZScore mean reversion (3) ─────────────────────────────────────────
    for p, z, et in [
        (50, 2.0, 100), (100, 2.0, 200), (50, 2.5, 100),
    ]:
        add(f"ZScore(p{p},z{z},et{et})",
            ZScoreStrategy(period=p, z_entry=z, ema_trend=et))

    assert len(configs) == 110, f"Expected 110 configs, got {len(configs)}"
    return configs


# ── Motor del torneo ──────────────────────────────────────────────────────────

def _run_one(name, strat, datasets):
    """
    Backtest de 1 config en todos los pares → métricas promediadas.
    Retorna dict con métricas incluyendo fee-adjusted net return.
    """
    all_m = []
    for pair, df in datasets.items():
        try:
            # Dispatch: SMC usa _precompute_signals, el resto generate_signals_batch
            if hasattr(strat, "swing_window") and isinstance(strat, SMCStrategy):
                sigs = _precompute_signals(
                    df,
                    swing_window    = strat.swing_window,
                    require_fvg     = strat.require_fvg,
                    use_choch_filter = strat.use_choch_filter,
                )
            else:
                sigs = strat.generate_signals_batch(df)

            trades, eq = _run_kelly_trades(sigs, df, SIZER, max_hold=MAX_HOLD)
            m = _kelly_metrics(trades, eq, periods_per_year=PPY)
            all_m.append(m)
        except Exception as e:
            pass   # par omitido si falla

    if not all_m:
        return None

    def avg(key):
        vals = [m[key] for m in all_m if m[key] is not None]
        return np.mean(vals) if vals else 0.0

    def avg_eq_end():
        ends = [m["equity_curve"][-1] for m in all_m if m["equity_curve"]]
        return np.mean(ends) if ends else 1.0

    total_trades = sum(m["trades"] for m in all_m)
    avg_frac     = avg("avg_fraction")

    # Bruto: retorno total sobre capital en % (suma de equity_end por par)
    gross_returns = [(m["equity_curve"][-1] - 1.0) * 100 for m in all_m
                     if m["equity_curve"]]
    gross_pct     = np.mean(gross_returns) if gross_returns else 0.0

    # Coste real de comisiones por par
    n_pairs       = len(all_m)
    trades_pp     = total_trades / n_pairs if n_pairs else 0
    fee_cost_pct  = trades_pp * FEE_ROUNDTRIP * avg_frac * 100

    net_pct       = gross_pct - fee_cost_pct

    # Net Sharpe: reescala el Sharpe bruto por la fracción neta
    gross_sharpe  = avg("sharpe")
    if abs(gross_pct) > 1e-6:
        net_sharpe = gross_sharpe * (net_pct / gross_pct)
    else:
        net_sharpe = 0.0

    # Trades por mes (4h: 180 candles/mes)
    candles_total  = sum(len(df) for df in datasets.values())
    months         = candles_total / (180 * n_pairs) if n_pairs else 1
    t_month        = trades_pp / months if months > 0 else 0

    return {
        "name":         name,
        "net_sharpe":   round(net_sharpe, 3),
        "gross_sharpe": round(gross_sharpe, 3),
        "net_pct":      round(net_pct, 3),
        "gross_pct":    round(gross_pct, 3),
        "fee_cost_pct": round(fee_cost_pct, 3),
        "winrate":      round(avg("winrate") * 100, 1),
        "profit_factor":round(avg("profit_factor"), 2),
        "max_dd":       round(avg("max_drawdown") * 100, 1),
        "t_month":      round(t_month, 1),
        "n_trades":     total_trades,
        "avg_fraction": round(avg_frac * 100, 2),
    }


def run_tournament(datasets: dict) -> pd.DataFrame:
    configs = build_configs()
    print(f"\n{'='*65}")
    print(f"  FEE-AWARE TOURNAMENT — {len(configs)} configs × {len(datasets)} pares  ")
    print(f"  KuCoin taker fee: 0.1% × 2 = 0.2% round-trip")
    print(f"{'='*65}\n")

    results = []
    for i, (name, strat) in enumerate(configs, 1):
        t0  = time.time()
        row = _run_one(name, strat, datasets)
        if row:
            results.append(row)
        elapsed = time.time() - t0
        if i % 20 == 0:
            print(f"  [{i:3d}/{len(configs)}]  ⏱ {elapsed:.1f}s last batch")

    df = pd.DataFrame(results)
    df = df.sort_values("net_sharpe", ascending=False).reset_index(drop=True)
    return df


# ── Presentación ──────────────────────────────────────────────────────────────

def print_podium(df: pd.DataFrame):
    # Solo rentables
    profitable = df[df["net_pct"] > 0].copy()
    print(f"\n{'='*80}")
    print(f"  FEE-AWARE RANKING — Estrategias rentables después de comisiones")
    print(f"  ({len(profitable)}/{len(df)} configuraciones superan el umbral de fees)")
    print(f"{'='*80}")
    print(f"  {'#':>3}  {'Nombre':<42}  {'NetSharpe':>9}  {'Net%':>7}  {'Gross%':>7}  "
          f"{'Fees%':>6}  {'WR%':>5}  {'t/mes':>5}  {'MaxDD%':>6}")
    print(f"  {'-'*3}  {'-'*42}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*5}  "
          f"{'-'*5}  {'-'*6}")

    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    for rank, (_, row) in enumerate(profitable.head(15).iterrows()):
        medal = medals.get(rank, "  ")
        print(f"  {rank+1:>3}. {medal} {row['name']:<40}  "
              f"{row['net_sharpe']:>9.3f}  "
              f"{row['net_pct']:>+7.2f}%  "
              f"{row['gross_pct']:>+7.2f}%  "
              f"{row['fee_cost_pct']:>6.2f}%  "
              f"{row['winrate']:>5.1f}%  "
              f"{row['t_month']:>5.1f}  "
              f"{row['max_dd']:>6.1f}%")

    print(f"\n{'='*80}")
    print("  TOP 5 PARA TRADING REAL:")
    for i, (_, row) in enumerate(profitable.head(5).iterrows(), 1):
        gross_pt = row['gross_pct'] / max(row['t_month'] * (len(PAIRS_4H)), 1)
        print(f"  {i}. {row['name']}")
        print(f"     Net: {row['net_pct']:+.2f}%  Gross: {row['gross_pct']:+.2f}%  "
              f"Fees: {row['fee_cost_pct']:.2f}%  Trades/mes: {row['t_month']:.1f}")
    print(f"{'='*80}\n")


def save_plots(df: pd.DataFrame):
    profitable = df[df["net_pct"] > 0].head(10)
    if profitable.empty:
        print("  ⚠ No profitable strategies to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    names    = [n[:25] for n in profitable["name"]]
    colors   = ["#00e5ff" if v > 0 else "#ff4444" for v in profitable["net_pct"]]

    # Net% vs Gross%
    x = np.arange(len(names))
    w = 0.35
    axes[0, 0].bar(x - w/2, profitable["gross_pct"], w, label="Gross%", color="#4fc3f7", alpha=0.8)
    axes[0, 0].bar(x + w/2, profitable["net_pct"],   w, label="Net%",   color="#00e5ff", alpha=0.9)
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(names, rotation=35, ha="right", fontsize=8, color="white")
    axes[0, 0].set_ylabel("Return %", color="white")
    axes[0, 0].set_title("Gross vs Net Return (after fees)", color="white")
    axes[0, 0].legend(framealpha=0.3, labelcolor="white")
    axes[0, 0].axhline(0, color="white", linewidth=0.5, alpha=0.5)

    # Net Sharpe
    bar_colors = ["#00e5ff" if s > 0 else "#ff4444" for s in profitable["net_sharpe"]]
    axes[0, 1].bar(names, profitable["net_sharpe"], color=bar_colors, alpha=0.9)
    axes[0, 1].set_xticks(range(len(names))); axes[0, 1].set_xticklabels(names, rotation=35, ha="right", fontsize=8, color="white")
    axes[0, 1].set_ylabel("Net Sharpe", color="white")
    axes[0, 1].set_title("Net Sharpe (fee-adjusted)", color="white")
    axes[0, 1].axhline(0, color="white", linewidth=0.5, alpha=0.5)

    # t/month vs fee cost
    axes[1, 0].scatter(profitable["t_month"], profitable["fee_cost_pct"],
                       c=profitable["net_pct"], cmap="RdYlGn", s=120, alpha=0.9, edgecolors="white", linewidths=0.5)
    for _, r in profitable.iterrows():
        axes[1, 0].annotate(r["name"][:15], (r["t_month"], r["fee_cost_pct"]),
                            fontsize=6, color="white", alpha=0.8)
    axes[1, 0].set_xlabel("Trades / mes", color="white")
    axes[1, 0].set_ylabel("Fee cost %", color="white")
    axes[1, 0].set_title("Frecuencia vs Coste Comisiones", color="white")

    # WR vs MaxDD
    axes[1, 1].scatter(profitable["winrate"], profitable["max_dd"],
                       c=profitable["net_sharpe"], cmap="plasma", s=120, alpha=0.9, edgecolors="white", linewidths=0.5)
    for _, r in profitable.iterrows():
        axes[1, 1].annotate(r["name"][:15], (r["winrate"], r["max_dd"]),
                            fontsize=6, color="white", alpha=0.8)
    axes[1, 1].set_xlabel("Win Rate %", color="white")
    axes[1, 1].set_ylabel("Max Drawdown %", color="white")
    axes[1, 1].set_title("Win Rate vs Max Drawdown", color="white")

    fig.suptitle("Fee-Aware Tournament — 4H KuCoin Data", color="white", fontsize=14)
    plt.tight_layout()

    out = "data/fee_aware_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Plot guardado: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    from data.kucoin_client import KuCoinClient
    client = KuCoinClient()

    print("\n📡 Descargando datos 4H de KuCoin...")
    datasets = {}
    for pair in PAIRS_4H:
        try:
            df = client.get_ohlcv(pair, interval="4hour", limit=1500)
            if df is not None and len(df) > 200:
                datasets[pair] = df
                print(f"  ✅ {pair}: {len(df)} velas")
            else:
                print(f"  ⚠ {pair}: datos insuficientes, omitido")
        except Exception as e:
            print(f"  ❌ {pair}: {e}")
        time.sleep(0.3)

    if not datasets:
        print("❌ Sin datos. Verifica conexión a KuCoin.")
        return

    print(f"\n  Pares activos: {list(datasets.keys())}")

    # ── Correr torneo ─────────────────────────────────────────────────────
    df = run_tournament(datasets)

    # ── Guardar CSV ───────────────────────────────────────────────────────
    out_csv = "data/fee_aware_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"  💾 CSV guardado: {out_csv}")

    # ── Podium ────────────────────────────────────────────────────────────
    print_podium(df)

    # ── Plots ─────────────────────────────────────────────────────────────
    save_plots(df)

    # ── Recomendación final ───────────────────────────────────────────────
    profitable = df[df["net_pct"] > 0]
    print(f"\n{'='*65}")
    print(f"  RESUMEN EJECUTIVO")
    print(f"{'='*65}")
    print(f"  Configuraciones testeadas : {len(df)}")
    print(f"  Rentables post-comisiones : {len(profitable)}")
    if not profitable.empty:
        best = profitable.iloc[0]
        print(f"\n  ★ MEJOR ESTRATEGIA ★")
        print(f"  {best['name']}")
        print(f"  → Net Sharpe : {best['net_sharpe']:.3f}")
        print(f"  → Net Return : {best['net_pct']:+.2f}%  (sobre 250 días, 4H)")
        print(f"  → Gross      : {best['gross_pct']:+.2f}%")
        print(f"  → Fees       : {best['fee_cost_pct']:.2f}%")
        print(f"  → Win Rate   : {best['winrate']:.1f}%")
        print(f"  → Trades/mes : {best['t_month']:.1f}")
        print(f"  → Max DD     : {best['max_dd']:.1f}%")
    print(f"{'='*65}\n")

    return df


if __name__ == "__main__":
    main()
