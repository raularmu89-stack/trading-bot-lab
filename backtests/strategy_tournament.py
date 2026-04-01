"""
strategy_tournament.py

Torneo de estrategias: genera ~100 configuraciones, backtest con datos reales
de KuCoin y selecciona las mejores.

Uso:
    python backtests/strategy_tournament.py
    # → imprime ranking y guarda CSV + plot

Funciones:
    build_configs()          → lista de 100 (nombre, instancia_estrategia)
    run_tournament(datasets) → DataFrame con resultados ordenados
    print_podium(df)         → top 15 en consola
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtests.backtester_fast import _precompute_signals
from backtests.kelly_backtest  import _run_kelly_trades, _kelly_metrics
from strategies.kelly_sizer    import KellySizer

from strategies.strategy_zoo import (
    EMACrossStrategy, TripleEMAStrategy, RSIDivergenceStrategy,
    StochasticStrategy, CCIStrategy, DonchianBreakoutStrategy,
    PinBarStrategy, EngulfingStrategy, SupertrendStrategy, ADXDIStrategy,
)
# Estrategias previas
from strategies.smc_strategy      import SMCStrategy
from strategies.macd_divergence   import MACDDivergenceStrategy
from strategies.bollinger_squeeze import BollingerSqueezeStrategy
from strategies.momentum_burst    import MomentumBurstStrategy
from strategies.trend_rider       import TrendRiderStrategy
from strategies.range_scalper     import RangeScalperStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.mean_reversion    import MeanReversionStrategy


# ── Catálogo de 100+ configuraciones ─────────────────────────────────────────

def build_configs() -> list:
    """Devuelve lista de (nombre, estrategia) con ~100 configs."""
    configs = []

    def add(name, strat):
        configs.append((name, strat))

    # ── EMA Cross (10 variantes) ──────────────────────────────────────────
    for fast, slow, trend in [(5,13,50),(5,21,50),(8,21,50),(9,21,55),(9,26,50),
                               (8,34,89),(5,13,34),(3,9,21),(13,34,89),(5,20,50)]:
        add(f"EMACross({fast},{slow},{trend})",
            EMACrossStrategy(fast=fast, slow=slow, trend=trend))

    # ── Triple EMA (8 variantes) ──────────────────────────────────────────
    for e1,e2,e3,pb in [(5,13,34,0.002),(8,21,55,0.003),(8,21,55,0.005),
                         (5,21,50,0.003),(10,25,60,0.004),(8,34,89,0.003),
                         (5,15,40,0.002),(10,30,80,0.005)]:
        add(f"TripleEMA({e1},{e2},{e3})",
            TripleEMAStrategy(e1=e1, e2=e2, e3=e3, pullback_pct=pb))

    # ── RSI Divergence (8 variantes) ─────────────────────────────────────
    for rp,lb,os,ob in [(14,5,35,65),(14,8,35,65),(14,10,30,70),(14,5,40,60),
                         (21,8,35,65),(10,5,35,65),(14,8,40,60),(7,5,35,65)]:
        add(f"RSIDiv(p{rp},lb{lb},os{os})",
            RSIDivergenceStrategy(rsi_period=rp, lookback=lb, oversold=os, overbought=ob))

    # ── Stochastic (8 variantes) ──────────────────────────────────────────
    for k,d,os,ob in [(14,3,20,80),(14,3,25,75),(9,3,20,80),(5,3,20,80),
                       (14,5,20,80),(21,3,20,80),(14,3,30,70),(9,5,25,75)]:
        add(f"Stoch(k{k},d{d},os{os})",
            StochasticStrategy(k_period=k, d_period=d, oversold=os, overbought=ob))

    # ── CCI (8 variantes) ────────────────────────────────────────────────
    for p,thr,et in [(20,100,50),(20,150,50),(14,100,50),(20,100,100),
                      (10,100,30),(20,80,50),(30,100,50),(14,150,100)]:
        add(f"CCI(p{p},thr{thr})",
            CCIStrategy(period=p, threshold=thr, ema_trend=et))

    # ── Donchian (8 variantes) ────────────────────────────────────────────
    for p,ep,am in [(20,10,0.5),(20,10,1.0),(20,10,0.0),(30,15,0.5),
                     (15,7,0.5),(40,20,0.5),(20,5,1.0),(50,25,0.5)]:
        add(f"Donchian(p{p},ep{ep})",
            DonchianBreakoutStrategy(period=p, exit_period=ep, atr_mult=am))

    # ── Pin Bar (8 variantes) ─────────────────────────────────────────────
    for wr,bp,et in [(2.0,0.4,50),(2.5,0.35,50),(3.0,0.3,50),(2.0,0.35,100),
                      (2.5,0.4,30),(2.0,0.4,100),(3.0,0.35,50),(2.0,0.3,50)]:
        add(f"PinBar(wr{wr},et{et})",
            PinBarStrategy(wick_ratio=wr, body_pct=bp, ema_trend=et))

    # ── Engulfing (8 variantes) ───────────────────────────────────────────
    for et,bm,ros,rob in [(50,1.0,45,55),(50,1.2,45,55),(50,1.5,45,55),
                           (100,1.2,45,55),(30,1.2,45,55),(50,1.2,40,60),
                           (50,1.0,40,60),(200,1.2,45,55)]:
        add(f"Engulfing(et{et},bm{bm})",
            EngulfingStrategy(ema_trend=et, body_mult=bm, rsi_os=ros, rsi_ob=rob))

    # ── Supertrend (8 variantes) ──────────────────────────────────────────
    for ap,mult in [(7,2.0),(7,3.0),(10,2.0),(10,3.0),
                    (14,2.0),(14,3.0),(10,2.5),(7,2.5)]:
        add(f"Supertrend(atr{ap},m{mult})",
            SupertrendStrategy(atr_period=ap, multiplier=mult))

    # ── ADX/DI (8 variantes) ─────────────────────────────────────────────
    for p,thr in [(14,20),(14,25),(14,15),(14,30),
                  (21,20),(7,20),(14,18),(21,25)]:
        add(f"ADXDI(p{p},thr{thr})",
            ADXDIStrategy(period=p, adx_threshold=thr))

    # ── Estrategias previas re-parametrizadas (12 variantes extra) ────────
    for sw in [3, 5, 8]:
        add(f"SMC(sw{sw})", SMCStrategy(swing_window=sw))
    add("MACDDiv(lb3)",  MACDDivergenceStrategy(lookback=3))
    add("MACDDiv(lb5)",  MACDDivergenceStrategy(lookback=5))
    add("BollSqz(cb1)",  BollingerSqueezeStrategy(confirm_bars=1))
    add("MomBurst(cr1.2,bm0.5)", MomentumBurstStrategy(compression_ratio=1.2, burst_mult=0.5,
                                                         rsi_bull_min=30, rsi_bear_max=70))
    add("TrendRider(sw5)",  TrendRiderStrategy(swing_window=5))
    add("RangeScalper(sw5)",RangeScalperStrategy(swing_window=5))
    add("Breakout(sw5)",    BreakoutStrategy(swing_window=5))
    add("MeanRev(sw5)",     MeanReversionStrategy(swing_window=5))
    add("MACDDiv(lb8,cc)",  MACDDivergenceStrategy(lookback=8, cruce_confirm=True))

    return configs


# ── Backtest de una estrategia sobre un dataset ───────────────────────────────

def _backtest_one(strategy, data: pd.DataFrame,
                  max_hold: int = 8, ppy: int = 8760,
                  kelly: str = "half_kelly") -> dict:
    """Ejecuta backtest de una estrategia y devuelve métricas."""
    sizer = KellySizer(variant=kelly, min_trades=20)
    n     = len(data)

    # Estrategias con batch son muy rápidas
    if hasattr(strategy, "generate_signals_batch"):
        try:
            signals = strategy.generate_signals_batch(data)
        except Exception:
            signals = ["hold"] * n
    else:
        # Fallback: señal vela a vela con ventana deslizante
        signals  = ["hold"] * n
        min_bars = getattr(strategy, "_min_bars", 60)
        for i in range(min_bars, n):
            window = data.iloc[max(0, i - 199): i + 1]
            try:
                r = strategy.generate_signal(window)
                signals[i] = r.get("signal", "hold")
            except Exception:
                pass

    trades, equity = _run_kelly_trades(signals, data, sizer, max_hold=max_hold)
    return _kelly_metrics(trades, equity, periods_per_year=ppy)


# ── Torneo ────────────────────────────────────────────────────────────────────

def run_tournament(datasets: dict,
                   max_hold: int = 8,
                   ppy: int = 8760,
                   kelly: str = "half_kelly",
                   objective: str = "sharpe",
                   verbose: bool = True) -> pd.DataFrame:
    """
    Ejecuta el torneo completo.

    datasets : {symbol: DataFrame}  — datos OHLCV reales
    Retorna  : DataFrame con resultados ordenados por objective
    """
    configs  = build_configs()
    n_config = len(configs)
    n_pairs  = len(datasets)

    if verbose:
        print(f"\n{'='*72}")
        print(f"  TORNEO DE ESTRATEGIAS")
        print(f"  {n_config} configuraciones × {n_pairs} pares = {n_config*n_pairs} backtests")
        print(f"  Objetivo: {objective}  |  Kelly: {kelly}  |  max_hold: {max_hold}")
        print(f"{'='*72}\n")

    records = []
    t0      = time.time()

    for idx, (name, strategy) in enumerate(configs):
        sharpes, wrs, rets, tms, dds, n_trades = [], [], [], [], [], []

        for sym, df in datasets.items():
            try:
                m = _backtest_one(strategy, df, max_hold=max_hold, ppy=ppy, kelly=kelly)
                if m["trades"] > 0:
                    sharpes.append(m["sharpe"])
                    wrs.append(m["winrate"] * 100)
                    rets.append((m["equity_curve"][-1] - 1) * 100)
                    candles_per_month = ppy / 12
                    tms.append(m["trades"] / (len(df) / candles_per_month))
                    dds.append(m["max_drawdown"] * 100)
                    n_trades.append(m["trades"])
            except Exception:
                pass

        record = {
            "name":     name,
            "sharpe":   round(np.mean(sharpes)  if sharpes  else 0.0, 3),
            "winrate":  round(np.mean(wrs)       if wrs      else 0.0, 2),
            "return":   round(np.mean(rets)      if rets     else 0.0, 2),
            "t_month":  round(np.mean(tms)       if tms      else 0.0, 1),
            "max_dd":   round(np.mean(dds)       if dds      else 0.0, 2),
            "n_trades": sum(n_trades),
            "n_pairs":  len(sharpes),
        }
        records.append(record)

        if verbose and (idx % max(1, n_config // 10) == 0 or idx == n_config - 1):
            elapsed  = time.time() - t0
            pct      = (idx + 1) / n_config * 100
            best_sh  = max(r["sharpe"] for r in records) if records else 0
            print(f"  [{pct:5.1f}%]  t={elapsed:5.1f}s  {idx+1}/{n_config}"
                  f"  mejor_sharpe={best_sh:.3f}")

    df = pd.DataFrame(records)
    df = df.sort_values(objective, ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ── Visualización ─────────────────────────────────────────────────────────────

def print_podium(df: pd.DataFrame, top_n: int = 15):
    print(f"\n{'='*88}")
    print(f"  🏆  TOP {top_n} ESTRATEGIAS DEL TORNEO")
    print(f"{'='*88}")
    print(f"  {'#':>3}  {'Estrategia':<36} {'Sharpe':>7} {'WR%':>6} "
          f"{'Ret%':>8} {'T/mes':>6} {'MaxDD':>7} {'Trades':>7}")
    print(f"  {'-'*84}")
    medals = {1:"🥇", 2:"🥈", 3:"🥉"}
    for _, row in df.head(top_n).iterrows():
        r    = int(row["rank"])
        med  = medals.get(r, "  ")
        name = row["name"][:34]
        print(f"  {med}{r:>2}  {name:<36} {row['sharpe']:>7.3f} "
              f"{row['winrate']:>5.1f}%  {row['return']:>+7.2f}%  "
              f"{row['t_month']:>5.0f}  {row['max_dd']:>+6.2f}%  {row['n_trades']:>6}")
    print()


def save_plots(df: pd.DataFrame,
               path: str = "data/plots/26_strategy_tournament.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    top20 = df.head(20)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Strategy Tournament — Top 20", fontsize=13, fontweight="bold")

    # Sharpe
    axes[0].barh(top20["name"][::-1], top20["sharpe"][::-1], color="#3fb950")
    axes[0].set_xlabel("Sharpe Ratio"); axes[0].set_title("Sharpe")
    axes[0].tick_params(axis="y", labelsize=7)

    # WinRate
    axes[1].barh(top20["name"][::-1], top20["winrate"][::-1], color="#58a6ff")
    axes[1].set_xlabel("WinRate %"); axes[1].set_title("Win Rate")
    axes[1].axvline(50, color="red", linestyle="--", alpha=0.5)
    axes[1].tick_params(axis="y", labelsize=7)

    # Return
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in top20["return"][::-1]]
    axes[2].barh(top20["name"][::-1], top20["return"][::-1], color=colors)
    axes[2].set_xlabel("Retorno %"); axes[2].set_title("Retorno")
    axes[2].tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico guardado: {path}")


# ── Runner standalone ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.kucoin_client import KuCoinClient

    # Timeframe: 1h → más señales que 4h
    INTERVAL = "1hour"
    PPY      = 8_760
    PAIRS    = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "LINK-USDT"]

    print("Descargando datos reales de KuCoin...")
    client   = KuCoinClient(verbose=False)
    datasets = {}
    for sym in PAIRS:
        df = client.get_ohlcv(sym, interval=INTERVAL, limit=1500)
        if df is not None and len(df) > 100:
            datasets[sym] = df
            print(f"  {sym}: {len(df)} velas  ({df.index[0].date()} → {df.index[-1].date()})")

    results = run_tournament(datasets, max_hold=8, ppy=PPY, verbose=True)
    print_podium(results, top_n=15)

    # Guardar
    results.to_csv("data/tournament_results.csv", index=False)
    print("  CSV guardado: data/tournament_results.csv")
    save_plots(results)

    # Mejores parámetros
    best = results.iloc[0]
    print(f"\n  GANADOR: {best['name']}")
    print(f"  Sharpe={best['sharpe']:.3f}  WR={best['winrate']:.1f}%  "
          f"Ret={best['return']:+.2f}%  Trades/mes={best['t_month']:.0f}")
