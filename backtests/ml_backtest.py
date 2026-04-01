"""
ml_backtest.py

Backtest con autoaprendizaje ML:
  - Corre una estrategia base (SMC w=5 en 15m)
  - Cada trade completado alimenta la red neuronal con su resultado
  - La red aprende a filtrar señales malas en tiempo real
  - Compara: base vs base+ML_filter vs ScenarioRouter+ML

Muestra:
  - Evolución de accuracy del modelo a lo largo del año
  - Trades filtrados vs aceptados por mes
  - Mejora en WR y Sharpe gracias al filtro
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.feature_extractor import extract_features, N_FEATURES
from ml.neural_net import TradingNet
from ml.signal_filter import MLSignalFilter
from strategies.smc_strategy import SMCStrategy
from strategies.scenario_router import ScenarioRouter, precompute_router_signals_fast
from strategies.kelly_sizer import KellySizer
from backtests.backtester_fast import _precompute_signals
from backtests.kelly_backtest import _kelly_metrics
from backtests.metrics import compute_all


# ── Generador de datos ────────────────────────────────────────────────────────

BASE_SLOPES = {
    "BTC": 0.004, "ETH": 0.005, "SOL": 0.007,
    "BNB": 0.003, "XAUT": 0.002, "LINK": 0.006,
}

def _gen(name, n, seed=None):
    rng = np.random.default_rng(seed or hash(name) % (2**31))
    prices = [100.0]
    sl = BASE_SLOPES.get(name, 0.004)
    phase_len = n // 8
    patterns = ["bull", "sideways", "bull", "bear", "sideways", "bull", "bear", "bull"]
    phases = []
    for ph in patterns:
        v = sl if ph == "bull" else (-sl * 0.75 if ph == "bear" else 0.0)
        phases.extend([v] * phase_len)
    while len(phases) < n:
        phases.append(sl * 0.3)
    vol = 0.6
    for i in range(n - 1):
        shock = rng.standard_normal() * vol
        vol = np.clip(0.95 * vol + 0.05 * (abs(shock) * 0.5 + 0.3), 0.2, 2.5)
        prices.append(max(1.0, prices[-1] * (1 + phases[i] / 100 + shock / 100)))
    prices = np.array(prices)
    spread = abs(rng.standard_normal(n)) * prices * 0.002
    return pd.DataFrame({
        "open": prices,
        "high": prices + spread,
        "low":  np.maximum(prices - spread, prices * 0.98),
        "close": prices,
        "volume": rng.integers(5000, 80000, n).astype(float),
    })


# ── Simulador ML-aware ────────────────────────────────────────────────────────

def run_ml_backtest(data: pd.DataFrame, strategy,
                    sizer: KellySizer,
                    max_hold: int = 8,
                    use_ml: bool = True,
                    threshold: float = 0.52,
                    min_trades_train: int = 30):
    """
    Simula trades con aprendizaje online ML.
    En cada vela:
      1. Genera señal de la estrategia (precomputada)
      2. Si use_ml: filtra con la red neuronal
      3. Si se abre trade, guarda las features
      4. Cuando el trade cierra, alimenta la red con el resultado (label)
      5. La red actualiza sus pesos con partial_fit
    """
    closes = data["close"].values
    highs  = data["high"].values
    lows   = data["low"].values
    n      = len(closes)

    # Precomputa señales de la estrategia base
    if hasattr(strategy, "swing_window"):
        signals_raw = _precompute_signals(
            data,
            swing_window=strategy.swing_window,
            require_fvg=getattr(strategy, "require_fvg", False),
            use_choch_filter=getattr(strategy, "use_choch_filter", False),
        )
    else:
        signals_raw = ["hold"] * n

    # Modelo ML
    model  = TradingNet(input_dim=N_FEATURES)
    active = False
    history: list = []   # (features, label)

    equity_candle = np.ones(n)
    current_equity = 1.0
    trade_history: list = []
    position = None

    # Para análisis
    ml_log = []     # (candle_idx, accepted, prob, actual_win)
    filtered_count = 0
    accepted_count = 0

    def _close_trade(side, pnl_raw, win, exit_type):
        nonlocal current_equity
        frac = sizer.position_fraction(trade_history)
        scaled = pnl_raw * frac
        current_equity *= (1 + scaled)
        t = {"side": side, "pnl": pnl_raw, "scaled_pnl": scaled,
             "fraction": frac, "win": win, "exit": exit_type}
        trade_history.append({"pnl": pnl_raw, "win": win, "fraction": frac})
        return t

    for i, raw_sig in enumerate(signals_raw):
        price = closes[i]

        # ── SL/TP no implementado en esta versión simplificada ────────
        # ── Salida por señal opuesta o max_hold ───────────────────────
        if position is not None:
            held = i - position["idx"]
            opposing = (
                (position["side"] == "buy"  and raw_sig == "sell") or
                (position["side"] == "sell" and raw_sig == "buy")
            )
            if opposing or held >= max_hold:
                entry = position["entry"]
                pnl   = (price - entry) / entry if position["side"] == "buy" \
                        else (entry - price) / entry
                win   = pnl > 0
                t     = _close_trade(position["side"], pnl, win, "signal")

                # Aprender del resultado
                if use_ml and position.get("features") is not None:
                    feats = position["features"]
                    label = 1.0 if win else 0.0
                    history.append((feats, label))

                    # Online update
                    model.partial_fit(feats, label)

                    # Activar filtro cuando hay suficientes datos
                    if not active and len(history) >= min_trades_train:
                        active = True
                        # Batch retrain
                        X = np.array([h[0] for h in history], dtype=np.float32)
                        y = np.array([h[1] for h in history], dtype=np.float32)
                        model.fit(X, y, epochs=20, verbose=False)

                    # Log
                    ml_log.append({
                        "candle": i,
                        "month": i * 12 // n,
                        "accepted": True,
                        "prob": position.get("ml_prob", 0.5),
                        "win": win,
                    })

                position = None

        # ── Filtro ML + apertura de posición ──────────────────────────
        if position is None and raw_sig in ("buy", "sell"):
            # Usar ventana fija de 120 velas para evitar O(n²)
            win_start = max(0, i - 119)
            features = extract_features(data.iloc[win_start:i+1], raw_sig) if use_ml else None
            ml_prob  = None
            accepted = True

            if use_ml and active and features is not None:
                ml_prob  = float(model.predict_single(features))
                accepted = ml_prob >= threshold
                if not accepted:
                    filtered_count += 1

            if accepted:
                accepted_count += 1
                position = {
                    "side": raw_sig, "entry": price, "idx": i,
                    "features": features, "ml_prob": ml_prob,
                }

        equity_candle[i] = current_equity

    # Cerrar posición final
    if position is not None:
        price = closes[-1]
        entry = position["entry"]
        pnl   = (price - entry) / entry if position["side"] == "buy" \
                else (entry - price) / entry
        win   = pnl > 0
        _close_trade(position["side"], pnl, win, "end")
        equity_candle[n - 1] = current_equity

    metrics = _kelly_metrics(trade_history, equity_candle.tolist(), periods_per_year=35040)

    # Score final del modelo
    model_score = {}
    if history:
        X = np.array([h[0] for h in history], dtype=np.float32)
        y = np.array([h[1] for h in history], dtype=np.float32)
        model_score = model.score(X, y)

    return {
        **metrics,
        "ml_log":        ml_log,
        "filtered":      filtered_count,
        "accepted":      accepted_count,
        "model_score":   model_score,
        "history_size":  len(history),
        "model_active_from": min_trades_train,
    }


# ── Runner ────────────────────────────────────────────────────────────────────

PAIRS    = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XAUT/USDT", "LINK/USDT"]
N_15M    = 35_040
CAPITAL  = 1_000.0
MAX_HOLD = 8

def run_all():
    sizer    = KellySizer(variant="half_kelly", min_trades=20, max_fraction=0.30)
    strategy = SMCStrategy(swing_window=5)

    configs = [
        ("Sin ML (baseline)",      False, 0.50),
        ("ML threshold=48%",       True,  0.48),
        ("ML threshold=52%",       True,  0.52),
        ("ML threshold=55%",       True,  0.55),
    ]

    results   = []
    all_equity = {name: [] for name, _, __ in configs}
    all_ml_log = []

    print(f"\n{'='*80}")
    print(f"  BACKTEST CON AUTOAPRENDIZAJE ML — 6 pares × 1 año × 15m")
    print(f"  Modelo: MLP 25→32→16→1 | Online learning | Half-Kelly")
    print(f"{'='*80}")

    for pair in PAIRS:
        name = pair.split("/")[0]
        data = _gen(name, N_15M)

        print(f"\n  Par: {pair}")
        for config_name, use_ml, thresh in configs:
            m = run_ml_backtest(data, strategy, sizer,
                                max_hold=MAX_HOLD, use_ml=use_ml,
                                threshold=thresh)

            final_eq     = m["equity_curve"][-1] * CAPITAL
            pct          = (final_eq - CAPITAL) / CAPITAL * 100
            trades_month = m["trades"] / 12

            all_equity[config_name].append(m["equity_curve"])

            row = {
                "par":          pair,
                "config":       config_name,
                "trades":       m["trades"],
                "trades_month": round(trades_month, 1),
                "winrate":      round(m["winrate"] * 100, 1),
                "compuesto":    round(pct, 1),
                "final_eur":    round(final_eq, 1),
                "sharpe":       m["sharpe"],
                "sortino":      m["sortino"],
                "max_dd":       round(m["max_drawdown"] * 100, 1),
                "filtered":     m.get("filtered", 0),
                "accepted":     m.get("accepted", 0),
                "ml_accuracy":  m.get("model_score", {}).get("accuracy", None),
                "ml_f1":        m.get("model_score", {}).get("f1", None),
            }
            results.append(row)

            ml_info = ""
            if use_ml:
                ms = m.get("model_score", {})
                filt_pct = m["filtered"] / max(1, m["filtered"] + m["accepted"]) * 100
                ml_info = (f"  [ML acc={ms.get('accuracy', 0):.3f} "
                           f"f1={ms.get('f1', 0):.3f} "
                           f"filtradas={filt_pct:.0f}%]")

            print(f"    {config_name:28s}  T:{m['trades']:5d} (~{trades_month:5.1f}/mes)"
                  f"  WR:{m['winrate']*100:5.1f}%  Ret:{pct:+8.1f}%"
                  f"  Sharpe:{m['sharpe']:5.2f}  DD:{m['max_drawdown']*100:5.1f}%{ml_info}")

    df = pd.DataFrame(results)
    _print_summary(df)
    _save_plots(df, all_equity)
    df.to_csv("data/ml_backtest_results.csv", index=False)
    print("\n  Guardado: data/ml_backtest_results.csv")
    return df


def _print_summary(df):
    print(f"\n{'='*80}")
    print("  RESUMEN COMPARATIVO — media 6 pares")
    print(f"{'='*80}")
    print(f"  {'Config':30s}  {'T/mes':>6}  {'WR%':>6}  {'Ret%':>9}  "
          f"{'Sharpe':>7}  {'DD%':>6}  {'Filtradas%':>10}")
    print(f"  {'-'*80}")
    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg]
        filt = sub["filtered"].mean()
        acc  = sub["accepted"].mean()
        filt_pct = filt / max(1, filt + acc) * 100
        ml_acc_str = ""
        if sub["ml_accuracy"].notna().any():
            ml_acc_str = f"  ML_acc={sub['ml_accuracy'].mean():.3f}"
        print(f"  {cfg:30s}  "
              f"{sub['trades_month'].mean():6.1f}  "
              f"{sub['winrate'].mean():6.1f}%  "
              f"{sub['compuesto'].mean():+9.1f}%  "
              f"{sub['sharpe'].mean():7.2f}  "
              f"{sub['max_dd'].mean():5.1f}%  "
              f"{filt_pct:9.1f}%{ml_acc_str}")


def _save_plots(df, all_equity):
    colors = {"Sin ML (baseline)": "#4e79a7", "ML threshold=48%": "#59a14f",
              "ML threshold=52%": "#e15759", "ML threshold=55%": "#f28e2b"}
    pairs  = PAIRS

    # ── Equity curves: base vs ML ─────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Autoaprendizaje ML — Base vs Filtrado ML (15m, Half-Kelly)",
                 fontsize=14, fontweight="bold")

    for ax, pair, idx in zip(axes.flat, pairs, range(len(pairs))):
        for config, color in colors.items():
            if config in all_equity and idx < len(all_equity[config]):
                eq = all_equity[config][idx]
                x  = np.linspace(0, 12, len(eq))
                ax.plot(x, eq, color=color, linewidth=1.3, label=config, alpha=0.85)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(pair, fontweight="bold")
        ax.set_xlabel("Mes")
        ax.set_ylabel("Equity (×inicial)")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/plots/23_ml_equity_curves.png", dpi=130)
    plt.close()
    print("  Guardado: data/plots/23_ml_equity_curves.png")

    # ── Comparativa barras ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle("ML Filter — Impacto en métricas (media 6 pares)", fontsize=13, fontweight="bold")

    for ax, (col, title) in zip(axes, [
        ("compuesto",    "Retorno Anual (%)"),
        ("winrate",      "Win Rate (%)"),
        ("sharpe",       "Sharpe Ratio"),
        ("max_dd",       "Max Drawdown (%)"),
    ]):
        for i, cfg in enumerate(df["config"].unique()):
            sub  = df[df["config"] == cfg]
            val  = sub[col].mean()
            c    = list(colors.values())[i]
            bar  = ax.bar(i, val, color=c, alpha=0.85, edgecolor="white", width=0.6)
            ax.text(i, val + abs(val) * 0.02, f"{val:.2f}",
                    ha="center", fontsize=9, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Base", "Base+ML"], fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/plots/24_ml_metrics_comparison.png", dpi=130)
    plt.close()
    print("  Guardado: data/plots/24_ml_metrics_comparison.png")


if __name__ == "__main__":
    run_all()
