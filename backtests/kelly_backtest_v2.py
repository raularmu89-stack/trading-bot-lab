"""
kelly_backtest_v2.py

Motor de backtest Kelly v2 — trailing stop + partial TP.

Mejoras sobre v1:
  1. Trailing stop ATR  — el stop sube (longs) / baja (shorts) con el precio,
     nunca retrocede. Captura más upside en tendencias fuertes.

  2. Partial TP — toma el 50% de la posición al llegar a 1R de beneficio
     y mueve el stop a breakeven. La mitad restante sigue con trailing.
     Resultado: más WR percibido + runners que capturan movimientos grandes.

  3. Pyramiding optional — añade al 50% de la posición cuando el precio
     se mueve 1R a nuestro favor (si pyramid=True).

Parámetros de salida por trade:
  exit: "sl" | "tp" | "partial_tp" | "trail" | "signal" | "end"

Uso:
    from backtests.kelly_backtest_v2 import run_kelly_v2, kelly_metrics_v2

    trades, equity = run_kelly_v2(
        signals, df, sizer,
        max_hold=24,
        risk_manager=rm,
        trailing_atr_mult=2.0,
        partial_tp=True,
        partial_ratio=0.5,
    )
"""

import numpy as np
import pandas as pd
from strategies.kelly_sizer import KellySizer


def _calc_atr(h, l, c, period: int = 14) -> np.ndarray:
    n = len(c)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period-1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


def run_kelly_v2(
    signals,
    data: pd.DataFrame,
    sizer: KellySizer,
    *,
    max_hold:           int   = 24,
    risk_manager              = None,
    trailing_atr_mult:  float = 2.0,    # 0 = desactivado
    partial_tp:         bool  = True,   # tomar 50% en 1R
    partial_ratio:      float = 0.50,   # fracción a cerrar en partial
    atr_period:         int   = 14,
):
    """
    Simulación Kelly v2 con trailing stop y partial TP.

    Retorna (trades, equity_curve).
    """
    closes = data["close"].values
    highs  = data["high"].values
    lows   = data["low"].values
    atrs   = _calc_atr(highs, lows, closes, atr_period)
    n      = len(closes)

    equity_candle   = np.ones(n)
    current_equity  = 1.0
    position        = None
    trade_history   = []
    trades          = []

    def _close(side, pnl_raw, win, exit_type, size_frac=1.0):
        nonlocal current_equity
        frac = sizer.position_fraction(trade_history) * size_frac
        scaled = pnl_raw * frac
        current_equity *= (1 + scaled)
        t = {
            "side": side, "pnl": pnl_raw, "scaled_pnl": scaled,
            "fraction": frac, "win": win, "exit": exit_type,
        }
        trades.append(t)
        trade_history.append({"pnl": pnl_raw, "win": win})
        return t

    for i, sig in enumerate(signals):
        price = closes[i]
        atr_i = atrs[i] if not np.isnan(atrs[i]) else (highs[i] - lows[i])

        # ── Gestión de posición abierta ───────────────────────────────
        if position is not None:
            side  = position["side"]
            entry = position["entry"]
            sl    = position["sl"]
            tp    = position["tp"]    # fixed TP (puede ser None)
            trail = position["trail"] # trailing stop level
            partial_done = position["partial_done"]

            # ── 1. Actualizar trailing stop ───────────────────────────
            if trailing_atr_mult > 0:
                if side == "buy":
                    new_trail = highs[i] - trailing_atr_mult * atr_i
                    if new_trail > trail:
                        position["trail"] = new_trail
                        trail = new_trail
                else:
                    new_trail = lows[i] + trailing_atr_mult * atr_i
                    if new_trail < trail:
                        position["trail"] = new_trail
                        trail = new_trail

            # Stop efectivo: máximo entre SL fijo y trailing
            if side == "buy":
                eff_sl = max(sl, trail) if trail is not None else sl
            else:
                eff_sl = min(sl, trail) if trail is not None else sl

            # ── 2. Partial TP en 1R ───────────────────────────────────
            if partial_tp and not partial_done and tp is not None:
                partial_price = None
                if side == "buy":
                    one_r = entry + (tp - entry) * partial_ratio
                    if highs[i] >= one_r:
                        partial_price = one_r
                else:
                    one_r = entry - (entry - tp) * partial_ratio
                    if lows[i] <= one_r:
                        partial_price = one_r

                if partial_price is not None:
                    pnl = (partial_price - entry) / entry if side == "buy" \
                          else (entry - partial_price) / entry
                    _close(side, pnl, True, "partial_tp", size_frac=partial_ratio)
                    position["partial_done"] = True
                    # Mover SL a breakeven para la parte restante
                    position["sl"] = entry
                    if side == "buy":
                        position["trail"] = max(position["trail"], entry)
                    else:
                        position["trail"] = min(position["trail"], entry) \
                            if position["trail"] is not None else entry
                    partial_done = True

            # ── 3. SL / trailing stop hit ─────────────────────────────
            hit_sl = False
            exit_price = None

            if eff_sl is not None:
                if side == "buy" and lows[i] <= eff_sl:
                    hit_sl = True
                    exit_price = eff_sl
                elif side == "sell" and highs[i] >= eff_sl:
                    hit_sl = True
                    exit_price = eff_sl

            if hit_sl and exit_price is not None:
                pnl = (exit_price - entry) / entry if side == "buy" \
                      else (entry - exit_price) / entry
                rem = 1.0 - (partial_ratio if partial_done else 0.0)
                _close(side, pnl, pnl > 0,
                       "trail" if trailing_atr_mult > 0 else "sl",
                       size_frac=rem)
                position = None

            # ── 4. Fixed TP hit ───────────────────────────────────────
            if position is not None and tp is not None:
                tp_hit = False
                if side == "buy" and highs[i] >= tp:
                    tp_hit = True
                elif side == "sell" and lows[i] <= tp:
                    tp_hit = True

                if tp_hit:
                    pnl = (tp - entry) / entry if side == "buy" \
                          else (entry - tp) / entry
                    rem = 1.0 - (partial_ratio if partial_done else 0.0)
                    _close(side, pnl, True, "tp", size_frac=rem)
                    position = None

        # ── Señal opuesta / max_hold ──────────────────────────────────
        if position is not None:
            held = i - position["idx"]
            opp  = (position["side"] == "buy"  and sig == "sell") or \
                   (position["side"] == "sell" and sig == "buy")
            if opp or held >= max_hold:
                entry = position["entry"]
                pnl   = (price - entry) / entry if position["side"] == "buy" \
                        else (entry - price) / entry
                rem = 1.0 - (partial_ratio if position["partial_done"] else 0.0)
                _close(position["side"], pnl, pnl > 0, "signal", size_frac=rem)
                position = None

        # ── Abrir nueva posición ──────────────────────────────────────
        if position is None and sig in ("buy", "sell"):
            sl_p, tp_p = None, None
            if risk_manager is not None:
                window = data.iloc[:i+1]
                sl_p, tp_p = risk_manager.calculate_levels(window, price, sig)

            # Trail inicial = SL
            if trailing_atr_mult > 0:
                trail_init = (price - trailing_atr_mult * atr_i) if sig == "buy" \
                             else (price + trailing_atr_mult * atr_i)
                # Usar el más conservador entre SL fijo y trail inicial
                if sl_p is not None:
                    trail_init = max(trail_init, sl_p) if sig == "buy" \
                                 else min(trail_init, sl_p)
            else:
                trail_init = sl_p

            position = {
                "side": sig, "entry": price, "idx": i,
                "sl": sl_p, "tp": tp_p,
                "trail": trail_init,
                "partial_done": False,
            }

        equity_candle[i] = current_equity

    # Cerrar posición abierta al final del backtest
    if position is not None:
        price = closes[-1]
        entry = position["entry"]
        pnl   = (price - entry) / entry if position["side"] == "buy" \
                else (entry - price) / entry
        rem = 1.0 - (partial_ratio if position["partial_done"] else 0.0)
        _close(position["side"], pnl, pnl > 0, "end", size_frac=rem)
        equity_candle[n-1] = current_equity

    return trades, equity_candle.tolist()


def kelly_metrics_v2(trades: list, equity_curve: list,
                     periods_per_year: int = 8760) -> dict:
    """Métricas completas para v2 (incluye desglose partial/trail/tp/sl)."""
    from backtests.metrics import compute_all

    if not trades:
        return {
            "trades": 0, "winrate": 0.0, "profit_factor": 0.0,
            "total_pnl": 0.0, "sl_hits": 0, "tp_hits": 0,
            "partial_tp_hits": 0, "trail_hits": 0,
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
            "calmar": 0.0, "equity_curve": equity_curve, "avg_fraction": 0.0,
        }

    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    gp     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))
    pf     = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    af     = sum(t["fraction"] for t in trades) / len(trades)

    rm = compute_all(equity_curve, periods_per_year=periods_per_year)

    return {
        "trades":          len(trades),
        "winrate":         round(len(wins) / len(trades), 4),
        "profit_factor":   round(pf, 4),
        "total_pnl":       round(sum(t["pnl"] for t in trades), 4),
        "sl_hits":         sum(1 for t in trades if t["exit"] in ("sl", "trail")),
        "tp_hits":         sum(1 for t in trades if t["exit"] == "tp"),
        "partial_tp_hits": sum(1 for t in trades if t["exit"] == "partial_tp"),
        "trail_hits":      sum(1 for t in trades if t["exit"] == "trail"),
        "sharpe":          rm["sharpe"],
        "sortino":         rm["sortino"],
        "max_drawdown":    rm["max_drawdown"],
        "calmar":          rm["calmar"],
        "equity_curve":    equity_curve,
        "avg_fraction":    round(af, 6),
    }
