"""
backtester_fast.py

Version O(n) del backtester: precomputa señales en una sola pasada.

Soporta:
  - Salida por señal opuesta o max_hold velas (modo original)
  - Stop Loss + Take Profit via RiskManager (comprueba high/low por vela)
  - Metricas de riesgo: Sharpe, Sortino, Max Drawdown, Calmar (via metrics.py)
"""

import numpy as np
from backtests.metrics import compute_all


def _precompute_signals(data, swing_window=5, require_fvg=False, use_choch_filter=True):
    """Calcula trend/bos/choch para cada vela en O(n * window)."""
    n = len(data)
    h = data["high"].values
    l = data["low"].values
    c = data["close"].values
    open_ = data["open"].values

    swing_highs = np.zeros(n, dtype=bool)
    swing_lows = np.zeros(n, dtype=bool)

    for i in range(swing_window, n - swing_window):
        if h[i] > h[i - swing_window:i].max() and h[i] >= h[i + 1:i + swing_window + 1].max():
            swing_highs[i] = True
        if l[i] < l[i - swing_window:i].min() and l[i] <= l[i + 1:i + swing_window + 1].min():
            swing_lows[i] = True

    bullish_fvg = np.zeros(n, dtype=bool)
    bearish_fvg = np.zeros(n, dtype=bool)
    if require_fvg:
        for i in range(2, n):
            if c[i] > h[i - 2]:
                bullish_fvg[i] = True
            elif open_[i] < l[i - 2]:
                bearish_fvg[i] = True

    signals = ["hold"] * n
    sh_hist = []
    sl_hist = []

    for i in range(n):
        if swing_highs[i]:
            sh_hist.append(h[i])
        if swing_lows[i]:
            sl_hist.append(l[i])

        if len(sh_hist) < 2 or len(sl_hist) < 2:
            continue

        last_sh, prev_sh = sh_hist[-1], sh_hist[-2]
        last_sl, prev_sl = sl_hist[-1], sl_hist[-2]

        if last_sh > prev_sh and last_sl > prev_sl:
            trend = "bullish"
        elif last_sh < prev_sh and last_sl < prev_sl:
            trend = "bearish"
        else:
            trend = "neutral"

        bos = False
        choch = False

        if c[i] > last_sh:
            bos = True
            if trend == "bearish":
                choch = True
            trend = "bullish"
        elif c[i] < last_sl:
            bos = True
            if trend == "bullish":
                choch = True
            trend = "bearish"

        if use_choch_filter and choch:
            continue

        if trend == "bullish" and bos:
            if require_fvg and not bullish_fvg[i]:
                continue
            signals[i] = "buy"
        elif trend == "bearish" and bos:
            if require_fvg and not bearish_fvg[i]:
                continue
            signals[i] = "sell"

    return signals


def _run_trades(signals, data, max_hold=10, risk_manager=None):
    """
    Simula trades sobre señales precomputadas.

    Si risk_manager es None: salida por señal opuesta o max_hold velas.
    Si risk_manager esta definido: salida por SL/TP (usando high/low) o
    por señal opuesta/max_hold si ninguno se alcanza antes.

    Devuelve (trades, equity_curve) donde equity_curve comienza en 1.0
    y acumula multiplicativamente los PnL de cada trade.
    """
    closes = data["close"].values
    highs  = data["high"].values
    lows   = data["low"].values

    n_candles = len(closes)
    trades = []
    # Equity candle-indexed (una entrada por vela) para annualizacion correcta
    equity_candle = np.ones(n_candles)
    current_equity = 1.0
    position = None

    def _close_trade(side, pnl, win, exit_type):
        nonlocal current_equity
        trades.append({"side": side, "pnl": pnl, "win": win, "exit": exit_type})
        current_equity *= (1 + pnl)

    for i, sig in enumerate(signals):
        price = closes[i]

        # ── Comprobar SL/TP si hay posicion abierta ──────────────────
        if position is not None and risk_manager is not None:
            sl = position["sl"]
            tp = position["tp"]
            side = position["side"]
            entry = position["entry"]

            if side == "buy":
                if lows[i] <= sl:
                    _close_trade(side, (sl - entry) / entry, False, "sl")
                    position = None
                elif highs[i] >= tp:
                    _close_trade(side, (tp - entry) / entry, True, "tp")
                    position = None
            else:  # sell
                if highs[i] >= sl:
                    _close_trade(side, (entry - sl) / entry, False, "sl")
                    position = None
                elif lows[i] <= tp:
                    _close_trade(side, (entry - tp) / entry, True, "tp")
                    position = None

        # ── Salida por señal opuesta o max_hold ──────────────────────
        if position is not None:
            held = i - position["idx"]
            opposing = (position["side"] == "buy" and sig == "sell") or \
                       (position["side"] == "sell" and sig == "buy")

            if opposing or held >= max_hold:
                entry = position["entry"]
                pnl = (price - entry) / entry if position["side"] == "buy" \
                      else (entry - price) / entry
                _close_trade(position["side"], pnl, pnl > 0, "signal")
                position = None

        # ── Abrir nueva posicion ─────────────────────────────────────
        if position is None and sig in ("buy", "sell"):
            sl_price, tp_price = (None, None)
            if risk_manager is not None:
                window = data.iloc[: i + 1]
                sl_price, tp_price = risk_manager.calculate_levels(window, price, sig)
            position = {"side": sig, "entry": price, "idx": i,
                        "sl": sl_price, "tp": tp_price}

        # Registrar equity al cierre de cada vela (se propaga si no hay trade)
        equity_candle[i] = current_equity

    # Cerrar posicion abierta al final del dataset
    if position is not None:
        price = closes[-1]
        entry = position["entry"]
        pnl = (price - entry) / entry if position["side"] == "buy" \
              else (entry - price) / entry
        _close_trade(position["side"], pnl, pnl > 0, "end")
        equity_candle[n_candles - 1] = current_equity

    return trades, equity_candle.tolist()


def _metrics(trades, equity_curve, periods_per_year=252):
    final_signal = {"signal": "hold", "reason": "precomputed"}

    if not trades:
        return {"trades": 0, "winrate": 0.0, "profit_factor": 0.0,
                "total_pnl": 0.0, "sl_hits": 0, "tp_hits": 0,
                "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
                "calmar": 0.0, "equity_curve": equity_curve,
                "signal": final_signal}

    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    winrate = len(wins) / len(trades)
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    sl_hits = sum(1 for t in trades if t.get("exit") == "sl")
    tp_hits = sum(1 for t in trades if t.get("exit") == "tp")

    risk_metrics = compute_all(equity_curve, periods_per_year=periods_per_year)

    return {
        "trades":        len(trades),
        "winrate":       round(winrate, 4),
        "profit_factor": round(profit_factor, 4),
        "total_pnl":     round(sum(t["pnl"] for t in trades), 4),
        "sl_hits":       sl_hits,
        "tp_hits":       tp_hits,
        "sharpe":        risk_metrics["sharpe"],
        "sortino":       risk_metrics["sortino"],
        "max_drawdown":  risk_metrics["max_drawdown"],
        "calmar":        risk_metrics["calmar"],
        "equity_curve":  equity_curve,
        "signal":        final_signal,
    }


class FastBacktester:
    """Backtester O(n) con soporte opcional de SL/TP via RiskManager."""

    def __init__(self, strategy, data, max_hold=10, risk_manager=None):
        self.strategy = strategy
        self.data = data
        self.max_hold = max_hold
        self.risk_manager = risk_manager

    def run(self, periods_per_year=252):
        signals = _precompute_signals(
            self.data,
            swing_window=self.strategy.swing_window,
            require_fvg=self.strategy.require_fvg,
            use_choch_filter=self.strategy.use_choch_filter,
        )
        trades, equity_curve = _run_trades(
            signals, self.data, self.max_hold, self.risk_manager
        )
        return _metrics(trades, equity_curve, periods_per_year)
