"""
backtester_fast.py

Version O(n) del backtester: precomputa las señales para todo el dataset
de una sola pasada en lugar de llamar a detect_market_structure n veces.

Usa el mismo criterio de entrada/salida que Backtester pero es ~100x mas rapido.
"""

import numpy as np


def _precompute_signals(data, swing_window=5, require_fvg=False, use_choch_filter=True):
    """
    Calcula trend/bos/choch para cada vela en O(n * window).
    Devuelve una lista de dicts con la señal en cada posicion.
    """
    n = len(data)
    h = data["high"].values
    l = data["low"].values
    c = data["close"].values
    open_ = data["open"].values

    # --- Swing detection (una sola pasada) ---
    swing_highs = np.zeros(n, dtype=bool)
    swing_lows = np.zeros(n, dtype=bool)

    for i in range(swing_window, n - swing_window):
        if h[i] > h[i - swing_window:i].max() and h[i] >= h[i + 1:i + swing_window + 1].max():
            swing_highs[i] = True
        if l[i] < l[i - swing_window:i].min() and l[i] <= l[i + 1:i + swing_window + 1].min():
            swing_lows[i] = True

    # --- FVG detection (para require_fvg) ---
    bullish_fvg = np.zeros(n, dtype=bool)
    bearish_fvg = np.zeros(n, dtype=bool)
    if require_fvg:
        for i in range(2, n):
            if c[i] > h[i - 2]:
                bullish_fvg[i] = True
            elif open_[i] < l[i - 2]:
                bearish_fvg[i] = True

    # --- Generar señal por vela ---
    signals = ["hold"] * n
    sh_hist = []   # [(idx, price)]
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

        # Tendencia por secuencia de swings
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
            continue  # hold

        if trend == "bullish" and bos:
            if require_fvg and not bullish_fvg[i]:
                continue
            signals[i] = "buy"
        elif trend == "bearish" and bos:
            if require_fvg and not bearish_fvg[i]:
                continue
            signals[i] = "sell"

    return signals


def _run_trades(signals, closes, max_hold=10):
    """Simula trades sobre la lista de señales precomputadas."""
    trades = []
    position = None

    for i, sig in enumerate(signals):
        price = closes[i]

        if position is None:
            if sig in ("buy", "sell"):
                position = {"side": sig, "entry": price, "idx": i}
        else:
            held = i - position["idx"]
            opposing = (position["side"] == "buy" and sig == "sell") or \
                       (position["side"] == "sell" and sig == "buy")

            if opposing or held >= max_hold:
                entry = position["entry"]
                pnl = (price - entry) / entry if position["side"] == "buy" \
                      else (entry - price) / entry
                trades.append({"side": position["side"], "pnl": pnl, "win": pnl > 0})
                position = None
                if sig in ("buy", "sell"):
                    position = {"side": sig, "entry": price, "idx": i}

    # Cerrar posicion abierta al final
    if position is not None:
        price = closes[-1]
        entry = position["entry"]
        pnl = (price - entry) / entry if position["side"] == "buy" \
              else (entry - price) / entry
        trades.append({"side": position["side"], "pnl": pnl, "win": pnl > 0})

    return trades


def _metrics(trades, signals, data):
    from strategies.smc_strategy import SMCStrategy
    final_signal = {"signal": "hold", "reason": "precomputed"}

    if not trades:
        return {"trades": 0, "winrate": 0.0, "profit_factor": 0.0,
                "total_pnl": 0.0, "signal": final_signal}

    wins = [t for t in trades if t["win"]]
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

    return {
        "trades": len(trades),
        "winrate": round(winrate, 4),
        "profit_factor": round(profit_factor, 4),
        "total_pnl": round(sum(t["pnl"] for t in trades), 4),
        "signal": final_signal,
    }


class FastBacktester:
    """Backtester O(n) — precomputa señales en una sola pasada."""

    def __init__(self, strategy, data, max_hold=10):
        self.strategy = strategy
        self.data = data
        self.max_hold = max_hold

    def run(self):
        signals = _precompute_signals(
            self.data,
            swing_window=self.strategy.swing_window,
            require_fvg=self.strategy.require_fvg,
            use_choch_filter=self.strategy.use_choch_filter,
        )
        closes = self.data["close"].values
        trades = _run_trades(signals, closes, self.max_hold)
        return _metrics(trades, signals, self.data)
