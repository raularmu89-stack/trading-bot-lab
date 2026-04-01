"""
kelly_backtest.py

Backtester con Kelly Criterion dinámico.

Diferencia con FastBacktester:
  - Cada trade usa la fracción Kelly calculada sobre los trades anteriores
  - El capital crece/mengua proporcionalmente al tamaño Kelly
  - Soporte de SL/TP con RiskManager

Uso:
    from backtests.kelly_backtest import KellyBacktester
    from strategies.smc_strategy import SMCStrategy
    from strategies.kelly_sizer import KellySizer

    strategy = SMCStrategy(swing_window=50)
    sizer = KellySizer(variant="half_kelly", min_trades=20)
    bt = KellyBacktester(strategy, data, sizer=sizer, risk_manager=rm)
    result = bt.run()
"""

import numpy as np
from backtests.backtester_fast import _precompute_signals
from backtests.metrics import compute_all
from strategies.kelly_sizer import KellySizer


def _run_kelly_trades(signals, data, sizer, max_hold=10, risk_manager=None):
    """
    Simula trades con tamaño dinámico Kelly.

    El PnL de cada trade se escala por la fracción Kelly calculada con
    el historial hasta ese momento:
        equity *= (1 + pnl_raw * kelly_fraction)

    Devuelve (trades, equity_curve)
    """
    closes = data["close"].values
    highs  = data["high"].values
    lows   = data["low"].values
    n      = len(closes)

    equity_candle = np.ones(n)
    current_equity = 1.0
    position = None
    trade_history = []   # historial acumulado de {pnl, win}

    def _close_trade(side, pnl_raw, win, exit_type, entry_price):
        nonlocal current_equity
        # Calcular fracción Kelly con historial hasta ANTES de este trade
        frac = sizer.position_fraction(trade_history)
        scaled_pnl = pnl_raw * frac
        current_equity *= (1 + scaled_pnl)
        trade = {
            "side": side, "pnl": pnl_raw, "scaled_pnl": scaled_pnl,
            "fraction": frac, "win": win, "exit": exit_type
        }
        trades.append(trade)
        trade_history.append({"pnl": pnl_raw, "win": win})

    trades = []

    for i, sig in enumerate(signals):
        price = closes[i]

        # ── SL/TP sobre posicion abierta ──────────────────────────────
        if position is not None and risk_manager is not None:
            sl    = position["sl"]
            tp    = position["tp"]
            side  = position["side"]
            entry = position["entry"]

            if side == "buy":
                if lows[i] <= sl:
                    pnl = (sl - entry) / entry
                    _close_trade(side, pnl, False, "sl", entry)
                    position = None
                elif highs[i] >= tp:
                    pnl = (tp - entry) / entry
                    _close_trade(side, pnl, True, "tp", entry)
                    position = None
            else:
                if highs[i] >= sl:
                    pnl = (entry - sl) / entry
                    _close_trade(side, pnl, False, "sl", entry)
                    position = None
                elif lows[i] <= tp:
                    pnl = (entry - tp) / entry
                    _close_trade(side, pnl, True, "tp", entry)
                    position = None

        # ── Salida por señal opuesta o max_hold ───────────────────────
        if position is not None:
            held = i - position["idx"]
            opposing = (position["side"] == "buy"  and sig == "sell") or \
                       (position["side"] == "sell" and sig == "buy")
            if opposing or held >= max_hold:
                entry = position["entry"]
                pnl   = (price - entry) / entry if position["side"] == "buy" \
                        else (entry - price) / entry
                _close_trade(position["side"], pnl, pnl > 0, "signal", entry)
                position = None

        # ── Abrir nueva posicion ──────────────────────────────────────
        if position is None and sig in ("buy", "sell"):
            sl_p, tp_p = None, None
            if risk_manager is not None:
                window = data.iloc[: i + 1]
                sl_p, tp_p = risk_manager.calculate_levels(window, price, sig)
            position = {"side": sig, "entry": price, "idx": i, "sl": sl_p, "tp": tp_p}

        equity_candle[i] = current_equity

    # Cerrar posición abierta al final
    if position is not None:
        price = closes[-1]
        entry = position["entry"]
        pnl   = (price - entry) / entry if position["side"] == "buy" \
                else (entry - price) / entry
        _close_trade(position["side"], pnl, pnl > 0, "end", entry)
        equity_candle[n - 1] = current_equity

    return trades, equity_candle.tolist()


def _kelly_metrics(trades, equity_curve, periods_per_year=252):
    if not trades:
        return {
            "trades": 0, "winrate": 0.0, "profit_factor": 0.0,
            "total_pnl": 0.0, "sl_hits": 0, "tp_hits": 0,
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
            "calmar": 0.0, "equity_curve": equity_curve,
            "avg_fraction": 0.0,
        }

    wins       = [t for t in trades if t["win"]]
    losses     = [t for t in trades if not t["win"]]
    winrate    = len(wins) / len(trades)
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss   = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 \
                    else (float("inf") if gross_profit > 0 else 0.0)

    sl_hits = sum(1 for t in trades if t.get("exit") == "sl")
    tp_hits = sum(1 for t in trades if t.get("exit") == "tp")
    avg_frac = sum(t["fraction"] for t in trades) / len(trades)

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
        "avg_fraction":  round(avg_frac, 6),
    }


class KellyBacktester:
    """
    Backtester con Kelly Criterion dinámico.

    Parámetros
    ----------
    strategy    : estrategia con atributos swing_window, require_fvg, use_choch_filter
    data        : DataFrame OHLCV
    sizer       : instancia de KellySizer (default: half_kelly, min_trades=20)
    max_hold    : velas máximas por trade
    risk_manager: RiskManager para SL/TP automáticos
    """

    def __init__(self, strategy, data, sizer=None, max_hold=10, risk_manager=None):
        self.strategy     = strategy
        self.data         = data
        self.sizer        = sizer or KellySizer(variant="half_kelly", min_trades=20)
        self.max_hold     = max_hold
        self.risk_manager = risk_manager

    def run(self, periods_per_year=252):
        signals = _precompute_signals(
            self.data,
            swing_window=self.strategy.swing_window,
            require_fvg=self.strategy.require_fvg,
            use_choch_filter=self.strategy.use_choch_filter,
        )
        trades, equity_curve = _run_kelly_trades(
            signals, self.data, self.sizer, self.max_hold, self.risk_manager
        )
        return _kelly_metrics(trades, equity_curve, periods_per_year)
