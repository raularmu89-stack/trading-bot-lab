"""
backtester.py

Backtester con modelo de ejecución realista.

Modelo de ejecución:
  - Señal generada en close[i]
  - Entrada en open[i+1] — elimina el look-ahead bias del bloque anterior
  - SL/TP fijados desde el precio de entrada (ATR de la vela de señal)
  - SL/TP comprobados en high/low de cada vela siguiente
  - Si SL y TP caen en la misma vela, SL tiene prioridad (conservador)
  - Salida por señal opuesta: close[i] con slippage
  - Slippage adverso en SL (market order); no en TP (limit order)
  - Comisiones round-trip: 2 × fee_rate

Nota sobre total_pnl:
  Es la suma aditiva de pnl_net por trade, no el retorno compuesto.
  Para retorno compuesto real: equity_curve[-1] - 1.
  Las métricas de riesgo (Sharpe, drawdown, Calmar) usan la equity curve
  compuesta, no total_pnl.

Interfaz pública estable: Backtester(strategy, data, ...).run() → dict
"""

import numpy as np
from backtests.metrics import compute_all


class Backtester:
    """
    Parámetros
    ----------
    strategy         : objeto con .generate_signal(window) → {"signal": str}
    data             : DataFrame con columnas open/high/low/close/volume
    sl_atr_mult      : multiplicador ATR para stop-loss  (default 2.0)
    tp_atr_mult      : multiplicador ATR para take-profit (default 4.0 → RR 1:2)
    atr_period       : periodo ATR Wilder (default 14)
    fee_rate         : comisión por lado como fracción (default 0.001 = 0.1%)
    slippage_pct     : slippage sobre el precio, aplicado adversamente
                       en entry y SL exits (default 0.0)
    max_hold         : velas máximas como timeout de último recurso
    min_candles      : velas mínimas antes de generar señales
    periods_per_year : para anualizaciones (252=diario, 8760=1h crypto)
    verbose          : imprimir resumen al finalizar
    """

    def __init__(
        self,
        strategy,
        data,
        sl_atr_mult      = 2.0,
        tp_atr_mult      = 4.0,
        atr_period       = 14,
        fee_rate         = 0.001,
        slippage_pct     = 0.0,
        max_hold         = 20,
        min_candles      = 20,
        periods_per_year = 252,
        verbose          = True,
    ):
        self.strategy        = strategy
        self.data            = data
        self.sl_atr_mult     = sl_atr_mult
        self.tp_atr_mult     = tp_atr_mult
        self.atr_period      = atr_period
        self.fee_rate        = fee_rate
        self.slippage_pct    = slippage_pct
        self.max_hold        = max_hold
        self.min_candles     = min_candles
        self.periods_per_year = periods_per_year
        self.verbose         = verbose

    # ── ATR Wilder smoothing ──────────────────────────────────────────────

    def _compute_atr(self):
        h = self.data["high"].values.astype(float)
        l = self.data["low"].values.astype(float)
        c = self.data["close"].values.astype(float)
        n = len(c)
        p = self.atr_period

        tr = np.zeros(n)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))

        atr = np.full(n, np.nan)
        if n >= p:
            atr[p - 1] = tr[:p].mean()       # seed: media simple estándar
            for i in range(p, n):
                atr[i] = (atr[i-1] * (p - 1) + tr[i]) / p   # Wilder: α=1/p
        return atr

    # ── SL/TP check intra-vela ─────────────────────────────────────────────

    def _check_sl_tp(self, position, high, low):
        """
        Detecta si SL o TP fue tocado durante la vela usando high/low.

        Cuando ambos niveles caen dentro del rango de la vela (gap extremo
        o alta volatilidad), SL tiene prioridad sobre TP.
        Esto es conservador: asume el peor orden de ejecución posible.
        """
        sl = position["sl"]
        tp = position["tp"]

        if position["side"] == "buy":
            sl_hit = low <= sl
            tp_hit = high >= tp
        else:
            sl_hit = high >= sl
            tp_hit = low <= tp

        # SL comprobado primero — conservador, evita sobreestimar PnL
        if sl_hit:
            return "sl", sl
        if tp_hit:
            return "tp", tp
        return None, None

    # ── Cierre de posición ────────────────────────────────────────────────

    def _close(self, position, exit_price, reason, exit_idx):
        entry = position["entry_price"]
        side  = position["side"]

        raw_pnl   = (exit_price - entry) / entry if side == "buy" \
                    else (entry - exit_price) / entry
        fee_total = 2 * self.fee_rate        # entrada + salida, un lado cada una
        pnl_net   = raw_pnl - fee_total

        return {
            "side":        side,
            "entry_price": round(entry, 8),
            "exit_price":  round(exit_price, 8),
            "entry_idx":   position["entry_idx"],
            "exit_idx":    exit_idx,
            "exit_reason": reason,
            "raw_pnl":     round(raw_pnl, 6),
            "fee":         round(fee_total, 6),
            "pnl_net":     round(pnl_net, 6),
            "pnl":         round(pnl_net, 6),   # alias compatibilidad
            "win":         pnl_net > 0,
        }

    # ── Loop principal ────────────────────────────────────────────────────

    def run(self):
        atr    = self._compute_atr()
        opens  = self.data["open"].values.astype(float)
        closes = self.data["close"].values.astype(float)
        highs  = self.data["high"].values.astype(float)
        lows   = self.data["low"].values.astype(float)
        n      = len(closes)

        trades   = []
        position = None
        capital  = 1.0
        equity   = []

        # pending: señal confirmada en close[i], ejecutar en open[i+1]
        # formato: (side: str, atr_value: float)
        pending  = None

        for i in range(n):
            equity.append(capital)

            # ── 0. Ejecutar entrada pendiente al open de esta vela ─────────
            # La señal se generó en close[i-1]; ahora entramos en open[i].
            # SL/TP se calculan desde el precio de entrada real (open[i]),
            # usando el ATR de la vela de señal (conservado en pending).
            if pending is not None and position is None:
                side_p, atr_p = pending
                pending = None

                entry = opens[i]
                slip  = entry * self.slippage_pct

                if side_p == "buy":
                    entry += slip
                    sl = entry - self.sl_atr_mult * atr_p
                    tp = entry + self.tp_atr_mult * atr_p
                else:
                    entry -= slip
                    sl = entry + self.sl_atr_mult * atr_p
                    tp = entry - self.tp_atr_mult * atr_p

                position = {
                    "side":        side_p,
                    "entry_price": entry,
                    "entry_idx":   i,
                    "sl":          sl,
                    "tp":          tp,
                }

            if i < self.min_candles:
                continue

            # ── 1. Gestionar posición abierta ──────────────────────────────
            if position is not None:
                reason, exit_px = self._check_sl_tp(position, highs[i], lows[i])

                # Timeout: último recurso, no estrategia de salida
                if reason is None and (i - position["entry_idx"]) >= self.max_hold:
                    reason  = "max_hold"
                    exit_px = closes[i]

                if reason is not None:
                    # Slippage adverso solo en SL (market order fill peor que el nivel).
                    # TP es limit order: se llena al precio pedido o mejor.
                    if reason == "sl":
                        slip = exit_px * self.slippage_pct
                        exit_px = exit_px - slip if position["side"] == "buy" \
                                  else exit_px + slip

                    trade    = self._close(position, exit_px, reason, i)
                    capital *= (1 + trade["pnl_net"])
                    equity[-1] = capital
                    trades.append(trade)
                    position = None

            # ── 2. Señal en close[i] ────────────────────────────────────────
            window  = self.data.iloc[: i + 1]
            sig     = self.strategy.generate_signal(window)
            sig_val = sig.get("signal", "hold")

            # ── 3. Señal opuesta → cerrar posición a close[i] ──────────────
            if position is not None:
                opp = (position["side"] == "buy"  and sig_val == "sell") or \
                      (position["side"] == "sell" and sig_val == "buy")
                if opp:
                    slip    = closes[i] * self.slippage_pct
                    exit_px = closes[i] - slip if position["side"] == "buy" \
                              else closes[i] + slip
                    trade    = self._close(position, exit_px, "signal", i)
                    capital *= (1 + trade["pnl_net"])
                    equity[-1] = capital
                    trades.append(trade)
                    position = None
                    # Encolar la señal nueva para abrir en open[i+1]
                    a = atr[i]
                    if not np.isnan(a) and a > 0:
                        pending = (sig_val, a)

            # ── 4. Sin posición → encolar señal para open[i+1] ─────────────
            if position is None and pending is None and sig_val in ("buy", "sell"):
                a = atr[i]
                if not np.isnan(a) and a > 0:
                    pending = (sig_val, a)

        # Cerrar posición abierta al final del dataset
        if position is not None:
            trade    = self._close(position, closes[-1], "end", n - 1)
            capital *= (1 + trade["pnl_net"])
            equity[-1] = capital
            trades.append(trade)

        final_signal = self.strategy.generate_signal(self.data)
        return self._build_result(trades, equity, final_signal)

    # ── Métricas finales ───────────────────────────────────────────────────

    def _build_result(self, trades, equity, final_signal):
        if not trades:
            if self.verbose:
                print("Backtest completado: sin trades generados")
            return {
                "trades": 0, "winrate": 0.0, "profit_factor": 0.0,
                "total_pnl": 0.0, "expectancy": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0,
                "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "calmar": 0.0, "ann_return": 0.0,
                "equity_curve": equity, "exit_breakdown": {},
                "signal": final_signal,
            }

        wins   = [t for t in trades if t["win"]]
        losses = [t for t in trades if not t["win"]]

        winrate      = len(wins) / len(trades)
        gross_profit = sum(t["pnl_net"] for t in wins)
        gross_loss   = abs(sum(t["pnl_net"] for t in losses))
        pf           = (gross_profit / gross_loss) if gross_loss > 0 \
                       else (float("inf") if gross_profit > 0 else 0.0)
        # Suma aditiva — ver nota en docstring sobre diferencia con equity[-1]-1
        total_pnl    = sum(t["pnl_net"] for t in trades)
        expectancy   = total_pnl / len(trades)
        avg_win      = gross_profit / len(wins)   if wins   else 0.0
        avg_loss     = gross_loss   / len(losses) if losses else 0.0

        exit_counts = {}
        for t in trades:
            r = t["exit_reason"]
            exit_counts[r] = exit_counts.get(r, 0) + 1

        rm = compute_all(equity, periods_per_year=self.periods_per_year)

        if self.verbose:
            print(
                f"Backtest: {len(trades)} trades | "
                f"WR={winrate:.1%} | PF={pf:.2f} | "
                f"net_pnl={total_pnl:.2%} | "
                f"expectancy={expectancy:.3%} | "
                f"sharpe={rm['sharpe']:.2f} | "
                f"maxDD={rm['max_drawdown']:.2%} | "
                f"exits={exit_counts}"
            )

        return {
            "trades":         len(trades),
            "winrate":        round(winrate, 4),
            "profit_factor":  round(pf, 4),
            "total_pnl":      round(total_pnl, 6),
            "signal":         final_signal,
            "expectancy":     round(expectancy, 6),
            "avg_win":        round(avg_win, 6),
            "avg_loss":       round(avg_loss, 6),
            "max_drawdown":   rm["max_drawdown"],
            "sharpe":         rm["sharpe"],
            "sortino":        rm["sortino"],
            "calmar":         rm["calmar"],
            "ann_return":     rm["ann_return"],
            "equity_curve":   equity,
            "exit_breakdown": exit_counts,
            "trades_detail":  trades,
        }
