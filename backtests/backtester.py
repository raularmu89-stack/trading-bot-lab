"""
backtester.py

Backtester con modelo de ejecución realista:
  - SL/TP basados en ATR, fijados en la entrada
  - Comisiones round-trip configurables (default 0.1% por lado = KuCoin taker)
  - Slippage como fracción del precio (default 0, configurable)
  - Equity curve compuesta vela a vela
  - Métricas completas: Sharpe, Sortino, max drawdown, Calmar, expectancy

Interfaz pública sin cambios: Backtester(strategy, data, ...).run() → dict
Los campos originales (trades, winrate, profit_factor, total_pnl, signal)
siguen presentes. Se añaden los campos nuevos.

Limitación conocida: entrada en close[i] de la vela de señal.
Introduce un sesgo de look-ahead menor (precio de cierre conocido al fin
de vela). El siguiente paso natural sería entrar en open[i+1].
"""

import numpy as np
from backtests.metrics import compute_all


class Backtester:
    """
    Parámetros
    ----------
    strategy       : objeto con .generate_signal(window) → {"signal": str}
    data           : DataFrame con columnas open/high/low/close/volume
    sl_atr_mult    : multiplicador ATR para stop-loss  (default 2.0)
    tp_atr_mult    : multiplicador ATR para take-profit (default 4.0 → RR 1:2)
    atr_period     : periodo ATR suavizado Wilder (default 14)
    fee_rate       : comisión por lado como fracción (default 0.001 = 0.1%)
    slippage_pct   : slippage adicional sobre el precio de entrada (default 0)
    max_hold       : velas máximas antes de forzar cierre (fallback, no estrategia)
    min_candles    : velas mínimas antes de empezar a generar señales
    periods_per_year: para anualizaciones (252=diario, 8760=1h crypto)
    verbose        : imprimir resumen al finalizar
    """

    def __init__(
        self,
        strategy,
        data,
        sl_atr_mult    = 2.0,
        tp_atr_mult    = 4.0,
        atr_period     = 14,
        fee_rate       = 0.001,
        slippage_pct   = 0.0,
        max_hold       = 20,
        min_candles    = 20,
        periods_per_year = 252,
        verbose        = True,
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

    # ── ATR (Wilder smoothing) ─────────────────────────────────────────────

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
            atr[p - 1] = tr[:p].mean()
            for i in range(p, n):
                atr[i] = (atr[i-1] * (p - 1) + tr[i]) / p
        return atr

    # ── SL/TP check intra-vela ────────────────────────────────────────────

    def _check_sl_tp(self, position, high, low):
        """
        Comprueba si SL o TP fue tocado durante la vela actual.

        Usa high/low — no el close — para detectar niveles intravela.
        Si ambos se tocan en la misma vela (posible en gaps o alta volatilidad),
        asume SL primero: postura conservadora que evita sobrestimar el PnL.

        Retorna (razón: str | None, precio_de_salida: float | None)
        """
        sl = position["sl"]
        tp = position["tp"]

        if position["side"] == "buy":
            sl_hit = low <= sl
            tp_hit = high >= tp
        else:
            sl_hit = high >= sl
            tp_hit = low <= tp

        if sl_hit:
            return "sl", sl
        if tp_hit:
            return "tp", tp
        return None, None

    # ── Cierre de posición ────────────────────────────────────────────────

    def _close(self, position, exit_price, reason, exit_idx):
        entry = position["entry_price"]
        side  = position["side"]

        raw_pnl = (exit_price - entry) / entry if side == "buy" \
                  else (entry - exit_price) / entry

        fee_total = 2 * self.fee_rate   # entrada + salida
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
            # alias para compatibilidad con código existente
            "pnl":         round(pnl_net, 6),
            "win":         pnl_net > 0,
        }

    # ── Loop principal ────────────────────────────────────────────────────

    def run(self):
        atr    = self._compute_atr()
        closes = self.data["close"].values.astype(float)
        highs  = self.data["high"].values.astype(float)
        lows   = self.data["low"].values.astype(float)
        n      = len(closes)

        trades   = []
        position = None
        capital  = 1.0
        equity   = []

        for i in range(n):
            equity.append(capital)

            if i < self.min_candles:
                continue

            # ── 1. Gestionar posición abierta ─────────────────────────────
            if position is not None:
                # a) SL/TP intravela
                reason, exit_px = self._check_sl_tp(position, highs[i], lows[i])

                # b) Timeout: max_hold como último recurso (no como estrategia)
                if reason is None and (i - position["entry_idx"]) >= self.max_hold:
                    reason  = "max_hold"
                    exit_px = closes[i]

                if reason is not None:
                    trade    = self._close(position, exit_px, reason, i)
                    capital *= (1 + trade["pnl_net"])
                    equity[-1] = capital
                    trades.append(trade)
                    position = None

            # ── 2. Señal en esta vela ─────────────────────────────────────
            window = self.data.iloc[: i + 1]
            sig    = self.strategy.generate_signal(window)
            sig_val = sig.get("signal", "hold")

            # Señal opuesta a posición abierta → cerrar a close[i]
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

            # ── 3. Abrir nueva posición ───────────────────────────────────
            if position is None and sig_val in ("buy", "sell"):
                a = atr[i]
                if np.isnan(a) or a <= 0:
                    continue

                slip  = closes[i] * self.slippage_pct
                if sig_val == "buy":
                    entry = closes[i] + slip
                    sl    = entry - self.sl_atr_mult * a
                    tp    = entry + self.tp_atr_mult * a
                else:
                    entry = closes[i] - slip
                    sl    = entry + self.sl_atr_mult * a
                    tp    = entry - self.tp_atr_mult * a

                position = {
                    "side":        sig_val,
                    "entry_price": entry,
                    "entry_idx":   i,
                    "sl":          sl,
                    "tp":          tp,
                }

        # Cerrar posición que queda abierta al final del dataset
        if position is not None:
            trade    = self._close(position, closes[-1], "end", n - 1)
            capital *= (1 + trade["pnl_net"])
            equity[-1] = capital
            trades.append(trade)

        final_signal = self.strategy.generate_signal(self.data)
        return self._build_result(trades, equity, final_signal)

    # ── Métricas finales ──────────────────────────────────────────────────

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
                "equity_curve": equity,
                "exit_breakdown": {},
                "signal": final_signal,
            }

        wins   = [t for t in trades if t["win"]]
        losses = [t for t in trades if not t["win"]]

        winrate      = len(wins) / len(trades)
        gross_profit = sum(t["pnl_net"] for t in wins)
        gross_loss   = abs(sum(t["pnl_net"] for t in losses))
        pf           = (gross_profit / gross_loss) if gross_loss > 0 \
                       else (float("inf") if gross_profit > 0 else 0.0)
        total_pnl    = sum(t["pnl_net"] for t in trades)
        expectancy   = total_pnl / len(trades)
        avg_win      = gross_profit / len(wins)  if wins   else 0.0
        avg_loss     = gross_loss   / len(losses) if losses else 0.0

        exit_counts = {}
        for t in trades:
            r = t["exit_reason"]
            exit_counts[r] = exit_counts.get(r, 0) + 1

        rm = compute_all(equity, periods_per_year=self.periods_per_year)

        if self.verbose:
            print(
                f"Backtest completado: {len(trades)} trades | "
                f"WR={winrate:.1%} | PF={pf:.2f} | "
                f"net_pnl={total_pnl:.2%} | "
                f"expectancy={expectancy:.3%} | "
                f"sharpe={rm['sharpe']:.2f} | "
                f"maxDD={rm['max_drawdown']:.2%} | "
                f"exits={exit_counts}"
            )

        return {
            # compatibilidad con código existente
            "trades":        len(trades),
            "winrate":       round(winrate, 4),
            "profit_factor": round(pf, 4),
            "total_pnl":     round(total_pnl, 6),
            "signal":        final_signal,
            # nuevas métricas
            "expectancy":    round(expectancy, 6),
            "avg_win":       round(avg_win, 6),
            "avg_loss":      round(avg_loss, 6),
            "max_drawdown":  rm["max_drawdown"],
            "sharpe":        rm["sharpe"],
            "sortino":       rm["sortino"],
            "calmar":        rm["calmar"],
            "ann_return":    rm["ann_return"],
            "equity_curve":  equity,
            "exit_breakdown": exit_counts,
            "trades_detail": trades,
        }
