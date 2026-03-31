class Backtester:
    def __init__(self, strategy, data, min_candles=20, max_hold=10):
        self.strategy = strategy
        self.data = data
        self.min_candles = min_candles
        self.max_hold = max_hold  # velas maximas por trade antes de forzar cierre

    def run(self):
        trades = []
        position = None  # {"side": str, "entry_price": float, "entry_idx": int}

        for i in range(self.min_candles, len(self.data)):
            window = self.data.iloc[: i + 1]
            signal = self.strategy.generate_signal(window)
            current_price = float(self.data["close"].iloc[i])

            if position is None:
                if signal["signal"] in ["buy", "sell"]:
                    position = {
                        "side": signal["signal"],
                        "entry_price": current_price,
                        "entry_idx": i,
                    }
            else:
                held_candles = i - position["entry_idx"]
                opposing = (
                    position["side"] == "buy" and signal["signal"] == "sell"
                ) or (
                    position["side"] == "sell" and signal["signal"] == "buy"
                )
                should_exit = opposing or held_candles >= self.max_hold

                if should_exit:
                    entry = position["entry_price"]
                    if position["side"] == "buy":
                        pnl = (current_price - entry) / entry
                    else:
                        pnl = (entry - current_price) / entry

                    trades.append({
                        "side": position["side"],
                        "entry_price": entry,
                        "exit_price": current_price,
                        "pnl": pnl,
                        "win": pnl > 0,
                    })
                    position = None

                    # Abrir nueva posicion si hay señal
                    if signal["signal"] in ["buy", "sell"]:
                        position = {
                            "side": signal["signal"],
                            "entry_price": current_price,
                            "entry_idx": i,
                        }

        # Cerrar posicion abierta al final
        if position is not None:
            final_price = float(self.data["close"].iloc[-1])
            entry = position["entry_price"]
            if position["side"] == "buy":
                pnl = (final_price - entry) / entry
            else:
                pnl = (entry - final_price) / entry
            trades.append({
                "side": position["side"],
                "entry_price": entry,
                "exit_price": final_price,
                "pnl": pnl,
                "win": pnl > 0,
            })

        final_signal = self.strategy.generate_signal(self.data)

        if not trades:
            print("Backtest completado: sin trades generados")
            return {
                "trades": 0,
                "winrate": 0.0,
                "profit_factor": 0.0,
                "total_pnl": 0.0,
                "signal": final_signal,
            }

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

        total_pnl = sum(t["pnl"] for t in trades)

        print(
            f"Backtest completado: {len(trades)} trades | "
            f"winrate={winrate:.1%} | "
            f"profit_factor={profit_factor:.2f} | "
            f"pnl_total={total_pnl:.2%}"
        )

        return {
            "trades": len(trades),
            "winrate": round(winrate, 4),
            "profit_factor": round(profit_factor, 4),
            "total_pnl": round(total_pnl, 4),
            "signal": final_signal,
        }
