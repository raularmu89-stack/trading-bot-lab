class Backtester:
    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data

    def run(self):
        signal = self.strategy.generate_signal(self.data)

        print("Backtest result:")
        print(signal)

        return {
            "trades": 1 if signal["signal"] in ["buy", "sell"] else 0,
            "winrate": 0,
            "profit_factor": 0,
            "signal": signal
        }
