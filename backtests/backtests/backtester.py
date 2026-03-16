class Backtester:
    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data

    def run(self):
        print("Ejecutando backtest...")
        return {
            "trades": 0,
            "winrate": 0,
            "profit_factor": 0
        }
