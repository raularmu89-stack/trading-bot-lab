from bot.data_fetcher import fetch_klines
from strategies.smc_strategy import SMCStrategy
from backtests.backtester import Backtester


class TradingEngine:
    def __init__(self):
        self.data = None
        self.strategy = SMCStrategy()
        self.backtester = None

    def load_data(self):
        print("Descargando datos de mercado...")
        self.data = fetch_klines()

        if self.data is None or self.data.empty:
            print("No se pudieron cargar datos.")
            return False

        print(f"Datos cargados correctamente: {len(self.data)} velas")
        return True

    def run_backtest(self):
        if self.data is None or self.data.empty:
            print("No hay datos cargados para ejecutar el backtest.")
            return None

        self.backtester = Backtester(strategy=self.strategy, data=self.data)

        result = self.backtester.run()

        return result

    def show_data(self, rows=5):
        if self.data is None or self.data.empty:
            print("No hay datos para mostrar.")
            return

        print(self.data.tail(rows))

    def run(self):
        loaded = self.load_data()

        if not loaded:
            return None

        self.show_data()

        result = self.run_backtest()

        print("Resultado final del motor:")
        print(result)

        return result
