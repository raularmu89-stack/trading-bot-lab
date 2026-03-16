import pandas as pd
from bot.data_fetcher import fetch_klines


class TradingEngine:

    def __init__(self, symbol="XBTUSDTM"):
        self.symbol = symbol
        self.data = None

    def load_data(self):
        print("Descargando datos...")
        self.data = fetch_klines(self.symbol)

        if self.data is None:
            print("Error cargando datos")
        else:
            print("Datos cargados:", len(self.data))

    def show_data(self):
        if self.data is not None:
            print(self.data.tail())
        else:
            print("No hay datos")


if __name__ == "__main__":
    engine = TradingEngine()
    engine.load_data()
    engine.show_data()
