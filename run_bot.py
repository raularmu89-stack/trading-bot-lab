from bot.data_fetcher import fetch_klines
from strategies.smc_strategy import SMCStrategy
from backtests.backtester import Backtester


def main():
    print("Iniciando bot de trading...")

    data = fetch_klines()

    if data is None:
        print("No se pudieron cargar datos.")
        return

    strategy = SMCStrategy()
    backtester = Backtester(strategy, data)

    result = backtester.run()

    print(result)


if __name__ == "__main__":
    main()
