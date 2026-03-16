from bot.engine import TradingEngine

def main():

    print("Iniciando bot de trading")

    engine = TradingEngine()

    engine.load_data()

    engine.show_data()


if __name__ == "__main__":
    main()
