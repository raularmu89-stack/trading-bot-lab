import requests
import pandas as pd
from config.settings import SYMBOL, TIMEFRAME, LIMIT


def fetch_klines():
    """
    Descarga velas de Binance y devuelve un DataFrame.
    """

    url = "https://api.binance.com/api/v3/klines"

    params = {
        "symbol": SYMBOL,
        "interval": TIMEFRAME,
        "limit": LIMIT
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Error descargando datos")
        return None

    data = response.json()

    df = pd.DataFrame(data, columns=[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore"
    ])

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df
