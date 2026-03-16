import requests
import pandas as pd
from datetime import datetime

BASE_URL = "https://api-futures.kucoin.com"

def fetch_klines(symbol="XBTUSDTM", interval="1min", limit=200):
    url = f"{BASE_URL}/api/v1/kline/query"
    
    params = {
        "symbol": symbol,
        "granularity": 1,
        "limit": limit
    }

    r = requests.get(url, params=params)
    data = r.json()

    if "data" not in data:
        print("Error descargando datos")
        return None

    df = pd.DataFrame(data["data"])
    df.columns = ["time","open","close","high","low","volume","turnover"]

    df["time"] = pd.to_datetime(df["time"], unit="ms")

    return df


if __name__ == "__main__":
    df = fetch_klines()
    print(df.head())
