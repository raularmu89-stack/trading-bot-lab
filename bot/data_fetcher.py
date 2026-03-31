import time as _time

import requests
import pandas as pd

from config.settings import SYMBOL, TIMEFRAME, LIMIT

_KLINE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
]

_ONE_YEAR_MS = 365 * 24 * 60 * 60 * 1000
_BINANCE_URL = "https://api.binance.com/api/v3/klines"


def _parse_raw(raw):
    df = pd.DataFrame(raw, columns=_KLINE_COLUMNS)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def fetch_klines(symbol=None, timeframe=None, limit=None):
    """Descarga las ultimas `limit` velas de un par y timeframe dados."""
    symbol = symbol or SYMBOL
    timeframe = timeframe or TIMEFRAME
    limit = limit or LIMIT

    params = {"symbol": symbol, "interval": timeframe, "limit": limit}
    response = requests.get(_BINANCE_URL, params=params)

    if response.status_code != 200:
        print(f"Error descargando datos ({symbol} {timeframe}): {response.status_code}")
        return None

    return _parse_raw(response.json())


def fetch_klines_year(symbol, timeframe):
    """
    Descarga exactamente 1 año de velas para el par y timeframe indicados.
    Usa paginacion automatica (maximo 1000 velas por peticion Binance).
    """
    end_ts = int(_time.time() * 1000)
    start_ts = end_ts - _ONE_YEAR_MS

    all_raw = []
    current_start = start_ts

    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": 1000,
        }
        response = requests.get(_BINANCE_URL, params=params)

        if response.status_code != 200:
            print(f"Error en paginacion ({symbol} {timeframe}): {response.status_code}")
            return None

        chunk = response.json()
        if not chunk:
            break

        all_raw.extend(chunk)

        # Avanzar al siguiente bloque
        current_start = chunk[-1][0] + 1

        if len(chunk) < 1000:
            break  # ultima pagina

    if not all_raw:
        return None

    return _parse_raw(all_raw)
