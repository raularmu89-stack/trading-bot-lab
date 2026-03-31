"""
mock_data.py

Genera datos OHLCV sinteticos realistas para testing sin API.
Cada par tiene precio base y volatilidad propios.
"""

import numpy as np
import pandas as pd

_BASE = {
    "BTCUSDT":  {"price": 45000, "vol": 0.018},
    "ETHUSDT":  {"price":  2500, "vol": 0.022},
    "XRPUSDT":  {"price":  0.55, "vol": 0.028},
    "SOLUSDT":  {"price":   120, "vol": 0.025},
}

_TF_CANDLES = {
    "1m":  200,
    "1d":  365,
    "4h":  365 * 6,
    "1h":  365 * 24,
}

_TF_FREQ = {
    "1m": "1min",
    "1d": "1D",
    "4h": "4h",
    "1h": "1h",
}


def generate_mock_klines(symbol="BTCUSDT", n_candles=None, timeframe="1m", seed=None):
    """
    Genera un DataFrame OHLCV sintetico para el simbolo y timeframe dados.

    Si n_candles es None, usa el numero tipico del timeframe (1 año equivalente).
    El seed es determinista por simbolo+timeframe para reproducibilidad.
    """
    if seed is None:
        seed = abs(hash(symbol + timeframe)) % (2 ** 31)

    rng = np.random.default_rng(seed)

    cfg = _BASE.get(symbol, {"price": 100, "vol": 0.02})
    base = cfg["price"]
    vol = cfg["vol"]

    if n_candles is None:
        n_candles = _TF_CANDLES.get(timeframe, 200)

    # Random walk con drift leve y cambios de regimen (tendencias)
    regime_len = max(20, n_candles // 10)
    drifts = rng.choice([-0.0003, 0.0, 0.0003], size=(n_candles // regime_len) + 1)
    drift_series = np.repeat(drifts, regime_len)[:n_candles]

    returns = rng.normal(drift_series, vol, n_candles)
    closes = base * np.cumprod(1 + returns)

    # OHLCV
    noise = rng.uniform(0.003, vol * 0.8, n_candles)
    opens = np.concatenate([[base], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + noise)
    lows = np.minimum(opens, closes) * (1 - noise)
    volumes = rng.uniform(50, 5000, n_candles) * (base / 1000)

    freq = _TF_FREQ.get(timeframe, "1min")
    timestamps = pd.date_range("2024-01-01", periods=n_candles, freq=freq)

    return pd.DataFrame({
        "timestamp": timestamps.astype(np.int64) // 10 ** 6,
        "open":   opens.round(6),
        "high":   highs.round(6),
        "low":    lows.round(6),
        "close":  closes.round(6),
        "volume": volumes.round(2),
    })
