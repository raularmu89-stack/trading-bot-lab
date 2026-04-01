"""
resampler.py

Convierte OHLCV de un timeframe bajo a uno alto mediante resample.

Uso:
    from indicators.resampler import resample
    df_4h = resample(df_1h, "4h")
    df_1d = resample(df_1h, "1d")

Equivalente Pine:
    request.security(syminfo.tickerid, "240", [...])
"""

import pandas as pd

_RULES = {
    "1m":  "1min",  "3m":  "3min",  "5m":  "5min",
    "15m": "15min", "30m": "30min",
    "1h":  "1h",    "2h":  "2h",    "4h":  "4h",
    "6h":  "6h",    "8h":  "8h",    "12h": "12h",
    "1d":  "1D",    "1w":  "1W",
}


def resample(df, target_tf):
    """
    Resamplea un DataFrame OHLCV al timeframe indicado.

    Parámetros:
      df        : DataFrame con columnas open/high/low/close/volume
                  e índice DatetimeIndex (o columna 'timestamp')
      target_tf : string como "4h", "1d", "1w"

    Devuelve DataFrame OHLCV con el nuevo timeframe.
    """
    rule = _RULES.get(target_tf)
    if rule is None:
        raise ValueError(f"Timeframe desconocido: {target_tf!r}. "
                         f"Opciones: {list(_RULES)}")

    work = df.copy()

    # Asegurar índice datetime
    if "timestamp" in work.columns and not isinstance(work.index, pd.DatetimeIndex):
        work = work.set_index(pd.to_datetime(work["timestamp"], unit="ms"))
    elif not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index)

    resampled = work.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()

    return resampled.reset_index(drop=True)


def resample_aligned(df_low, target_tf):
    """
    Resamplea y alinea con el DataFrame de bajo TF:
    devuelve un array del mismo largo que df_low donde cada fila
    contiene los valores del candle de alto TF al que pertenece.

    Útil para obtener la tendencia del TF superior en cada vela del TF inferior.
    """
    rule = _RULES.get(target_tf)
    if rule is None:
        raise ValueError(f"Timeframe desconocido: {target_tf!r}")

    work = df_low.copy()
    if "timestamp" in work.columns:
        work.index = pd.to_datetime(work["timestamp"], unit="ms")
    elif not isinstance(work.index, pd.DatetimeIndex):
        # Crear índice sintético si no hay timestamps reales
        work.index = pd.date_range("2024-01-01", periods=len(work), freq="1h")

    high_tf = work.resample(rule).agg({
        "open": "first", "high": "max",
        "low":  "min",   "close": "last", "volume": "sum",
    })

    # Forward-fill para alinear con el TF bajo
    aligned = high_tf.reindex(work.index, method="ffill").fillna(method="bfill")
    aligned = aligned.reset_index(drop=True)
    return aligned
