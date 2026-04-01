"""
ema_rsi.py

Conversión desde Pine Script 'raul smc' (TradingView).

EMAs y RSI con presets multi-timeframe según estilo de trading.

Pine equivalent:
    tradingStyleInput     → TRADING_STYLES dict
    emaFastVal = ta.ema(close, emaFastLen)
    emaSlowVal = ta.ema(close, emaSlowLen)
    rsiValue   = ta.rsi(close, rsiLen)

Presets de Pine (grupo 'EMA & RSI'):
  ┌──────────┬──────────┬──────────┬───────┬───────┬───────┬──────────┬──────────┬──────────┐
  │  Estilo  │ EMA Fast │ EMA Slow │  RSI  │  OB   │  OS   │ Entry TF │  Dir TF  │ Trend TF │
  ├──────────┼──────────┼──────────┼───────┼───────┼───────┼──────────┼──────────┼──────────┤
  │ Scalping │    9     │    20    │   9   │  80   │  20   │   5m     │   15m    │   1h     │
  │ Intraday │   20     │    50    │   9   │  80   │  20   │  15m     │    1h    │   1d     │
  │ Swing    │   50     │   200    │  14   │  80   │  20   │   1h     │   4h     │   1w     │
  └──────────┴──────────┴──────────┴───────┴───────┴───────┴──────────┴──────────┴──────────┘
"""

import numpy as np

TRADING_STYLES = {
    "scalping": {
        "ema_fast":    9,
        "ema_slow":   20,
        "rsi_period":  9,
        "rsi_ob":     80,
        "rsi_os":     20,
        "entry_tf":   "5m",
        "dir_tf":    "15m",
        "trend_tf":   "1h",
    },
    "intraday": {
        "ema_fast":   20,
        "ema_slow":   50,
        "rsi_period":  9,
        "rsi_ob":     80,
        "rsi_os":     20,
        "entry_tf":  "15m",
        "dir_tf":     "1h",
        "trend_tf":    "1d",
    },
    "swing": {
        "ema_fast":   50,
        "ema_slow":  200,
        "rsi_period": 14,
        "rsi_ob":     80,
        "rsi_os":     20,
        "entry_tf":   "1h",
        "dir_tf":     "4h",
        "trend_tf":    "1w",
    },
}


def _ema(series, period):
    """EMA sin pandas para evitar dependencias circulares."""
    n   = len(series)
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = np.mean(series[:period])
    k = 2.0 / (period + 1)
    for i in range(period, n):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(series, period):
    """RSI sin pandas."""
    n      = len(series)
    out    = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta  = np.diff(series)
    gains  = np.maximum(delta, 0)
    losses = np.abs(np.minimum(delta, 0))

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs       = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        out[i + 1] = 100 - (100 / (1 + rs))

    return out


def compute_ema_rsi(data, style="scalping", fast=None, slow=None, rsi_period=None):
    """
    Calcula EMAs y RSI según el estilo de trading o parámetros custom.

    Pine equivalent:
        emaFastVal = ta.ema(close, emaFastLen)
        emaSlowVal = ta.ema(close, emaSlowLen)
        rsiValue   = ta.rsi(close, rsiLen)

    Parametros:
      style      : "scalping" | "intraday" | "swing"  (ignora si se dan fast/slow/rsi_period)
      fast       : override EMA rápida
      slow       : override EMA lenta
      rsi_period : override RSI

    Devuelve dict:
      ema_fast      : array numpy
      ema_slow      : array numpy
      rsi           : array numpy
      rsi_ob        : nivel sobrecompra
      rsi_os        : nivel sobreventa
      style_params  : dict con los parámetros del estilo seleccionado
      cross_up      : bool — EMA fast cruzó al alza EMA slow en la última vela
      cross_down    : bool — EMA fast cruzó a la baja EMA slow en la última vela
      rsi_overbought: bool — RSI > ob en la última vela
      rsi_oversold  : bool — RSI < os en la última vela
      signal        : "buy" | "sell" | "hold" basado en cruce de EMAs
    """
    params = TRADING_STYLES.get(style.lower(), TRADING_STYLES["scalping"])
    f_len  = fast       if fast       is not None else params["ema_fast"]
    s_len  = slow       if slow       is not None else params["ema_slow"]
    r_len  = rsi_period if rsi_period is not None else params["rsi_period"]
    rsi_ob = params["rsi_ob"]
    rsi_os = params["rsi_os"]

    closes = data["close"].values.astype(float)

    ema_f = _ema(closes, f_len)
    ema_s = _ema(closes, s_len)
    rsi   = _rsi(closes, r_len)

    # Cruce EMA (última y penúltima vela)
    cross_up   = False
    cross_down = False
    if len(ema_f) >= 2 and not np.isnan(ema_f[-1]) and not np.isnan(ema_s[-1]):
        prev_above = ema_f[-2] > ema_s[-2]
        curr_above = ema_f[-1] > ema_s[-1]
        cross_up   = not prev_above and curr_above
        cross_down = prev_above and not curr_above

    rsi_last      = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
    rsi_overbought = rsi_last > rsi_ob
    rsi_oversold   = rsi_last < rsi_os

    # Señal simple: cruce de EMAs confirmado por RSI
    if cross_up and not rsi_overbought:
        signal = "buy"
    elif cross_down and not rsi_oversold:
        signal = "sell"
    else:
        signal = "hold"

    return {
        "ema_fast":       ema_f,
        "ema_slow":       ema_s,
        "rsi":            rsi,
        "rsi_ob":         rsi_ob,
        "rsi_os":         rsi_os,
        "style_params":   params,
        "cross_up":       cross_up,
        "cross_down":     cross_down,
        "rsi_overbought": rsi_overbought,
        "rsi_oversold":   rsi_oversold,
        "signal":         signal,
    }


def ema_trend_filter(data, style="scalping"):
    """
    Filtro de tendencia basado en EMAs: ¿el precio está por encima de la EMA lenta?

    Retorna:
      "bullish" si close[-1] > ema_slow[-1]
      "bearish" si close[-1] < ema_slow[-1]
      "neutral"  si ema_slow no tiene datos suficientes
    """
    result = compute_ema_rsi(data, style=style)
    ema_s  = result["ema_slow"]
    close  = data["close"].values.astype(float)

    if np.isnan(ema_s[-1]):
        return "neutral"
    if close[-1] > ema_s[-1]:
        return "bullish"
    elif close[-1] < ema_s[-1]:
        return "bearish"
    return "neutral"
