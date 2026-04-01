"""
fvg.py

Conversión desde Pine Script 'raul smc' (TradingView).

Fair Value Gaps (FVG) — gaps entre la vela 1 y la vela 3 de una secuencia de 3:

  Bullish FVG: low[3] > high[1]  y  close[2] > high[1]   → gap alcista
  Bearish FVG: high[3] < low[1]  y  close[2] < low[1]    → gap bajista

Pine equivalent:
    bullishFairValueGap = currentLow > last2High and lastClose > last2High
    bearishFairValueGap = currentHigh < last2Low and lastClose < last2Low

Con filtro de umbral automático:
    threshold = cumulative_mean(|bar_delta_pct|) * 2
    Solo se acepta el FVG si |barDeltaPct| > threshold
"""

import numpy as np


def _cumulative_mean_threshold(opens, closes):
    """
    Pine equivalent:
        barDeltaPercent = (close - open) / (open * 100)
        threshold = ta.cum(|barDeltaPercent|) / bar_index * 2

    Devuelve array de thresholds acumulativos.
    """
    n = len(closes)
    delta_pct = np.abs((closes - opens) / np.where(opens != 0, opens, 1e-10) * 100)
    cum_mean  = np.cumsum(delta_pct) / (np.arange(1, n + 1))
    return cum_mean * 2


def detect_fvg(data, auto_threshold=False):
    """
    Detecta Fair Value Gaps en el dataset.

    Pine equivalent:
        drawFairValueGaps()  con fairValueGapsThresholdInput

    Parametros:
      auto_threshold : si True, filtra FVGs pequeños con umbral dinámico
                       (Pine: fairValueGapsThresholdInput = true)

    Devuelve lista de dicts:
      type       : "bullish" | "bearish"
      top        : precio superior del gap
      bottom     : precio inferior del gap
      mid        : punto medio del gap
      index      : índice de la vela central (vela 2)
      mitigated  : True si el precio ya cruzó el gap
      start      : alias de bottom (compat. backward)
      end        : alias de top   (compat. backward)
    """
    if data is None or len(data) < 3:
        return []

    o = data["open"].values.astype(float)
    h = data["high"].values.astype(float)
    l = data["low"].values.astype(float)
    c = data["close"].values.astype(float)
    n = len(data)

    thresholds = _cumulative_mean_threshold(o, c) if auto_threshold else None

    fvgs = []

    for i in range(2, n):
        candle1_high = h[i - 2]
        candle1_low  = l[i - 2]
        candle2_close = c[i - 1]
        candle3_low   = l[i]
        candle3_high  = h[i]

        bar_delta_pct = abs((c[i - 1] - o[i - 1]) / max(o[i - 1], 1e-10) * 100)
        threshold     = thresholds[i - 1] if auto_threshold else 0.0

        # ── Bullish FVG ──────────────────────────────────────────────
        if candle3_low > candle1_high and candle2_close > candle1_high:
            if bar_delta_pct > threshold:
                top    = candle3_low
                bottom = candle1_high
                mid    = (top + bottom) / 2
                # Mitigado si el precio baja hasta el bottom del gap
                mitigated = bool(np.any(l[i + 1:] < bottom)) if i + 1 < n else False
                fvgs.append({
                    "type":      "bullish",
                    "top":       top,
                    "bottom":    bottom,
                    "mid":       mid,
                    "index":     i,
                    "mitigated": mitigated,
                    "start":     bottom,   # backward compat
                    "end":       top,      # backward compat
                })

        # ── Bearish FVG ──────────────────────────────────────────────
        elif candle3_high < candle1_low and candle2_close < candle1_low:
            if bar_delta_pct > threshold:
                top    = candle1_low
                bottom = candle3_high
                mid    = (top + bottom) / 2
                mitigated = bool(np.any(h[i + 1:] > top)) if i + 1 < n else False
                fvgs.append({
                    "type":      "bearish",
                    "top":       top,
                    "bottom":    bottom,
                    "mid":       mid,
                    "index":     i,
                    "mitigated": mitigated,
                    "start":     bottom,   # backward compat
                    "end":       top,      # backward compat
                })

    return fvgs


def get_active_fvgs(data, auto_threshold=False):
    """Devuelve solo los FVGs que no han sido mitigados."""
    return [f for f in detect_fvg(data, auto_threshold) if not f["mitigated"]]


def price_in_fvg(price, fvgs, fvg_type=None):
    """
    Comprueba si un precio está dentro de algún FVG.

    fvg_type : "bullish" | "bearish" | None (cualquiera)
    """
    for f in fvgs:
        if fvg_type and f["type"] != fvg_type:
            continue
        if f["bottom"] <= price <= f["top"]:
            return True
    return False
