"""
mtf_smc.py

Multi-Timeframe SMC Strategy.

Lógica:
  - Tendencia 4H: precio vs EMA(trend_ema) en velas 4H resampleadas
    → solo long si precio > EMA4H  (tendencia alcista)
    → solo short si precio < EMA4H (tendencia bajista)
  - Señal 1H: SMC standard (_precompute_signals)
  - Combinar: emitir señal solo si SMC 1H coincide con tendencia 4H

Esto filtra señales contra-tendencia → WR esperada 52-57% vs 47% base
→ Kelly fraction sube de ~20% a ~28-35% → retornos +40-70% más.
"""

import numpy as np
import pandas as pd
from backtests.backtester_fast import _precompute_signals


def _ema(arr, p):
    a   = 2 / (p + 1)
    out = np.full(len(arr), np.nan)
    if len(arr) < p:
        return out
    out[p - 1] = arr[:p].mean()
    for i in range(p, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out


class MultiTFSMC:
    """
    SMC 1H filtrado por tendencia 4H.

    Parámetros
    ----------
    swing_window  : ventana SMC para señales 1H
    trend_ema     : periodo EMA sobre velas 4H (default 50 → 200h)
    require_fvg   : heredado de SMCStrategy
    use_choch     : heredado de SMCStrategy
    """

    # Atributos compatibles con el backtester
    require_fvg      = False
    use_choch_filter = False

    def __init__(self, swing_window=5, trend_ema=50,
                 require_fvg=False, use_choch=False):
        self.swing_window    = swing_window
        self.trend_ema       = trend_ema
        self.require_fvg     = require_fvg
        self.use_choch_filter = use_choch

    def generate_signals_batch(self, data_1h: pd.DataFrame) -> list:
        """
        Genera señales combinando SMC 1H + tendencia EMA sobre 4H.

        data_1h : DataFrame OHLCV con índice DatetimeIndex 1H
        """
        # ── Señales SMC 1H ────────────────────────────────────────────
        smc_sigs = _precompute_signals(
            data_1h,
            swing_window     = self.swing_window,
            require_fvg      = self.require_fvg,
            use_choch_filter = self.use_choch_filter,
        )

        # ── Tendencia 4H resampleando el OHLCV ───────────────────────
        data_4h = data_1h.resample("4h").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()

        ema_4h = _ema(data_4h["close"].values, self.trend_ema)
        trend_series = pd.Series(
            np.where(data_4h["close"].values > ema_4h, 1, -1),
            index=data_4h.index,
        )

        # Reindexar a 1H (forward-fill: la tendencia 4H se mantiene
        # hasta que llega la siguiente vela 4H)
        trend_1h = trend_series.reindex(data_1h.index, method="ffill").fillna(0)
        trend_arr = trend_1h.values

        # ── Combinar: solo señales alineadas con tendencia ────────────
        n    = len(smc_sigs)
        sigs = ["hold"] * n
        for i in range(n):
            s = smc_sigs[i]
            t = trend_arr[i]
            if s == "buy"  and t >= 0:   # alcista o neutral → OK
                sigs[i] = "buy"
            elif s == "sell" and t <= 0:  # bajista o neutral → OK
                sigs[i] = "sell"
            # si s != "hold" pero t va en contra → filtrado

        return sigs

    def __repr__(self):
        return f"MultiTFSMC(sw={self.swing_window},ema4h={self.trend_ema})"
