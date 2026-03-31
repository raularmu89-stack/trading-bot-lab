"""
risk_manager.py

Calcula niveles de Stop Loss y Take Profit por operacion.

Metodos disponibles:
  - "fixed"  : SL/TP como porcentaje fijo del precio de entrada
  - "atr"    : SL basado en ATR (Average True Range), TP = SL * rr_ratio
"""

import numpy as np


def _calc_atr(data, period=14):
    high = data["high"].values
    low = data["low"].values
    close = data["close"].values

    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) > 0 else 0.0
    return float(np.mean(tr[-period:]))


class RiskManager:
    def __init__(self, sl_pct=0.02, rr_ratio=2.0, method="fixed", atr_multiplier=1.5, atr_period=14):
        """
        sl_pct          : Stop loss como fraccion del precio (metodo fixed)
        rr_ratio        : Risk/Reward ratio → TP = SL_distancia * rr_ratio
        method          : "fixed" o "atr"
        atr_multiplier  : Multiplicador del ATR para el SL (metodo atr)
        atr_period      : Periodo del ATR
        """
        self.sl_pct = sl_pct
        self.rr_ratio = rr_ratio
        self.method = method
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period

    def calculate_levels(self, data, entry_price, side):
        """
        Devuelve (stop_loss, take_profit) para una entrada dada.

        data  : DataFrame OHLCV con el historico hasta el momento de entrada
        entry : precio de entrada
        side  : "buy" o "sell"
        """
        if self.method == "atr":
            atr = _calc_atr(data, self.atr_period)
            sl_dist = atr * self.atr_multiplier if atr > 0 else entry_price * 0.02
        else:
            sl_dist = entry_price * self.sl_pct

        tp_dist = sl_dist * self.rr_ratio

        if side == "buy":
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        return round(sl, 8), round(tp, 8)

    def __repr__(self):
        return (
            f"RiskManager(method={self.method}, sl_pct={self.sl_pct}, "
            f"rr_ratio={self.rr_ratio})"
        )
