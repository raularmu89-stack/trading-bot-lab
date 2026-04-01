"""
kelly_sizer.py

Gestión de posición dinámica basada en el criterio de Kelly.

Criterio de Kelly:
    f* = (p * b - q) / b
    donde:
      p  = probabilidad de ganar (win rate)
      q  = 1 - p  (probabilidad de perder)
      b  = ratio ganancia/pérdida media (profit factor / trade)

Variantes:
  full_kelly    : tamaño óptimo teórico (máximo CAGR, alto drawdown)
  half_kelly    : f*/2  (equilibrio CAGR/drawdown — recomendado en práctica)
  quarter_kelly : f*/4  (conservador, mínimo drawdown)
  fixed_pct     : porcentaje fijo clásico (no-Kelly)

El KellySizer se combina con RiskManager para calcular el tamaño real:
  position_size = kelly_fraction * balance / (entry - sl)

Uso:
    from strategies.kelly_sizer import KellySizer
    ks = KellySizer(variant="half_kelly", max_fraction=0.25, min_trades=10)
    size = ks.position_size(balance=1000, entry=45000, sl=44100, trade_history=trades)
"""

import math


class KellySizer:
    """
    Calcula el tamaño de posición óptimo con Kelly criterion.

    Parámetros
    ----------
    variant      : "full_kelly" | "half_kelly" | "quarter_kelly" | "fixed_pct"
    fixed_pct    : fracción fija si variant="fixed_pct" (ej. 0.02 = 2%)
    max_fraction : fracción máxima del balance por trade (cap de seguridad)
    min_fraction : fracción mínima aunque Kelly sea muy baja
    min_trades   : trades mínimos en historial antes de activar Kelly
                   (antes usa fixed_pct como fallback)
    """

    def __init__(
        self,
        variant="half_kelly",
        fixed_pct=0.02,
        max_fraction=0.25,
        min_fraction=0.005,
        min_trades=20,
    ):
        self.variant      = variant
        self.fixed_pct    = fixed_pct
        self.max_fraction = max_fraction
        self.min_fraction = min_fraction
        self.min_trades   = min_trades

    def kelly_fraction(self, trade_history):
        """
        Calcula f* de Kelly a partir del historial de trades.

        trade_history : lista de dicts {"pnl": float, "win": bool}
                        o lista de floats (pnl directo)

        Devuelve fracción f* (puede ser negativa → sin edge → no operar)
        """
        if len(trade_history) < self.min_trades:
            return self.fixed_pct  # fallback a fixed hasta tener datos

        # Normalizar
        pnls = []
        for t in trade_history:
            if isinstance(t, dict):
                pnls.append(t.get("pnl", 0.0))
            else:
                pnls.append(float(t))

        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        if not wins or not losses:
            return self.fixed_pct

        p  = len(wins) / len(pnls)          # win rate
        q  = 1 - p                           # loss rate
        b  = (sum(wins) / len(wins)) / (abs(sum(losses) / len(losses)))  # profit/loss ratio

        if b <= 0:
            return 0.0

        f_star = (p * b - q) / b

        # Aplicar variante
        if self.variant == "full_kelly":
            return f_star
        elif self.variant == "half_kelly":
            return f_star / 2
        elif self.variant == "quarter_kelly":
            return f_star / 4
        else:  # fixed_pct
            return self.fixed_pct

    def position_fraction(self, trade_history):
        """
        Devuelve la fracción del balance a arriesgar en el próximo trade.
        Aplicando min/max como safety net.
        """
        f = self.kelly_fraction(trade_history)
        f = max(self.min_fraction, min(self.max_fraction, f))
        return f

    def position_size(self, balance, entry, sl, trade_history):
        """
        Calcula el tamaño de posición en unidades monetarias.

        Parámetros:
          balance       : balance actual
          entry         : precio de entrada
          sl            : precio de stop loss
          trade_history : historial de trades pasados

        Devuelve dict:
          fraction      : fracción de Kelly aplicada
          risk_amount   : capital en riesgo (balance * fraction)
          units         : unidades del activo a comprar/vender
          position_value: valor total de la posición
        """
        fraction    = self.position_fraction(trade_history)
        risk_amount = balance * fraction
        sl_distance = abs(entry - sl) / entry  # como fracción del precio

        if sl_distance <= 0:
            return {"fraction": fraction, "risk_amount": 0,
                    "units": 0, "position_value": 0}

        # Units para que la pérdida máxima = risk_amount
        # risk_amount = units * entry * sl_distance
        units          = risk_amount / (entry * sl_distance)
        position_value = units * entry

        return {
            "fraction":       round(fraction, 6),
            "risk_amount":    round(risk_amount, 4),
            "units":          round(units, 8),
            "position_value": round(position_value, 4),
        }

    def stats(self, trade_history):
        """Devuelve estadísticas del edge actual."""
        if len(trade_history) < 2:
            return {}

        pnls  = [t["pnl"] if isinstance(t, dict) else t for t in trade_history]
        wins  = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        p     = len(wins) / len(pnls) if pnls else 0
        avg_w = sum(wins) / len(wins) if wins else 0
        avg_l = sum(losses) / len(losses) if losses else 0
        b     = avg_w / avg_l if avg_l > 0 else 0
        f_full = (p * b - (1-p)) / b if b > 0 else 0

        return {
            "win_rate":     round(p, 4),
            "avg_win":      round(avg_w, 4),
            "avg_loss":     round(avg_l, 4),
            "profit_factor": round(b, 4),
            "full_kelly":    round(f_full, 4),
            "half_kelly":    round(f_full / 2, 4),
            "quarter_kelly": round(f_full / 4, 4),
            "applied_fraction": round(self.position_fraction(trade_history), 4),
        }
