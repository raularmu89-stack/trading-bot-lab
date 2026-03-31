"""
metrics.py

Métricas de rendimiento y riesgo para evaluar estrategias de trading.

Funciones:
  sharpe_ratio    — retorno ajustado por volatilidad total
  sortino_ratio   — retorno ajustado por volatilidad bajista
  max_drawdown    — maxima caida pico a valle
  calmar_ratio    — retorno anualizado / max drawdown
  compute_all     — calcula todas las metricas de una curva de equity
"""

import numpy as np


def _returns(equity):
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return np.array([])
    return np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1e-10)


def sharpe_ratio(equity, risk_free=0.0, periods_per_year=252):
    """
    Sharpe ratio anualizado.
    risk_free      : tasa libre de riesgo anual (ej. 0.04 = 4%)
    periods_per_year: 252 (diario), 365 (crypto diario), 8760 (1h)
    """
    r = _returns(equity)
    if len(r) == 0 or np.std(r) == 0:
        return 0.0
    excess = r - risk_free / periods_per_year
    return round(float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year)), 4)


def sortino_ratio(equity, risk_free=0.0, periods_per_year=252):
    """
    Sortino ratio — igual que Sharpe pero solo penaliza la volatilidad bajista.
    Mas apropiado para estrategias asimetricas.
    """
    r = _returns(equity)
    if len(r) == 0:
        return 0.0
    excess = r - risk_free / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if np.mean(excess) > 0 else 0.0
    downside_std = np.std(downside) * np.sqrt(periods_per_year)
    if downside_std == 0:
        return 0.0
    return round(float(np.mean(excess) * periods_per_year / downside_std), 4)


def max_drawdown(equity):
    """
    Maxima caida pico a valle como fraccion negativa.
    Ej: -0.25 significa una caida del 25% desde el maximo.
    """
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak != 0, peak, 1e-10)
    return round(float(dd.min()), 4)


def drawdown_series(equity):
    """Devuelve la serie completa de drawdown (para graficar)."""
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    return (eq - peak) / np.where(peak != 0, peak, 1e-10)


def calmar_ratio(equity, periods_per_year=252):
    """
    Calmar ratio = retorno anualizado / |max drawdown|.
    Mide cuanto retorno se consigue por unidad de drawdown maximo.
    """
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2 or eq[0] == 0:
        return 0.0
    n = len(eq) - 1
    total_return = eq[-1] / eq[0] - 1
    ann_return = (1 + total_return) ** (periods_per_year / n) - 1 if n > 0 else 0.0
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return float("inf") if ann_return > 0 else 0.0
    return round(float(ann_return / mdd), 4)


def compute_all(equity, periods_per_year=252, risk_free=0.0):
    """
    Calcula todas las metricas de riesgo/rendimiento para una curva de equity.

    Devuelve dict con:
      total_return   : retorno total (fraccion)
      ann_return     : retorno anualizado
      volatility     : desviacion estandar anualizada de los retornos
      max_drawdown   : maxima caida pico a valle (negativo)
      sharpe         : Sharpe ratio anualizado
      sortino        : Sortino ratio anualizado
      calmar         : Calmar ratio
    """
    eq = np.asarray(equity, dtype=float)
    r  = _returns(eq)
    n  = len(eq) - 1

    total_return = float(eq[-1] / eq[0] - 1) if eq[0] != 0 else 0.0
    ann_return   = float((1 + total_return) ** (periods_per_year / n) - 1) if n > 0 else 0.0
    volatility   = float(np.std(r) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0

    return {
        "total_return": round(total_return, 4),
        "ann_return":   round(ann_return, 4),
        "volatility":   round(volatility, 4),
        "max_drawdown": max_drawdown(eq),
        "sharpe":       sharpe_ratio(eq, risk_free, periods_per_year),
        "sortino":      sortino_ratio(eq, risk_free, periods_per_year),
        "calmar":       calmar_ratio(eq, periods_per_year),
    }
