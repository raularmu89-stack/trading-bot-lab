"""
regime_filter.py

Filtro de régimen de mercado — envuelve cualquier estrategia y solo
pasa señales cuando el mercado está en tendencia activa.

Métricas de filtrado:
  1. ADX > adx_min       → tendencia suficientemente fuerte
  2. +DI/-DI alignment   → dirección de la señal alineada con el impulso
  3. ATR% en rango óptimo → ni demasiado plano ni demasiado explosivo

Efecto esperado:
  - WR: +5-10 pp (48% → 53-58%)
  - Trades/mes: −30-40% (más selectivo)
  - Net Sharpe: ↑ (menos trades pero más precisos)

Uso:
    from strategies.regime_filter import RegimeFilteredStrategy
    from strategies.mtf_smc import MultiTFSMC

    base  = MultiTFSMC(swing_window=5, trend_ema=50)
    strat = RegimeFilteredStrategy(base, adx_min=20, adx_period=14)
    sigs  = strat.generate_signals_batch(df)
"""

import numpy as np
import pandas as pd


# ── Indicadores ───────────────────────────────────────────────────────────────

def _true_range(h, l, c):
    n = len(c)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    return tr


def _adx_dmi(h, l, c, period: int = 14):
    """Calcula ADX, +DI, -DI. Retorna tres arrays length=n."""
    n = len(c)
    tr  = _true_range(h, l, c)
    pdm = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]),
                   np.maximum(h[1:] - h[:-1], 0), 0.0)
    ndm = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]),
                   np.maximum(l[:-1] - l[1:], 0), 0.0)

    # Wilder smoothing
    def _smooth(arr, p):
        out = np.zeros(len(arr) + 1)
        out[p] = arr[:p].sum()
        for i in range(p, len(arr)):
            out[i+1] = out[i] - out[i]/p + arr[i]
        return out[1:]

    atr14  = _smooth(tr[1:],  period)
    pdm14  = _smooth(pdm,     period)
    ndm14  = _smooth(ndm,     period)

    pdi = np.where(atr14 > 0, 100 * pdm14 / atr14, 0.0)
    ndi = np.where(atr14 > 0, 100 * ndm14 / atr14, 0.0)
    dx  = np.where(pdi + ndi > 0, 100 * np.abs(pdi - ndi) / (pdi + ndi), 0.0)

    adx = np.zeros(len(dx))
    if len(dx) >= period:
        adx[period-1] = dx[:period].mean()
        for i in range(period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period

    # Pad para alinear con array original (perdemos 1 por diff)
    adx_full = np.concatenate([[0.0], adx])
    pdi_full = np.concatenate([[0.0], pdi])
    ndi_full = np.concatenate([[0.0], ndi])
    return adx_full[:n], pdi_full[:n], ndi_full[:n]


def _atr_pct(h, l, c, period: int = 14) -> np.ndarray:
    tr = _true_range(h, l, c)
    atr = np.full(len(c), np.nan)
    if len(c) < period:
        return atr
    atr[period-1] = tr[:period].mean()
    for i in range(period, len(c)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return np.where(c > 0, atr / c * 100, 0.0)


# ── Filtro de régimen ─────────────────────────────────────────────────────────

class RegimeFilteredStrategy:
    """
    Envuelve cualquier estrategia con un filtro ADX + DI + ATR%.

    Solo emite señal cuando TODOS los filtros activos pasan:
      - adx_min    : ADX > adx_min  (tendencia activa)
      - di_align   : +DI > -DI para BUY, -DI > +DI para SELL
      - atr_min_pct: ATR% > atr_min_pct  (volatilidad mínima)
      - atr_max_pct: ATR% < atr_max_pct  (no demasiado explosivo)
    """

    def __init__(
        self,
        base_strategy,
        adx_min:      float = 20.0,
        adx_period:   int   = 14,
        di_align:     bool  = True,
        atr_min_pct:  float = 0.3,
        atr_max_pct:  float = 5.0,
        atr_period:   int   = 14,
    ):
        self.base       = base_strategy
        self.adx_min    = adx_min
        self.adx_period = adx_period
        self.di_align   = di_align
        self.atr_min_pct = atr_min_pct
        self.atr_max_pct = atr_max_pct
        self.atr_period  = atr_period

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        # 1. Señales base
        base_sigs = self.base.generate_signals_batch(df)

        h = df["high"].values
        l = df["low"].values
        c = df["close"].values

        # 2. Indicadores de régimen
        adx, pdi, ndi = _adx_dmi(h, l, c, self.adx_period)
        atr_p         = _atr_pct(h, l, c, self.atr_period)

        # 3. Aplicar filtros
        result = []
        for i, sig in enumerate(base_sigs):
            if sig == "hold":
                result.append("hold")
                continue

            # ADX mínimo
            if adx[i] < self.adx_min:
                result.append("hold")
                continue

            # Alineación DI
            if self.di_align:
                if sig == "buy"  and pdi[i] <= ndi[i]:
                    result.append("hold")
                    continue
                if sig == "sell" and ndi[i] <= pdi[i]:
                    result.append("hold")
                    continue

            # ATR% rango operativo
            atp = atr_p[i]
            if not np.isnan(atp):
                if atp < self.atr_min_pct or atp > self.atr_max_pct:
                    result.append("hold")
                    continue

            result.append(sig)

        return result

    def generate_signal(self, df: pd.DataFrame) -> dict:
        sigs = self.generate_signals_batch(df)
        return {"signal": sigs[-1] if sigs else "hold"}

    def __repr__(self):
        return (f"RegimeFiltered({self.base.__class__.__name__} "
                f"ADX>{self.adx_min} DI={self.di_align} "
                f"ATR={self.atr_min_pct}-{self.atr_max_pct}%)")


# ── Configuraciones preestablecidas ──────────────────────────────────────────

def with_regime(strategy, preset: str = "standard") -> RegimeFilteredStrategy:
    """
    Envuelve una estrategia con un preset de filtro:
      "light"    : ADX>15, solo ATR mínimo
      "standard" : ADX>20, DI align, ATR 0.3-5%  (default)
      "strict"   : ADX>25, DI align, ATR 0.5-4%
      "ultra"    : ADX>30, DI align, ATR 0.5-3%
    """
    presets = {
        "light":    dict(adx_min=15, di_align=False, atr_min_pct=0.2, atr_max_pct=6.0),
        "standard": dict(adx_min=20, di_align=True,  atr_min_pct=0.3, atr_max_pct=5.0),
        "strict":   dict(adx_min=25, di_align=True,  atr_min_pct=0.5, atr_max_pct=4.0),
        "ultra":    dict(adx_min=30, di_align=True,  atr_min_pct=0.5, atr_max_pct=3.0),
    }
    return RegimeFilteredStrategy(strategy, **presets[preset])
