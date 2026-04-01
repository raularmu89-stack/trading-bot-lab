"""
feature_extractor.py

Extrae un vector de ~25 features numéricas de los últimos N candles.
Usado por la red neuronal para aprender qué condiciones predicen trades rentables.

Features extraídas:
  Momentum   : ADX, ADX_slope, plus_di, minus_di
  Volatilidad: ATR_ratio, ATR_slope, range_width
  Tendencia  : ema_fast_slope, ema_slow_slope, price_vs_ema_fast, price_vs_ema_slow
  Osciladores: RSI, RSI_slope
  Volumen    : vol_ratio
  Régimen    : one-hot de 9 categorías
  Señal      : direction (+1 buy / -1 sell)

Total: 25 features, todas normalizadas en [-1, 1] o [0, 1].
"""

import numpy as np
from indicators.regime_detector import (
    _atr, _ema, _adx, _rsi, detect_regime, RegimeDetector
)

# Orden fijo de regímenes para one-hot (debe ser estable entre versiones)
REGIME_ORDER = [
    "strong_trend_bull",
    "strong_trend_bear",
    "weak_trend_bull",
    "weak_trend_bear",
    "ranging",
    "breakout",
    "high_volatility",
    "mean_reversion_bull",
    "mean_reversion_bear",
]
N_REGIME_FEATURES = len(REGIME_ORDER)
N_FEATURES = 16 + N_REGIME_FEATURES  # 16 indicadores + 9 one-hot régimen = 25

FEATURE_NAMES = [
    "adx",
    "adx_slope",
    "plus_di",
    "minus_di",
    "atr_ratio",
    "atr_slope",
    "range_width",
    "ema_fast_slope",
    "ema_slow_slope",
    "price_vs_ema_fast",
    "price_vs_ema_slow",
    "rsi",
    "rsi_slope",
    "vol_ratio",
    "price_momentum_5",
    "price_momentum_20",
] + [f"regime_{r}" for r in REGIME_ORDER]


def extract_features(data, signal: str, lookback: int = 5) -> np.ndarray:
    """
    Extrae el vector de features de la última vela de `data`.

    Parámetros
    ----------
    data   : DataFrame OHLCV (mínimo ~60 velas para indicadores fiables)
    signal : "buy" | "sell" | "hold" — la señal a evaluar
    lookback: velas para calcular slopes

    Retorna
    -------
    np.ndarray de shape (N_FEATURES,), dtype float32.
    None si hay datos insuficientes.
    """
    if data is None or len(data) < 60:
        return None

    h = data["high"].values
    l = data["low"].values
    c = data["close"].values
    n = len(c)

    try:
        # ── Indicadores base ──────────────────────────────────────────
        adx_d    = _adx(h, l, c, 14)
        atr_arr  = _atr(h, l, c, 14)
        rsi_arr  = _rsi(c, 14)
        ema_f    = _ema(c, 20)
        ema_s    = _ema(c, 50)

        adx_val    = adx_d["adx"][-1]
        plus_di    = adx_d["plus_di"][-1]
        minus_di   = adx_d["minus_di"][-1]
        atr_val    = atr_arr[-1]
        atr_mean   = np.mean(atr_arr[max(0, n-50):n])
        atr_ratio  = atr_val / atr_mean if atr_mean > 0 else 1.0
        rsi_val    = rsi_arr[-1]
        ef         = ema_f[-1]
        es         = ema_s[-1]
        price      = c[-1]

        # ── Slopes (cambio en los últimos `lookback` velas) ───────────
        lb = min(lookback, n - 1)
        adx_slope      = (adx_d["adx"][-1]   - adx_d["adx"][-1-lb])   / (adx_d["adx"][-1-lb]   + 1e-9)
        atr_slope      = (atr_arr[-1]         - atr_arr[-1-lb])         / (atr_arr[-1-lb]         + 1e-9)
        ema_f_slope    = (ema_f[-1]           - ema_f[-1-lb])           / (ema_f[-1-lb]           + 1e-9)
        ema_s_slope    = (ema_s[-1]           - ema_s[-1-lb])           / (ema_s[-1-lb]           + 1e-9)
        rsi_slope      = (rsi_arr[-1]         - rsi_arr[-1-lb])         / 100.0

        # ── Precio vs EMAs ────────────────────────────────────────────
        price_vs_ef    = (price - ef) / ef if ef > 0 else 0.0
        price_vs_es    = (price - es) / es if es > 0 else 0.0

        # ── Ancho del rango ───────────────────────────────────────────
        rw_bars        = min(50, n)
        top            = h[n-rw_bars:n].max()
        bot            = l[n-rw_bars:n].min()
        mid            = (top + bot) / 2
        range_width    = (top - bot) / mid if mid > 0 else 0.0

        # ── Volumen ───────────────────────────────────────────────────
        if "volume" in data.columns:
            vol     = data["volume"].values
            vol_now = vol[-1]
            vol_m   = np.mean(vol[max(0, n-20):n])
            vol_ratio = vol_now / vol_m if vol_m > 0 else 1.0
        else:
            vol_ratio = 1.0

        # ── Momentum de precio ────────────────────────────────────────
        p5  = (c[-1] - c[max(0, n-6)])  / c[max(0, n-6)]  if n > 5  else 0.0
        p20 = (c[-1] - c[max(0, n-21)]) / c[max(0, n-21)] if n > 20 else 0.0

        # ── Régimen one-hot ───────────────────────────────────────────
        try:
            regime_info = detect_regime(data)
            regime      = regime_info.get("regime", "ranging")
        except Exception:
            regime = "ranging"

        one_hot = np.zeros(N_REGIME_FEATURES, dtype=np.float32)
        if regime in REGIME_ORDER:
            one_hot[REGIME_ORDER.index(regime)] = 1.0

        # ── Signo de la señal (+1 buy, -1 sell, 0 hold) ──────────────
        # Se incluye en los features para que la red aprenda asimetría bull/bear
        # Usamos price_momentum_5 con signo de la señal como proxy
        if signal == "sell":
            p5  = -p5
            p20 = -p20
            price_vs_ef = -price_vs_ef
            price_vs_es = -price_vs_es
            ema_f_slope = -ema_f_slope
            ema_s_slope = -ema_s_slope

        # ── Normalización ─────────────────────────────────────────────
        features = np.array([
            np.clip(adx_val / 50.0, 0, 2),           # ADX normalizado
            np.clip(adx_slope * 5, -1, 1),            # ADX slope
            np.clip(plus_di / 50.0, 0, 2),            # DI+
            np.clip(minus_di / 50.0, 0, 2),           # DI-
            np.clip(atr_ratio, 0, 4) / 4,             # ATR ratio
            np.clip(atr_slope * 5, -1, 1),            # ATR slope
            np.clip(range_width * 10, 0, 1),          # Range width
            np.clip(ema_f_slope * 100, -1, 1),        # EMA fast slope
            np.clip(ema_s_slope * 100, -1, 1),        # EMA slow slope
            np.clip(price_vs_ef * 20, -1, 1),         # Price vs EMA fast
            np.clip(price_vs_es * 10, -1, 1),         # Price vs EMA slow
            rsi_val / 100.0,                           # RSI
            np.clip(rsi_slope * 10, -1, 1),           # RSI slope
            np.clip(vol_ratio, 0, 5) / 5,             # Vol ratio
            np.clip(p5 * 50, -1, 1),                  # Price momentum 5
            np.clip(p20 * 20, -1, 1),                 # Price momentum 20
        ], dtype=np.float32)

        return np.concatenate([features, one_hot])

    except Exception:
        return None


def batch_extract(data_slices: list, signals: list) -> tuple:
    """
    Extrae features para una lista de (data_slice, signal).
    Retorna (X, indices_válidos) donde X tiene shape (n_valid, N_FEATURES).
    """
    X_list, valid_idx = [], []
    for i, (data, sig) in enumerate(zip(data_slices, signals)):
        f = extract_features(data, sig)
        if f is not None:
            X_list.append(f)
            valid_idx.append(i)
    if not X_list:
        return np.empty((0, N_FEATURES), dtype=np.float32), []
    return np.array(X_list, dtype=np.float32), valid_idx
