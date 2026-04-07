"""
price_action_strategies.py

Estrategias de Price Action + indicadores técnicos avanzados.

  1. SuperTrendStrategy    — ATR supertrend (muy usado en crypto)
  2. VWAPStrategy          — VWAP + bandas de desviación estándar
  3. HullMAStrategy        — Hull Moving Average (rápida y suave)
  4. KeltnerBreakoutStrategy — Keltner channel squeeze→breakout
  5. PinBarStrategy        — Pin bar / hammer / shooting star
  6. RSIDivergenceStrategy — Divergencia RSI precio (señal de reversión)
"""

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ema(arr, p):
    a = 2 / (p + 1)
    out = np.full(len(arr), np.nan)
    if len(arr) < p:
        return out
    out[p - 1] = arr[:p].mean()
    for i in range(p, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out

def _sma(arr, p):
    out = np.full(len(arr), np.nan)
    for i in range(p - 1, len(arr)):
        out[i] = arr[i - p + 1 : i + 1].mean()
    return out

def _atr(h, l, c, p):
    n = len(c)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    out = np.full(n, np.nan)
    if n < p:
        return out
    out[p - 1] = tr[:p].mean()
    for i in range(p, n):
        out[i] = (out[i-1] * (p-1) + tr[i]) / p
    return out

def _rsi(c, p=14):
    n = len(c)
    out = np.full(n, np.nan)
    if n < p + 1:
        return out
    d = np.diff(c)
    g = np.where(d > 0, d, 0.0)
    lo = np.where(d < 0, -d, 0.0)
    ag, al = g[:p].mean(), lo[:p].mean()
    for i in range(p, n - 1):
        ag = (ag * (p - 1) + g[i]) / p
        al = (al * (p - 1) + lo[i]) / p
        out[i + 1] = 100 - 100 / (1 + ag / (al + 1e-10))
    return out

def _wma(arr, p):
    """Weighted Moving Average."""
    weights = np.arange(1, p + 1, dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(p - 1, len(arr)):
        out[i] = np.dot(arr[i - p + 1 : i + 1], weights) / weights.sum()
    return out


# ── 1. SuperTrend ─────────────────────────────────────────────────────────────

class SuperTrendStrategy:
    """
    SuperTrend ATR — uno de los indicadores de tendencia más populares en crypto.

    Lógica:
      upper_band = (H+L)/2 + mult * ATR
      lower_band = (H+L)/2 - mult * ATR
      Trend = +1 si close > lower_band (bullish)
      Trend = -1 si close < upper_band (bearish)

    Señal: cruce de tendencia → buy/sell
    """

    def __init__(self, atr_period: int = 10, multiplier: float = 3.0):
        self.atr_period = atr_period
        self.multiplier = multiplier

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        n = len(c)

        atr = _atr(h, l, c, self.atr_period)
        hl2 = (h + l) / 2

        upper = hl2 + self.multiplier * atr
        lower = hl2 - self.multiplier * atr

        # Supertrend con ratchet (no se mueve en contra de la tendencia)
        st = np.full(n, np.nan)
        trend = np.zeros(n, dtype=int)

        for i in range(1, n):
            if np.isnan(atr[i]):
                continue

            # Ajustar bandas
            if not np.isnan(st[i - 1]):
                if lower[i] < lower[i - 1]:
                    lower[i] = lower[i - 1]
                if upper[i] > upper[i - 1]:
                    upper[i] = upper[i - 1]

            # Determinar tendencia
            if c[i] > upper[i - 1] if not np.isnan(st[i-1]) else c[i] > upper[i]:
                trend[i] = 1
                st[i]    = lower[i]
            elif c[i] < lower[i - 1] if not np.isnan(st[i-1]) else c[i] < lower[i]:
                trend[i] = -1
                st[i]    = upper[i]
            else:
                trend[i] = trend[i - 1]
                st[i]    = st[i - 1]

        sigs = ["hold"] * n
        for i in range(1, n):
            if trend[i] == 1 and trend[i - 1] == -1:
                sigs[i] = "buy"
            elif trend[i] == -1 and trend[i - 1] == 1:
                sigs[i] = "sell"

        return sigs


# ── 2. VWAP ───────────────────────────────────────────────────────────────────

class VWAPStrategy:
    """
    VWAP + bandas de desviación estándar.

    Buy:  precio cruza al alza VWAP + está por encima de VWAP − 1σ
    Sell: precio cruza a la baja VWAP − está por debajo de VWAP + 1σ

    VWAP se reinicia cada N velas (simula sesión diaria en 1H → 24 velas).
    """

    def __init__(self, session_len: int = 24, band_mult: float = 1.5):
        self.session_len = session_len
        self.band_mult   = band_mult

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        h   = df["high"].values
        l   = df["low"].values
        c   = df["close"].values
        v   = df["volume"].values
        n   = len(c)
        tp  = (h + l + c) / 3   # typical price

        vwap  = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)

        for i in range(n):
            sess_start = (i // self.session_len) * self.session_len
            sl = slice(sess_start, i + 1)
            cum_vol = v[sl].sum()
            if cum_vol <= 0:
                continue
            vw = (tp[sl] * v[sl]).sum() / cum_vol
            vwap[i] = vw
            if i - sess_start >= 1:
                std = np.std(tp[sl])
                upper[i] = vw + self.band_mult * std
                lower[i] = vw - self.band_mult * std

        sigs = ["hold"] * n
        for i in range(1, n):
            if np.isnan(vwap[i]) or np.isnan(vwap[i-1]):
                continue
            # Precio cruza VWAP al alza desde debajo
            if c[i] > vwap[i] and c[i-1] <= vwap[i-1]:
                sigs[i] = "buy"
            # Precio cruza VWAP a la baja
            elif c[i] < vwap[i] and c[i-1] >= vwap[i-1]:
                sigs[i] = "sell"

        return sigs


# ── 3. Hull MA ────────────────────────────────────────────────────────────────

class HullMAStrategy:
    """
    Hull Moving Average — más rápida y suave que EMA.

    HMA(n) = WMA(2*WMA(n/2) - WMA(n), sqrt(n))

    Señal: cruce de HMA rápida sobre HMA lenta, o cambio de pendiente HMA.
    """

    def __init__(self, fast: int = 20, slow: int = 55):
        self.fast = fast
        self.slow = slow

    @staticmethod
    def _hma(arr, p):
        half = max(1, p // 2)
        sqrp = max(1, int(np.sqrt(p)))
        w1 = _wma(arr, half)
        w2 = _wma(arr, p)
        diff = np.where(np.isnan(w1) | np.isnan(w2), np.nan, 2 * w1 - w2)
        return _wma(np.nan_to_num(diff, nan=0.0), sqrp)

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        c    = df["close"].values
        hf   = self._hma(c, self.fast)
        hs   = self._hma(c, self.slow)
        n    = len(c)
        sigs = ["hold"] * n

        for i in range(1, n):
            if np.isnan(hf[i]) or np.isnan(hs[i]):
                continue
            if hf[i] > hs[i] and hf[i-1] <= hs[i-1]:
                sigs[i] = "buy"
            elif hf[i] < hs[i] and hf[i-1] >= hs[i-1]:
                sigs[i] = "sell"

        return sigs


# ── 4. Keltner Channel Breakout ───────────────────────────────────────────────

class KeltnerBreakoutStrategy:
    """
    Keltner Channel breakout — squeeze + expansión de volatilidad.

    KC_upper = EMA(close, n) + mult * ATR(n)
    KC_lower = EMA(close, n) - mult * ATR(n)

    Buy:  precio cierra por encima de KC_upper (breakout alcista)
    Sell: precio cierra por debajo de KC_lower (breakout bajista)

    Opcionalmente filtra por squeeze previo (BB < KC).
    """

    def __init__(self, ema_period: int = 20, atr_period: int = 20,
                 mult: float = 1.5, squeeze_filter: bool = True):
        self.ema_period     = ema_period
        self.atr_period     = atr_period
        self.mult           = mult
        self.squeeze_filter = squeeze_filter

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        n = len(c)

        ema  = _ema(c, self.ema_period)
        atr  = _atr(h, l, c, self.atr_period)
        kc_u = ema + self.mult * atr
        kc_l = ema - self.mult * atr

        # BB para squeeze
        if self.squeeze_filter:
            std  = np.array([np.std(c[max(0, i-self.ema_period+1):i+1])
                             if i >= self.ema_period-1 else np.nan
                             for i in range(n)])
            bb_u = ema + 2.0 * std
            bb_l = ema - 2.0 * std
            in_squeeze = (bb_u < kc_u) & (bb_l > kc_l)
        else:
            in_squeeze = np.ones(n, dtype=bool)

        sigs = ["hold"] * n
        for i in range(1, n):
            if np.isnan(kc_u[i]) or np.isnan(kc_l[i]):
                continue
            # Solo en breakout tras squeeze
            squeeze_before = in_squeeze[max(0, i-5):i].any() \
                             if self.squeeze_filter else True

            if c[i] > kc_u[i] and c[i-1] <= kc_u[i-1] and squeeze_before:
                sigs[i] = "buy"
            elif c[i] < kc_l[i] and c[i-1] >= kc_l[i-1] and squeeze_before:
                sigs[i] = "sell"

        return sigs


# ── 5. Pin Bar ────────────────────────────────────────────────────────────────

class PinBarStrategy:
    """
    Pin Bar (Hammer / Shooting Star) — rechazo de precio con mecha larga.

    Hammer (buy):  mecha inferior >= 2× cuerpo, cierre en tercio superior
    Shooting star (sell): mecha superior >= 2× cuerpo, cierre en tercio inferior

    Filtro: EMA trend para no ir contra la tendencia.
    """

    def __init__(self, ema_trend: int = 50, wick_ratio: float = 2.0,
                 body_pct_max: float = 0.4):
        self.ema_trend    = ema_trend
        self.wick_ratio   = wick_ratio
        self.body_pct_max = body_pct_max

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        n = len(c)

        ema = _ema(c, self.ema_trend)
        sigs = ["hold"] * n

        for i in range(1, n):
            if np.isnan(ema[i]):
                continue

            rng  = h[i] - l[i]
            if rng < 1e-10:
                continue

            body      = abs(c[i] - o[i])
            body_pct  = body / rng

            if body_pct > self.body_pct_max:
                continue   # cuerpo demasiado grande, no es pin bar

            upper_wick = h[i] - max(o[i], c[i])
            lower_wick = min(o[i], c[i]) - l[i]

            # Hammer — mecha larga abajo, tendencia alcista
            if (lower_wick >= self.wick_ratio * body and
                    upper_wick <= body and
                    c[i] > ema[i]):
                sigs[i] = "buy"

            # Shooting star — mecha larga arriba, tendencia bajista
            elif (upper_wick >= self.wick_ratio * body and
                    lower_wick <= body and
                    c[i] < ema[i]):
                sigs[i] = "sell"

        return sigs


# ── 6. RSI Divergence ─────────────────────────────────────────────────────────

class RSIDivergenceStrategy:
    """
    Divergencia RSI — señal potente de reversión de tendencia.

    Divergencia alcista (buy):  precio hace Lower Low, RSI hace Higher Low
    Divergencia bajista (sell): precio hace Higher High, RSI hace Lower High

    Lookback: busca divergencia en ventana de N velas.
    """

    def __init__(self, rsi_period: int = 14, lookback: int = 20,
                 rsi_ob: float = 65.0, rsi_os: float = 35.0,
                 ema_trend: int = 100):
        self.rsi_period = rsi_period
        self.lookback   = lookback
        self.rsi_ob     = rsi_ob
        self.rsi_os     = rsi_os
        self.ema_trend  = ema_trend

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        c    = df["close"].values
        n    = len(c)
        rsi  = _rsi(c, self.rsi_period)
        ema  = _ema(c, self.ema_trend)
        sigs = ["hold"] * n

        lb = self.lookback

        for i in range(lb, n):
            if np.isnan(rsi[i]) or np.isnan(ema[i]):
                continue

            w_c   = c[i - lb : i + 1]
            w_rsi = rsi[i - lb : i + 1]

            if np.any(np.isnan(w_rsi)):
                continue

            # ── Divergencia alcista ───────────────────────────────────
            # Precio: nuevo mínimo | RSI: mínimo más alto
            price_min_now  = w_c[-1]  == w_c.min()
            price_prev_low = w_c[:-1].min()
            rsi_prev_low   = w_rsi[np.argmin(w_c[:-1])]

            if (price_min_now and
                    w_c[-1] < price_prev_low and
                    w_rsi[-1] > rsi_prev_low and
                    w_rsi[-1] < self.rsi_os + 15 and
                    c[i] > ema[i] * 0.97):   # no demasiado lejos de tendencia
                sigs[i] = "buy"
                continue

            # ── Divergencia bajista ───────────────────────────────────
            # Precio: nuevo máximo | RSI: máximo más bajo
            price_max_now  = w_c[-1] == w_c.max()
            price_prev_high = w_c[:-1].max()
            rsi_prev_high   = w_rsi[np.argmax(w_c[:-1])]

            if (price_max_now and
                    w_c[-1] > price_prev_high and
                    w_rsi[-1] < rsi_prev_high and
                    w_rsi[-1] > self.rsi_ob - 15 and
                    c[i] < ema[i] * 1.03):
                sigs[i] = "sell"

        return sigs
