"""
momentum_oscillator_strategies.py

8 nuevas estrategias basadas en osciladores de momentum y volumen.
Todas generan señales frecuentes en 1H — optimizadas para trading activo.

  1. AwesomeOscillatorStrategy — AO cruce de cero con aceleración
  2. CCIStrategy               — CCI(20) extremos ±100 con EMA filtro
  3. WilliamsRStrategy         — %R sobrecompra/sobreventa con tendencia
  4. SqueezeMomentumStrategy   — BB inside KC = squeeze → momentum burst
  5. ChaikinMFStrategy         — Chaikin Money Flow + precio cruce EMA
  6. ElderRayStrategy          — Bull/Bear Power de Elder con EMA13
  7. HeikinAshiEMAStrategy     — Velas Heikin Ashi + doble EMA filtro
  8. ParabolicSARStrategy      — SAR Parabólico + EMA tendencia
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


# ── 1. Awesome Oscillator ─────────────────────────────────────────────────────

class AwesomeOscillatorStrategy:
    """
    AO = SMA(midpoint, 5) - SMA(midpoint, 34)   midpoint = (H+L)/2

    Señal:
      Buy:  AO cruza cero al alza (saucer o twin peaks bullish)
      Sell: AO cruza cero a la baja

    Filtro extra: aceleración — AO[i] > AO[i-1] para buy
    """

    def __init__(self, fast: int = 5, slow: int = 34, ema_trend: int = 50):
        self.fast      = fast
        self.slow      = slow
        self.ema_trend = ema_trend

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        h   = df["high"].values
        l   = df["low"].values
        c   = df["close"].values
        mid = (h + l) / 2

        ao   = _sma(mid, self.fast) - _sma(mid, self.slow)
        ema  = _ema(c, self.ema_trend)
        n    = len(c)
        sigs = ["hold"] * n

        for i in range(1, n):
            if np.isnan(ao[i]) or np.isnan(ao[i-1]) or np.isnan(ema[i]):
                continue
            # Cruce de cero + aceleración + filtro tendencia
            if ao[i] > 0 and ao[i-1] <= 0 and ao[i] > ao[i-1] and c[i] > ema[i]:
                sigs[i] = "buy"
            elif ao[i] < 0 and ao[i-1] >= 0 and ao[i] < ao[i-1] and c[i] < ema[i]:
                sigs[i] = "sell"

        return sigs


# ── 2. CCI Strategy ───────────────────────────────────────────────────────────

class CCIStrategy:
    """
    CCI(period) = (TP - SMA(TP)) / (0.015 * MeanDeviation)

    Buy:  CCI sale de sobreventa (<-100) al alza + precio > EMA
    Sell: CCI sale de sobrecompra (>+100) a la baja + precio < EMA
    """

    def __init__(self, period: int = 20, ema_trend: int = 50,
                 ob: float = 100.0, os: float = -100.0):
        self.period    = period
        self.ema_trend = ema_trend
        self.ob        = ob
        self.os        = os

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        h  = df["high"].values
        l  = df["low"].values
        c  = df["close"].values
        tp = (h + l + c) / 3
        n  = len(c)

        cci = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            sl   = tp[i - self.period + 1 : i + 1]
            mean = sl.mean()
            mad  = np.mean(np.abs(sl - mean))
            cci[i] = (tp[i] - mean) / (0.015 * mad) if mad > 0 else 0.0

        ema  = _ema(c, self.ema_trend)
        sigs = ["hold"] * n

        for i in range(1, n):
            if np.isnan(cci[i]) or np.isnan(cci[i-1]) or np.isnan(ema[i]):
                continue
            if cci[i] > self.os and cci[i-1] <= self.os and c[i] > ema[i]:
                sigs[i] = "buy"
            elif cci[i] < self.ob and cci[i-1] >= self.ob and c[i] < ema[i]:
                sigs[i] = "sell"

        return sigs


# ── 3. Williams %R ────────────────────────────────────────────────────────────

class WilliamsRStrategy:
    """
    %R = (Highest_High - Close) / (Highest_High - Lowest_Low) * -100

    Buy:  %R sale de -80..-100 (sobreventa) + EMA alcista
    Sell: %R sale de 0..-20 (sobrecompra) + EMA bajista
    """

    def __init__(self, period: int = 14, ema_trend: int = 50,
                 os: float = -80.0, ob: float = -20.0):
        self.period    = period
        self.ema_trend = ema_trend
        self.os        = os
        self.ob        = ob

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        h  = df["high"].values
        l  = df["low"].values
        c  = df["close"].values
        n  = len(c)

        wr = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            hh = h[i - self.period + 1 : i + 1].max()
            ll = l[i - self.period + 1 : i + 1].min()
            wr[i] = (hh - c[i]) / (hh - ll) * -100 if (hh - ll) > 0 else -50.0

        ema  = _ema(c, self.ema_trend)
        sigs = ["hold"] * n

        for i in range(1, n):
            if np.isnan(wr[i]) or np.isnan(wr[i-1]) or np.isnan(ema[i]):
                continue
            if wr[i] > self.os and wr[i-1] <= self.os and c[i] > ema[i]:
                sigs[i] = "buy"
            elif wr[i] < self.ob and wr[i-1] >= self.ob and c[i] < ema[i]:
                sigs[i] = "sell"

        return sigs


# ── 4. Squeeze Momentum ───────────────────────────────────────────────────────

class SqueezeMomentumStrategy:
    """
    LazyBear Squeeze Momentum adaptado.

    Squeeze ON  = BB inside KC (mercado comprimido)
    Squeeze OFF = BB expande fuera de KC → dirección = momentum histograma

    Momentum = regresión lineal de (close - media(BB, KC)) sobre N velas
    """

    def __init__(self, bb_period: int = 20, kc_period: int = 20,
                 bb_mult: float = 2.0, kc_mult: float = 1.5,
                 mom_period: int = 12):
        self.bb_period  = bb_period
        self.kc_period  = kc_period
        self.bb_mult    = bb_mult
        self.kc_mult    = kc_mult
        self.mom_period = mom_period

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        h  = df["high"].values
        l  = df["low"].values
        c  = df["close"].values
        n  = len(c)

        # BB
        bb_mid = _sma(c, self.bb_period)
        bb_std = np.array([
            np.std(c[max(0, i-self.bb_period+1):i+1]) if i >= self.bb_period-1 else np.nan
            for i in range(n)])
        bb_upper = bb_mid + self.bb_mult * bb_std
        bb_lower = bb_mid - self.bb_mult * bb_std

        # KC
        atr_kc   = _atr(h, l, c, self.kc_period)
        ema_kc   = _ema(c, self.kc_period)
        kc_upper = ema_kc + self.kc_mult * atr_kc
        kc_lower = ema_kc - self.kc_mult * atr_kc

        # Squeeze
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        # Momentum linreg
        def _linreg_val(arr, i, p):
            if i < p - 1 or np.any(np.isnan(arr[i-p+1:i+1])):
                return np.nan
            x = np.arange(p, dtype=float)
            y = arr[i-p+1:i+1]
            xm, ym = x.mean(), y.mean()
            denom = ((x - xm)**2).sum()
            if denom == 0:
                return 0.0
            b = ((x - xm) * (y - ym)).sum() / denom
            return b * (p - 1) + (ym - b * xm)

        # Valor del momentum = close - media de (midpoint BB, midpoint KC)
        delta = np.full(n, np.nan)
        for i in range(n):
            if np.isnan(bb_mid[i]) or np.isnan(ema_kc[i]):
                continue
            highest = max(h[max(0,i-self.kc_period+1):i+1])
            lowest  = min(l[max(0,i-self.kc_period+1):i+1])
            mid_hl  = (highest + lowest) / 2
            mid_all = (bb_mid[i] + mid_hl) / 2
            delta[i] = c[i] - mid_all

        mom = np.array([_linreg_val(delta, i, self.mom_period) for i in range(n)])

        sigs = ["hold"] * n
        in_squeeze = False

        for i in range(1, n):
            if np.isnan(mom[i]) or np.isnan(mom[i-1]):
                continue
            was_squeeze = squeeze[i-1] if i > 0 else False
            is_squeeze  = squeeze[i]

            # Squeeze release: salida del squeeze
            if was_squeeze and not is_squeeze:
                if mom[i] > 0 and mom[i] > mom[i-1]:
                    sigs[i] = "buy"
                elif mom[i] < 0 and mom[i] < mom[i-1]:
                    sigs[i] = "sell"
            # Momentum cruce de cero fuera del squeeze
            elif not is_squeeze:
                if mom[i] > 0 and mom[i-1] <= 0:
                    sigs[i] = "buy"
                elif mom[i] < 0 and mom[i-1] >= 0:
                    sigs[i] = "sell"

        return sigs


# ── 5. Chaikin Money Flow ─────────────────────────────────────────────────────

class ChaikinMFStrategy:
    """
    CMF = SUM(MFV, N) / SUM(Volume, N)
    MFV = ((Close-Low)-(High-Close)) / (High-Low) * Volume

    Buy:  CMF > 0.1 (dinero entrando) + precio > EMA
    Sell: CMF < -0.1 (dinero saliendo) + precio < EMA
    """

    def __init__(self, period: int = 20, ema_trend: int = 50,
                 threshold: float = 0.10):
        self.period    = period
        self.ema_trend = ema_trend
        self.threshold = threshold

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        h  = df["high"].values
        l  = df["low"].values
        c  = df["close"].values
        v  = df["volume"].values
        n  = len(c)

        rng = h - l
        mfm = np.where(rng > 0, ((c - l) - (h - c)) / rng, 0.0)
        mfv = mfm * v

        cmf = np.full(n, np.nan)
        for i in range(self.period - 1, n):
            sv = v[i-self.period+1:i+1].sum()
            cmf[i] = mfv[i-self.period+1:i+1].sum() / sv if sv > 0 else 0.0

        ema  = _ema(c, self.ema_trend)
        sigs = ["hold"] * n

        for i in range(1, n):
            if np.isnan(cmf[i]) or np.isnan(cmf[i-1]) or np.isnan(ema[i]):
                continue
            if cmf[i] > self.threshold and cmf[i-1] <= self.threshold and c[i] > ema[i]:
                sigs[i] = "buy"
            elif cmf[i] < -self.threshold and cmf[i-1] >= -self.threshold and c[i] < ema[i]:
                sigs[i] = "sell"

        return sigs


# ── 6. Elder Ray ──────────────────────────────────────────────────────────────

class ElderRayStrategy:
    """
    Elder Ray Index:
      Bull Power = High - EMA(13)
      Bear Power = Low  - EMA(13)

    Buy:  Bear Power negativo pero subiendo (compresión bajista → rebote)
          + EMA subiendo
    Sell: Bull Power positivo pero bajando (compresión alcista → caída)
          + EMA bajando
    """

    def __init__(self, ema_period: int = 13, trend_ema: int = 50):
        self.ema_period = ema_period
        self.trend_ema  = trend_ema

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        h  = df["high"].values
        l  = df["low"].values
        c  = df["close"].values
        n  = len(c)

        ema13 = _ema(c, self.ema_period)
        ema50 = _ema(c, self.trend_ema)
        bull  = h - ema13
        bear  = l - ema13
        sigs  = ["hold"] * n

        for i in range(1, n):
            if np.isnan(ema13[i]) or np.isnan(ema50[i]):
                continue
            trend_up   = ema50[i] > ema50[i-1]
            trend_down = ema50[i] < ema50[i-1]

            # Acumulación: bear power negativo pero aumentando en tendencia alcista
            if (bear[i] < 0 and bear[i] > bear[i-1] and
                    bull[i] > bull[i-1] and trend_up):
                sigs[i] = "buy"
            # Distribución: bull power positivo pero disminuyendo en tendencia bajista
            elif (bull[i] > 0 and bull[i] < bull[i-1] and
                    bear[i] < bear[i-1] and trend_down):
                sigs[i] = "sell"

        return sigs


# ── 7. Heikin Ashi + EMA ──────────────────────────────────────────────────────

class HeikinAshiEMAStrategy:
    """
    Velas Heikin Ashi + confirmación de doble EMA.

    HA_Close = (O+H+L+C)/4
    HA_Open  = (HA_Open_prev + HA_Close_prev) / 2
    HA_High  = max(H, HA_Open, HA_Close)
    HA_Low   = min(L, HA_Open, HA_Close)

    Buy:  3+ velas HA verdes consecutivas + HA_Low subiendo + EMA fast > EMA slow
    Sell: 3+ velas HA rojas consecutivas + HA_High bajando + EMA fast < EMA slow
    """

    def __init__(self, ema_fast: int = 20, ema_slow: int = 50,
                 consecutive: int = 3):
        self.ema_fast    = ema_fast
        self.ema_slow    = ema_slow
        self.consecutive = consecutive

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        o  = df["open"].values
        h  = df["high"].values
        l  = df["low"].values
        c  = df["close"].values
        n  = len(c)

        # Calcular Heikin Ashi
        ha_c = (o + h + l + c) / 4
        ha_o = np.zeros(n)
        ha_h = np.zeros(n)
        ha_l = np.zeros(n)

        ha_o[0] = (o[0] + c[0]) / 2
        for i in range(1, n):
            ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2
        ha_h = np.maximum(h, np.maximum(ha_o, ha_c))
        ha_l = np.minimum(l, np.minimum(ha_o, ha_c))

        ha_green = ha_c > ha_o   # vela verde HA
        ha_red   = ha_c < ha_o   # vela roja HA

        ef   = _ema(c, self.ema_fast)
        es   = _ema(c, self.ema_slow)
        sigs = ["hold"] * n
        k    = self.consecutive

        for i in range(k, n):
            if np.isnan(ef[i]) or np.isnan(es[i]):
                continue

            all_green = all(ha_green[i-k+1:i+1])
            all_red   = all(ha_red[i-k+1:i+1])
            ema_up    = ef[i] > es[i] and ef[i] > ef[i-1]
            ema_down  = ef[i] < es[i] and ef[i] < ef[i-1]

            # Primera señal de la racha verde
            if all_green and not ha_green[i-k] and ema_up:
                sigs[i] = "buy"
            elif all_red and not ha_red[i-k] and ema_down:
                sigs[i] = "sell"

        return sigs


# ── 8. Parabolic SAR ──────────────────────────────────────────────────────────

class ParabolicSARStrategy:
    """
    Parabolic SAR clásico con filtro EMA de tendencia.

    Aceleración inicial (AF) = 0.02, max = 0.20
    SAR sube en tendencia alcista, baja en tendencia bajista.

    Buy:  precio supera SAR (cambio a tendencia alcista) + EMA subiendo
    Sell: precio cae bajo SAR (cambio a tendencia bajista) + EMA bajando
    """

    def __init__(self, af_start: float = 0.02, af_step: float = 0.02,
                 af_max: float = 0.20, ema_trend: int = 50):
        self.af_start  = af_start
        self.af_step   = af_step
        self.af_max    = af_max
        self.ema_trend = ema_trend

    def generate_signals_batch(self, df: pd.DataFrame) -> list:
        h  = df["high"].values
        l  = df["low"].values
        c  = df["close"].values
        n  = len(c)

        sar   = np.full(n, np.nan)
        trend = np.zeros(n, dtype=int)   # +1 alcista, -1 bajista

        # Inicializar
        trend[0] = 1
        sar[0]   = l[0]
        ep       = h[0]   # extreme point
        af       = self.af_start

        for i in range(1, n):
            prev_sar   = sar[i-1]
            prev_trend = trend[i-1]

            if prev_trend == 1:
                new_sar = prev_sar + af * (ep - prev_sar)
                new_sar = min(new_sar, l[i-1], l[max(0,i-2)])
                if l[i] < new_sar:
                    trend[i] = -1
                    sar[i]   = ep
                    ep       = l[i]
                    af       = self.af_start
                else:
                    trend[i] = 1
                    sar[i]   = new_sar
                    if h[i] > ep:
                        ep = h[i]
                        af = min(af + self.af_step, self.af_max)
            else:
                new_sar = prev_sar - af * (prev_sar - ep)
                new_sar = max(new_sar, h[i-1], h[max(0,i-2)])
                if h[i] > new_sar:
                    trend[i] = 1
                    sar[i]   = ep
                    ep       = h[i]
                    af       = self.af_start
                else:
                    trend[i] = -1
                    sar[i]   = new_sar
                    if l[i] < ep:
                        ep = l[i]
                        af = min(af + self.af_step, self.af_max)

        ema  = _ema(c, self.ema_trend)
        sigs = ["hold"] * n

        for i in range(1, n):
            if np.isnan(ema[i]):
                continue
            ema_up   = ema[i] > ema[i-1]
            ema_down = ema[i] < ema[i-1]
            if trend[i] == 1 and trend[i-1] == -1 and ema_up:
                sigs[i] = "buy"
            elif trend[i] == -1 and trend[i-1] == 1 and ema_down:
                sigs[i] = "sell"

        return sigs
