"""
advanced_strategies.py

10 estrategias avanzadas de alta calidad para el torneo final.
Todas con generate_signals_batch(data) -> list O(n).

  1.  StochRSIStrategy        — StochRSI cruce en zonas extremas
  2.  IchimokuStrategy        — Tenkan/Kijun cross + nube Ichimoku
  3.  FibRetracementStrategy  — rebote en niveles Fibonacci 0.382/0.618
  4.  MarketStructureStrategy — HH/HL (uptrend) o LH/LL (downtrend)
  5.  MACDRSIStrategy         — confluencia MACD histogram + RSI
  6.  LinearRegStrategy       — ruptura del canal de regresión lineal
  7.  BollingerMomStrategy    — BB squeeze + momentum burst
  8.  OrderBlockStrategy      — SMC order block (último swing con gran vela)
  9.  BreakoutVolStrategy     — ruptura N-bar high/low + volumen 2×
  10. AdaptiveMTFStrategy     — MTF-SMC con EMA adaptativa por volatilidad
"""

import numpy as np
import pandas as pd


# ── Indicadores compartidos ───────────────────────────────────────────────────

def _ema(arr, p):
    a = 2 / (p + 1); out = np.full(len(arr), np.nan)
    if len(arr) < p: return out
    out[p-1] = arr[:p].mean()
    for i in range(p, len(arr)): out[i] = a*arr[i] + (1-a)*out[i-1]
    return out

def _sma(arr, p):
    out = np.full(len(arr), np.nan)
    for i in range(p-1, len(arr)): out[i] = arr[i-p+1:i+1].mean()
    return out

def _std(arr, p):
    out = np.full(len(arr), np.nan)
    for i in range(p-1, len(arr)): out[i] = arr[i-p+1:i+1].std(ddof=1)
    return out

def _atr(h, l, c, p):
    n = len(c); tr = np.zeros(n); tr[0] = h[0]-l[0]
    for i in range(1, n): tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    out = np.full(n, np.nan)
    if n < p: return out
    out[p-1] = tr[:p].mean()
    for i in range(p, n): out[i] = (out[i-1]*(p-1)+tr[i])/p
    return out

def _rsi(c, p=14):
    n = len(c); out = np.full(n, np.nan)
    if n < p+1: return out
    d = np.diff(c); g = np.where(d>0,d,0.); ll = np.where(d<0,-d,0.)
    ag, al = g[:p].mean(), ll[:p].mean()
    for i in range(p, n-1):
        ag = (ag*(p-1)+g[i])/p; al = (al*(p-1)+ll[i])/p
        out[i+1] = 100-100/(1+ag/(al+1e-10))
    return out

def _macd(c, fast=12, slow=26, sig=9):
    ef = _ema(c, fast); es = _ema(c, slow)
    line = ef - es; signal = _ema(np.where(np.isnan(line), 0, line), sig)
    signal[np.isnan(line)] = np.nan
    return line, signal, line - signal


# ── Base ──────────────────────────────────────────────────────────────────────

class _Base:
    swing_window = 5; require_fvg = False; use_choch_filter = False
    _min_bars = 60

    def generate_signal(self, data):
        if data is None or len(data) < self._min_bars: return {"signal":"hold"}
        s = self.generate_signals_batch(data)
        return {"signal": s[-1] if s else "hold"}

    def generate_signals_batch(self, data): raise NotImplementedError


# ════════════════════════════════════════════════════════════════════════════
# 1. Stochastic RSI
# ════════════════════════════════════════════════════════════════════════════
class StochRSIStrategy(_Base):
    """StochRSI K/D cruce saliendo de zonas extremas con filtro EMA."""
    def __init__(self, rsi_p=14, stoch_p=14, k_smooth=3, d_smooth=3,
                 oversold=20, overbought=80, ema_trend=50):
        self.rsi_p=rsi_p; self.stoch_p=stoch_p; self.k_smooth=k_smooth
        self.d_smooth=d_smooth; self.oversold=oversold; self.overbought=overbought
        self.ema_trend=ema_trend; self._min_bars=rsi_p+stoch_p+k_smooth+d_smooth+10
        self.swing_window=rsi_p

    def generate_signals_batch(self, data):
        c = data["close"].values; n = len(c)
        rsi = _rsi(c, self.rsi_p)
        # Stoch of RSI
        k_raw = np.full(n, np.nan)
        for i in range(self.stoch_p-1, n):
            w = rsi[i-self.stoch_p+1:i+1]
            if np.any(~np.isnan(w)):
                mn,mx = np.nanmin(w), np.nanmax(w)
                k_raw[i] = 100*(rsi[i]-mn)/(mx-mn+1e-10) if mx>mn else 50
        K = _sma(np.where(np.isnan(k_raw),0,k_raw), self.k_smooth)
        K[np.isnan(k_raw)] = np.nan
        D = _sma(np.where(np.isnan(K),0,K), self.d_smooth)
        D[np.isnan(K)] = np.nan
        et = _ema(c, self.ema_trend)
        sigs = ["hold"]*n
        for i in range(1, n):
            if any(np.isnan([K[i],K[i-1],D[i],D[i-1],et[i]])): continue
            cross_up   = K[i-1]<D[i-1] and K[i]>D[i] and K[i]<self.oversold+20
            cross_down = K[i-1]>D[i-1] and K[i]<D[i] and K[i]>self.overbought-20
            if cross_up   and c[i]>et[i]: sigs[i]="buy"
            elif cross_down and c[i]<et[i]: sigs[i]="sell"
        return sigs

    def __repr__(self): return f"StochRSI(rsi={self.rsi_p},et={self.ema_trend})"


# ════════════════════════════════════════════════════════════════════════════
# 2. Ichimoku
# ════════════════════════════════════════════════════════════════════════════
class IchimokuStrategy(_Base):
    """Tenkan/Kijun cross encima/debajo de la nube Kumo."""
    def __init__(self, tenkan=9, kijun=26, senkou_b=52, ema_trend=50):
        self.tenkan=tenkan; self.kijun=kijun; self.senkou_b=senkou_b
        self.ema_trend=ema_trend; self._min_bars=senkou_b+kijun+10
        self.swing_window=kijun

    def _mid(self, h, l, p):
        out = np.full(len(h), np.nan)
        for i in range(p-1, len(h)):
            out[i] = (h[i-p+1:i+1].max()+l[i-p+1:i+1].min())/2
        return out

    def generate_signals_batch(self, data):
        h=data["high"].values; l=data["low"].values; c=data["close"].values; n=len(c)
        tenkan  = self._mid(h,l,self.tenkan)
        kijun   = self._mid(h,l,self.kijun)
        senkou_a= (tenkan+kijun)/2
        senkou_b= self._mid(h,l,self.senkou_b)
        et = _ema(c, self.ema_trend)
        sigs=["hold"]*n
        for i in range(1,n):
            if any(np.isnan([tenkan[i],kijun[i],senkou_a[i],senkou_b[i],et[i]])): continue
            cloud_top = max(senkou_a[i], senkou_b[i])
            cloud_bot = min(senkou_a[i], senkou_b[i])
            cross_bull = tenkan[i-1]<=kijun[i-1] and tenkan[i]>kijun[i]
            cross_bear = tenkan[i-1]>=kijun[i-1] and tenkan[i]<kijun[i]
            above_cloud = c[i] > cloud_top
            below_cloud = c[i] < cloud_bot
            if cross_bull and above_cloud and c[i]>et[i]: sigs[i]="buy"
            elif cross_bear and below_cloud and c[i]<et[i]: sigs[i]="sell"
        return sigs

    def __repr__(self): return f"Ichimoku(t={self.tenkan},k={self.kijun})"


# ════════════════════════════════════════════════════════════════════════════
# 3. Fibonacci Retracement
# ════════════════════════════════════════════════════════════════════════════
class FibRetracementStrategy(_Base):
    """Precio rebota en niveles Fibonacci 0.382 o 0.618 del último swing."""
    def __init__(self, swing_lookback=50, fib_levels=(0.382,0.5,0.618),
                 tolerance=0.005, ema_trend=50):
        self.swing_lookback=swing_lookback; self.fib_levels=fib_levels
        self.tolerance=tolerance; self.ema_trend=ema_trend
        self._min_bars=swing_lookback+ema_trend+5; self.swing_window=swing_lookback//5

    def generate_signals_batch(self, data):
        h=data["high"].values; l=data["low"].values; c=data["close"].values; n=len(c)
        et=_ema(c,self.ema_trend); sigs=["hold"]*n
        lb=self.swing_lookback
        for i in range(lb, n):
            if np.isnan(et[i]): continue
            window_h = h[i-lb:i]; window_l = l[i-lb:i]
            swing_high = window_h.max(); swing_low = window_l.min()
            rng = swing_high - swing_low
            if rng < 1e-10: continue
            for fib in self.fib_levels:
                # bull: precio en retroceso fib desde swing_high
                bull_level = swing_high - fib*rng
                bear_level = swing_low  + fib*rng
                near_bull = abs(c[i]-bull_level)/(bull_level+1e-10) < self.tolerance
                near_bear = abs(c[i]-bear_level)/(bear_level+1e-10) < self.tolerance
                if near_bull and c[i]>et[i]:   sigs[i]="buy";  break
                if near_bear and c[i]<et[i]:   sigs[i]="sell"; break
        return sigs

    def __repr__(self): return f"FibRetracement(lb={self.swing_lookback},et={self.ema_trend})"


# ════════════════════════════════════════════════════════════════════════════
# 4. Market Structure
# ════════════════════════════════════════════════════════════════════════════
class MarketStructureStrategy(_Base):
    """HH+HL = uptrend (buy pullback); LH+LL = downtrend (sell rally)."""
    def __init__(self, swing_p=10, pullback_pct=0.003, ema_trend=50):
        self.swing_p=swing_p; self.pullback_pct=pullback_pct
        self.ema_trend=ema_trend; self._min_bars=swing_p*4+ema_trend+5
        self.swing_window=swing_p

    def _pivots(self, h, l, p):
        n=len(h); highs=[]; lows=[]
        for i in range(p, n-p):
            if h[i]==h[i-p:i+p+1].max(): highs.append((i,h[i]))
            if l[i]==l[i-p:i+p+1].min(): lows.append((i,l[i]))
        return highs, lows

    def generate_signals_batch(self, data):
        h=data["high"].values; l=data["low"].values; c=data["close"].values; n=len(c)
        et=_ema(c,self.ema_trend); sigs=["hold"]*n
        pivots_h, pivots_l = self._pivots(h, l, self.swing_p)
        ph_idx={idx:v for idx,v in pivots_h}; pl_idx={idx:v for idx,v in pivots_l}
        for i in range(self.swing_p*4, n):
            if np.isnan(et[i]): continue
            recent_h = [(idx,v) for idx,v in pivots_h if idx<i][-3:]
            recent_l = [(idx,v) for idx,v in pivots_l if idx<i][-3:]
            if len(recent_h)<2 or len(recent_l)<2: continue
            hh = recent_h[-1][1]>recent_h[-2][1]  # higher high
            hl = recent_l[-1][1]>recent_l[-2][1]  # higher low
            lh = recent_h[-1][1]<recent_h[-2][1]  # lower high
            ll = recent_l[-1][1]<recent_l[-2][1]  # lower low
            near_low  = abs(c[i]-recent_l[-1][1])/(recent_l[-1][1]+1e-10) < self.pullback_pct
            near_high = abs(c[i]-recent_h[-1][1])/(recent_h[-1][1]+1e-10) < self.pullback_pct
            if hh and hl and near_low  and c[i]>et[i]: sigs[i]="buy"
            elif lh and ll and near_high and c[i]<et[i]: sigs[i]="sell"
        return sigs

    def __repr__(self): return f"MarketStructure(sw={self.swing_p},et={self.ema_trend})"


# ════════════════════════════════════════════════════════════════════════════
# 5. MACD + RSI Confluence
# ════════════════════════════════════════════════════════════════════════════
class MACDRSIStrategy(_Base):
    """MACD histogram cruza cero + RSI en zona favorable."""
    def __init__(self, fast=12, slow=26, sig=9, rsi_p=14,
                 rsi_bull=50, rsi_bear=50, ema_trend=50):
        self.fast=fast; self.slow=slow; self.sig=sig; self.rsi_p=rsi_p
        self.rsi_bull=rsi_bull; self.rsi_bear=rsi_bear; self.ema_trend=ema_trend
        self._min_bars=slow+sig+rsi_p+10; self.swing_window=slow

    def generate_signals_batch(self, data):
        c=data["close"].values; n=len(c)
        _, _, hist = _macd(c, self.fast, self.slow, self.sig)
        rsi = _rsi(c, self.rsi_p); et = _ema(c, self.ema_trend)
        sigs=["hold"]*n
        for i in range(1,n):
            if any(np.isnan([hist[i],hist[i-1],rsi[i],et[i]])): continue
            cross_up   = hist[i-1]<0 and hist[i]>=0
            cross_down = hist[i-1]>0 and hist[i]<=0
            if cross_up   and rsi[i]>self.rsi_bull and c[i]>et[i]: sigs[i]="buy"
            elif cross_down and rsi[i]<self.rsi_bear and c[i]<et[i]: sigs[i]="sell"
        return sigs

    def __repr__(self): return f"MACDRSI(fast={self.fast},slow={self.slow},et={self.ema_trend})"


# ════════════════════════════════════════════════════════════════════════════
# 6. Linear Regression Channel
# ════════════════════════════════════════════════════════════════════════════
class LinearRegStrategy(_Base):
    """Precio fuera del canal de regresión lineal → ruptura de tendencia."""
    def __init__(self, period=50, std_mult=2.0, ema_trend=100):
        self.period=period; self.std_mult=std_mult; self.ema_trend=ema_trend
        self._min_bars=period+ema_trend+5; self.swing_window=period//5

    def generate_signals_batch(self, data):
        c=data["close"].values; n=len(c); et=_ema(c,self.ema_trend)
        sigs=["hold"]*n
        p=self.period
        for i in range(p-1, n):
            if np.isnan(et[i]): continue
            y=c[i-p+1:i+1]; x=np.arange(p)
            slope,intercept=np.polyfit(x,y,1)
            fitted=slope*x+intercept; resid=y-fitted
            std_r=resid.std()
            upper=fitted[-1]+self.std_mult*std_r
            lower=fitted[-1]-self.std_mult*std_r
            if c[i]>upper and c[i]>et[i]:    sigs[i]="buy"
            elif c[i]<lower and c[i]<et[i]:  sigs[i]="sell"
        return sigs

    def __repr__(self): return f"LinearReg(p={self.period},std={self.std_mult})"


# ════════════════════════════════════════════════════════════════════════════
# 7. Bollinger Band + Momentum
# ════════════════════════════════════════════════════════════════════════════
class BollingerMomStrategy(_Base):
    """Squeeze de BB (banda estrecha) → expansión con momentum positivo."""
    def __init__(self, bb_p=20, bb_std=2.0, kc_mult=1.5, mom_p=12, ema_trend=50):
        self.bb_p=bb_p; self.bb_std=bb_std; self.kc_mult=kc_mult
        self.mom_p=mom_p; self.ema_trend=ema_trend
        self._min_bars=bb_p+mom_p+ema_trend+10; self.swing_window=bb_p

    def generate_signals_batch(self, data):
        h=data["high"].values; l=data["low"].values; c=data["close"].values; n=len(c)
        mid=_sma(c,self.bb_p); std=_std(c,self.bb_p)
        bb_up=mid+self.bb_std*std; bb_dn=mid-self.bb_std*std
        atr14=_atr(h,l,c,14)
        kc_up=mid+self.kc_mult*atr14; kc_dn=mid-self.kc_mult*atr14
        # squeeze: BB dentro de KC
        squeeze=np.zeros(n)
        for i in range(n):
            if not any(np.isnan([bb_up[i],bb_dn[i],kc_up[i],kc_dn[i]])):
                squeeze[i]=1 if bb_up[i]<kc_up[i] and bb_dn[i]>kc_dn[i] else 0
        mom = c - _sma(c, self.mom_p)
        et  = _ema(c, self.ema_trend)
        sigs=["hold"]*n
        for i in range(1,n):
            if np.isnan(et[i]) or np.isnan(mom[i]): continue
            was_squeezed = squeeze[i-1]==1
            released     = squeeze[i]==0
            if was_squeezed and released:
                if mom[i]>0 and c[i]>et[i]:   sigs[i]="buy"
                elif mom[i]<0 and c[i]<et[i]: sigs[i]="sell"
        return sigs

    def __repr__(self): return f"BBMom(bb={self.bb_p},kc={self.kc_mult},et={self.ema_trend})"


# ════════════════════════════════════════════════════════════════════════════
# 8. Order Block (SMC-inspired)
# ════════════════════════════════════════════════════════════════════════════
class OrderBlockStrategy(_Base):
    """Última gran vela bajista antes de un impulso alcista = Order Block."""
    def __init__(self, ob_lookback=20, body_mult=1.5, ema_trend=50, tolerance=0.003):
        self.ob_lookback=ob_lookback; self.body_mult=body_mult
        self.ema_trend=ema_trend; self.tolerance=tolerance
        self._min_bars=ob_lookback+ema_trend+5; self.swing_window=ob_lookback//3

    def generate_signals_batch(self, data):
        o=data["open"].values; h=data["high"].values
        l=data["low"].values;  c=data["close"].values; n=len(c)
        body=np.abs(c-o); avg_body=_sma(body,20)
        et=_ema(c,self.ema_trend); sigs=["hold"]*n
        lb=self.ob_lookback
        for i in range(lb+1, n):
            if np.isnan(et[i]) or np.isnan(avg_body[i]): continue
            # buscar order block bearish (vela bajista grande) en lookback
            for j in range(i-1, max(0,i-lb), -1):
                is_bear_ob = (c[j]<o[j] and
                              body[j]>self.body_mult*avg_body[j] and
                              not np.isnan(avg_body[j]))
                is_bull_ob = (c[j]>o[j] and
                              body[j]>self.body_mult*avg_body[j] and
                              not np.isnan(avg_body[j]))
                ob_high = max(o[j],c[j]); ob_low = min(o[j],c[j])
                near = abs(c[i]-ob_high)/(ob_high+1e-10)<self.tolerance or \
                       abs(c[i]-ob_low)/(ob_low+1e-10)<self.tolerance
                if is_bear_ob and near and c[i]>et[i]: sigs[i]="buy";  break
                if is_bull_ob and near and c[i]<et[i]: sigs[i]="sell"; break
        return sigs

    def __repr__(self): return f"OrderBlock(lb={self.ob_lookback},bm={self.body_mult})"


# ════════════════════════════════════════════════════════════════════════════
# 9. Breakout + Volume
# ════════════════════════════════════════════════════════════════════════════
class BreakoutVolStrategy(_Base):
    """Cierre sobre/bajo el N-bar high/low con volumen 2× la media."""
    def __init__(self, period=20, vol_mult=1.8, ema_trend=50):
        self.period=period; self.vol_mult=vol_mult; self.ema_trend=ema_trend
        self._min_bars=period+ema_trend+5; self.swing_window=period//3

    def generate_signals_batch(self, data):
        h=data["high"].values; l=data["low"].values; c=data["close"].values
        v=data["volume"].values if "volume" in data.columns else np.ones(len(c))
        n=len(c); et=_ema(c,self.ema_trend); avg_vol=_sma(v,self.period)
        sigs=["hold"]*n
        for i in range(self.period, n):
            if np.isnan(et[i]) or np.isnan(avg_vol[i]): continue
            prev_high = h[i-self.period:i].max()
            prev_low  = l[i-self.period:i].min()
            vol_surge = v[i] > self.vol_mult*avg_vol[i]
            if c[i]>prev_high and vol_surge and c[i]>et[i]:   sigs[i]="buy"
            elif c[i]<prev_low and vol_surge and c[i]<et[i]:  sigs[i]="sell"
        return sigs

    def __repr__(self): return f"BreakoutVol(p={self.period},vm={self.vol_mult})"


# ════════════════════════════════════════════════════════════════════════════
# 10. Adaptive MTF (volatility-adjusted EMA periods)
# ════════════════════════════════════════════════════════════════════════════
class AdaptiveMTFStrategy(_Base):
    """MTF-SMC con EMA de tendencia adaptada según volatilidad ATR."""
    def __init__(self, swing_window=5, base_ema=50, atr_period=14,
                 vol_scale=2.0):
        self.swing_window=swing_window; self.base_ema=base_ema
        self.atr_period=atr_period; self.vol_scale=vol_scale
        self._min_bars=base_ema*2+atr_period+10; self.require_fvg=False
        self.use_choch_filter=False

    def generate_signals_batch(self, data):
        from backtests.backtester_fast import _precompute_signals
        c=data["close"].values; h=data["high"].values; l=data["low"].values
        n=len(c)
        # SMC base signals
        smc_sigs = _precompute_signals(data, swing_window=self.swing_window,
                                       require_fvg=False, use_choch_filter=False)
        # Adaptive EMA: more volatile → longer period (smoother trend)
        atr14 = _atr(h,l,c,self.atr_period)
        atr_norm = atr14 / (c+1e-10)  # ATR as % of price
        med_atr = np.nanmedian(atr_norm)
        # Resample to 4H for trend
        try:
            df4 = data.resample("4h").agg({"open":"first","high":"max",
                                           "low":"min","close":"last","volume":"sum"}).dropna()
            # adaptive period per 4H bar
            atr4 = _atr(df4["high"].values, df4["low"].values,
                        df4["close"].values, self.atr_period)
            atr4_norm = atr4 / (df4["close"].values+1e-10)
            med4 = np.nanmedian(atr4_norm)
            # scale: high vol → longer period
            scale = max(0.5, min(3.0, np.nanmean(atr4_norm[-20:]) / (med4+1e-10)))
            ema_p = max(10, int(self.base_ema * scale))
            et4 = _ema(df4["close"].values, ema_p)
            trend_s = pd.Series(np.where(df4["close"].values>et4,1,-1),
                                index=df4.index)
            trend_1h = trend_s.reindex(data.index, method="ffill").fillna(0).values
        except Exception:
            et = _ema(c, self.base_ema)
            trend_1h = np.where(c>et, 1, -1)

        sigs=["hold"]*n
        for i in range(n):
            s=smc_sigs[i]; t=trend_1h[i]
            if s=="buy"  and t>=0: sigs[i]="buy"
            elif s=="sell" and t<=0: sigs[i]="sell"
        return sigs

    def __repr__(self): return f"AdaptiveMTF(sw={self.swing_window},base_ema={self.base_ema})"
