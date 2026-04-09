"""
statistical_strategies.py — 13 estrategias cuantitativas/estadísticas

  1.  ZScoreStrategy          — Z-score precio vs media móvil
  2.  BollingerReversionStrategy — rebote desde extremos BB hacia media
  3.  OBVTrendStrategy         — On-Balance Volume tendencia
  4.  MFIStrategy              — Money Flow Index extremos
  5.  ForceIndexStrategy       — Elder Force Index cruce de cero
  6.  AccDistStrategy          — Accumulation/Distribution línea de tendencia
  7.  ROCStrategy              — Rate of Change con filtro EMA
  8.  TSIStrategy              — True Strength Index
  9.  CMOStrategy              — Chande Momentum Oscillator
 10.  UltimateOscStrategy      — Ultimate Oscillator (3 periodos)
 11.  InsideBarBreakoutStrategy — Inside bar + breakout direccional
 12.  FVGStrategy              — Fair Value Gap (SMC) con retest
 13.  VolumeOscStrategy        — Oscilador de volumen fast/slow
"""

import numpy as np
import pandas as pd


def _ema(arr, p):
    a = 2/(p+1); out = np.full(len(arr), np.nan)
    if len(arr) < p: return out
    out[p-1] = arr[:p].mean()
    for i in range(p, len(arr)): out[i] = a*arr[i]+(1-a)*out[i-1]
    return out

def _sma(arr, p):
    out = np.full(len(arr), np.nan)
    for i in range(p-1, len(arr)): out[i] = arr[i-p+1:i+1].mean()
    return out

def _atr(h, l, c, p):
    n=len(c); tr=np.zeros(n); tr[0]=h[0]-l[0]
    for i in range(1,n): tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    out=np.full(n,np.nan)
    if n<p: return out
    out[p-1]=tr[:p].mean()
    for i in range(p,n): out[i]=(out[i-1]*(p-1)+tr[i])/p
    return out


# 1. Z-Score
class ZScoreStrategy:
    def __init__(self, period=50, entry_z=2.0, ema_trend=100):
        self.period=period; self.ez=entry_z; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        c=df["close"].values; n=len(c)
        ema=_ema(c,self.ema_trend); sigs=["hold"]*n
        for i in range(self.period,n):
            if np.isnan(ema[i]): continue
            sl=c[i-self.period:i+1]; m=sl.mean(); s=sl.std(ddof=1)
            if s<1e-10: continue
            z=(c[i]-m)/s; z_p=(c[i-1]-sl[:-1].mean())/(sl[:-1].std(ddof=1)+1e-10)
            if z<-self.ez and z>z_p and c[i]>ema[i]*0.97: sigs[i]="buy"
            elif z>self.ez and z<z_p and c[i]<ema[i]*1.03: sigs[i]="sell"
        return sigs

# 2. Bollinger Reversion
class BollingerReversionStrategy:
    def __init__(self, period=20, mult=2.0, ema_trend=50):
        self.period=period; self.mult=mult; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        c=df["close"].values; n=len(c)
        mid=_sma(c,self.period); ema=_ema(c,self.ema_trend)
        std=np.array([c[max(0,i-self.period+1):i+1].std(ddof=1)
                      if i>=self.period-1 else np.nan for i in range(n)])
        lo=mid-self.mult*std; hi=mid+self.mult*std; sigs=["hold"]*n
        for i in range(1,n):
            if np.isnan(lo[i]) or np.isnan(ema[i]): continue
            if c[i]<lo[i] and c[i-1]>=lo[i-1] and c[i]>ema[i]*0.95: sigs[i]="buy"
            elif c[i]>hi[i] and c[i-1]<=hi[i-1] and c[i]<ema[i]*1.05: sigs[i]="sell"
        return sigs

# 3. OBV Trend
class OBVTrendStrategy:
    def __init__(self, obv_fast=10, obv_slow=30, ema_trend=50):
        self.obv_fast=obv_fast; self.obv_slow=obv_slow; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        c=df["close"].values; v=df["volume"].values; n=len(c)
        obv=np.zeros(n)
        for i in range(1,n):
            if c[i]>c[i-1]: obv[i]=obv[i-1]+v[i]
            elif c[i]<c[i-1]: obv[i]=obv[i-1]-v[i]
            else: obv[i]=obv[i-1]
        of=_ema(obv,self.obv_fast); os2=_ema(obv,self.obv_slow)
        ema=_ema(c,self.ema_trend); sigs=["hold"]*n
        for i in range(1,n):
            if np.isnan(of[i]) or np.isnan(os2[i]) or np.isnan(ema[i]): continue
            if of[i]>os2[i] and of[i-1]<=os2[i-1] and c[i]>ema[i]: sigs[i]="buy"
            elif of[i]<os2[i] and of[i-1]>=os2[i-1] and c[i]<ema[i]: sigs[i]="sell"
        return sigs

# 4. MFI Strategy
class MFIStrategy:
    def __init__(self, period=14, os=25, ob=75, ema_trend=50):
        self.period=period; self.os=os; self.ob=ob; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        h=df["high"].values; l=df["low"].values
        c=df["close"].values; v=df["volume"].values; n=len(c)
        tp=(h+l+c)/3; rmf=tp*v
        mfi=np.full(n,np.nan); ema=_ema(c,self.ema_trend)
        for i in range(self.period,n):
            pos=sum(rmf[j] for j in range(i-self.period+1,i+1) if tp[j]>tp[j-1])
            neg=sum(rmf[j] for j in range(i-self.period+1,i+1) if tp[j]<tp[j-1])
            mfi[i]=100-100/(1+pos/(neg+1e-10))
        sigs=["hold"]*n
        for i in range(1,n):
            if np.isnan(mfi[i]) or np.isnan(ema[i]): continue
            if mfi[i]>self.os and mfi[i-1]<=self.os and c[i]>ema[i]: sigs[i]="buy"
            elif mfi[i]<self.ob and mfi[i-1]>=self.ob and c[i]<ema[i]: sigs[i]="sell"
        return sigs

# 5. Force Index
class ForceIndexStrategy:
    def __init__(self, period=13, ema_trend=50):
        self.period=period; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        c=df["close"].values; v=df["volume"].values; n=len(c)
        fi=np.zeros(n)
        for i in range(1,n): fi[i]=(c[i]-c[i-1])*v[i]
        fi_ema=_ema(fi,self.period); ema=_ema(c,self.ema_trend); sigs=["hold"]*n
        for i in range(1,n):
            if np.isnan(fi_ema[i]) or np.isnan(ema[i]): continue
            if fi_ema[i]>0 and fi_ema[i-1]<=0 and c[i]>ema[i]: sigs[i]="buy"
            elif fi_ema[i]<0 and fi_ema[i-1]>=0 and c[i]<ema[i]: sigs[i]="sell"
        return sigs

# 6. Accumulation/Distribution
class AccDistStrategy:
    def __init__(self, fast=3, slow=10, ema_trend=50):
        self.fast=fast; self.slow=slow; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        h=df["high"].values; l=df["low"].values
        c=df["close"].values; v=df["volume"].values; n=len(c)
        ad=np.zeros(n)
        for i in range(n):
            rng=h[i]-l[i]
            if rng>0: ad[i]=(ad[i-1] if i>0 else 0)+((c[i]-l[i])-(h[i]-c[i]))/rng*v[i]
        af=_ema(ad,self.fast); as2=_ema(ad,self.slow)
        ema=_ema(c,self.ema_trend); sigs=["hold"]*n
        for i in range(1,n):
            if np.isnan(af[i]) or np.isnan(as2[i]) or np.isnan(ema[i]): continue
            if af[i]>as2[i] and af[i-1]<=as2[i-1] and c[i]>ema[i]: sigs[i]="buy"
            elif af[i]<as2[i] and af[i-1]>=as2[i-1] and c[i]<ema[i]: sigs[i]="sell"
        return sigs

# 7. ROC Strategy
class ROCStrategy:
    def __init__(self, period=10, smooth=3, ema_trend=50):
        self.period=period; self.smooth=smooth; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        c=df["close"].values; n=len(c)
        roc=np.full(n,np.nan)
        for i in range(self.period,n):
            if c[i-self.period]>0: roc[i]=(c[i]-c[i-self.period])/c[i-self.period]*100
        roc_s=_ema(np.nan_to_num(roc),self.smooth); ema=_ema(c,self.ema_trend)
        sigs=["hold"]*n
        for i in range(1,n):
            if np.isnan(roc_s[i]) or np.isnan(ema[i]): continue
            if roc_s[i]>0 and roc_s[i-1]<=0 and c[i]>ema[i]: sigs[i]="buy"
            elif roc_s[i]<0 and roc_s[i-1]>=0 and c[i]<ema[i]: sigs[i]="sell"
        return sigs

# 8. TSI (True Strength Index)
class TSIStrategy:
    def __init__(self, r=25, s=13, ema_trend=50):
        self.r=r; self.s=s; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        c=df["close"].values; n=len(c)
        pc=np.diff(c, prepend=c[0])
        ds1=_ema(pc,self.r); ds2=_ema(ds1,self.s)
        abs1=_ema(np.abs(pc),self.r); abs2=_ema(abs1,self.s)
        tsi=np.where(abs2>0, 100*ds2/abs2, 0.0)
        sig_line=_ema(tsi,7); ema=_ema(c,self.ema_trend); sigs=["hold"]*n
        for i in range(1,n):
            if np.isnan(tsi[i]) or np.isnan(ema[i]): continue
            if tsi[i]>sig_line[i] and tsi[i-1]<=sig_line[i-1] and c[i]>ema[i]: sigs[i]="buy"
            elif tsi[i]<sig_line[i] and tsi[i-1]>=sig_line[i-1] and c[i]<ema[i]: sigs[i]="sell"
        return sigs

# 9. CMO (Chande Momentum Oscillator)
class CMOStrategy:
    def __init__(self, period=14, os=-50, ob=50, ema_trend=50):
        self.period=period; self.os=os; self.ob=ob; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        c=df["close"].values; n=len(c)
        d=np.diff(c,prepend=c[0])
        up=np.where(d>0,d,0.); dn=np.where(d<0,-d,0.)
        cmo=np.full(n,np.nan); ema=_ema(c,self.ema_trend)
        for i in range(self.period,n):
            su=up[i-self.period+1:i+1].sum(); sd=dn[i-self.period+1:i+1].sum()
            cmo[i]=100*(su-sd)/(su+sd+1e-10)
        sigs=["hold"]*n
        for i in range(1,n):
            if np.isnan(cmo[i]) or np.isnan(ema[i]): continue
            if cmo[i]>self.os and cmo[i-1]<=self.os and c[i]>ema[i]: sigs[i]="buy"
            elif cmo[i]<self.ob and cmo[i-1]>=self.ob and c[i]<ema[i]: sigs[i]="sell"
        return sigs

# 10. Ultimate Oscillator
class UltimateOscStrategy:
    def __init__(self, p1=7, p2=14, p3=28, os=30, ob=70, ema_trend=50):
        self.p1=p1; self.p2=p2; self.p3=p3
        self.os=os; self.ob=ob; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        h=df["high"].values; l=df["low"].values
        c=df["close"].values; n=len(c)
        bp=np.zeros(n); tr=np.zeros(n)
        for i in range(1,n):
            pc=c[i-1]; bp[i]=c[i]-min(l[i],pc)
            tr[i]=max(h[i],pc)-min(l[i],pc)
        uo=np.full(n,np.nan); ema=_ema(c,self.ema_trend)
        for i in range(self.p3,n):
            def avg(p): s=bp[i-p+1:i+1].sum(); t=tr[i-p+1:i+1].sum(); return s/(t+1e-10)
            uo[i]=100*(4*avg(self.p1)+2*avg(self.p2)+avg(self.p3))/7
        sigs=["hold"]*n
        for i in range(1,n):
            if np.isnan(uo[i]) or np.isnan(ema[i]): continue
            if uo[i]>self.os and uo[i-1]<=self.os and c[i]>ema[i]: sigs[i]="buy"
            elif uo[i]<self.ob and uo[i-1]>=self.ob and c[i]<ema[i]: sigs[i]="sell"
        return sigs

# 11. Inside Bar Breakout
class InsideBarBreakoutStrategy:
    def __init__(self, ema_trend=50, min_inside: int=1):
        self.ema_trend=ema_trend; self.min_inside=min_inside
    def generate_signals_batch(self, df):
        h=df["high"].values; l=df["low"].values; c=df["close"].values; n=len(c)
        ema=_ema(c,self.ema_trend); sigs=["hold"]*n
        inside_count=0
        for i in range(1,n):
            if np.isnan(ema[i]): continue
            if h[i]<=h[i-1] and l[i]>=l[i-1]:
                inside_count+=1
            else:
                if inside_count>=self.min_inside:
                    prev_h=h[i-inside_count-1]; prev_l=l[i-inside_count-1]
                    if c[i]>prev_h and c[i]>ema[i]: sigs[i]="buy"
                    elif c[i]<prev_l and c[i]<ema[i]: sigs[i]="sell"
                inside_count=0
        return sigs

# 12. Fair Value Gap (FVG)
class FVGStrategy:
    def __init__(self, ema_trend=50, gap_min_pct=0.1):
        self.ema_trend=ema_trend; self.gap_min=gap_min_pct/100
    def generate_signals_batch(self, df):
        h=df["high"].values; l=df["low"].values; c=df["close"].values; n=len(c)
        ema=_ema(c,self.ema_trend); sigs=["hold"]*n
        bull_fvgs=[]; bear_fvgs=[]
        for i in range(2,n):
            if np.isnan(ema[i]): continue
            # Bullish FVG: low[i] > high[i-2]
            if l[i]>h[i-2] and (l[i]-h[i-2])/h[i-2]>self.gap_min:
                bull_fvgs.append((h[i-2], l[i]))
            # Bearish FVG: high[i] < low[i-2]
            if h[i]<l[i-2] and (l[i-2]-h[i])/l[i-2]>self.gap_min:
                bear_fvgs.append((h[i], l[i-2]))
            # Check retest of bullish FVG (buy at retest)
            for fvg_lo, fvg_hi in bull_fvgs[-5:]:
                if fvg_lo<=c[i]<=fvg_hi and c[i]>ema[i]:
                    sigs[i]="buy"; break
            # Check retest of bearish FVG (sell at retest)
            for fvg_lo, fvg_hi in bear_fvgs[-5:]:
                if fvg_lo<=c[i]<=fvg_hi and c[i]<ema[i]:
                    sigs[i]="sell"; break
        return sigs

# 13. Volume Oscillator
class VolumeOscStrategy:
    def __init__(self, fast=5, slow=20, ema_trend=50):
        self.fast=fast; self.slow=slow; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        c=df["close"].values; v=df["volume"].values; n=len(c)
        vf=_ema(v,self.fast); vs=_ema(v,self.slow)
        vo=np.where(vs>0,(vf-vs)/vs*100,0.); ema=_ema(c,self.ema_trend)
        sigs=["hold"]*n
        for i in range(1,n):
            if np.isnan(vo[i]) or np.isnan(ema[i]): continue
            if vo[i]>0 and vo[i-1]<=0 and c[i]>ema[i]: sigs[i]="buy"
            elif vo[i]<0 and vo[i-1]>=0 and c[i]<ema[i]: sigs[i]="sell"
        return sigs
