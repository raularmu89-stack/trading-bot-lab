"""
hybrid_strategies.py — 12 estrategias híbridas avanzadas

Combinaciones multi-indicador de alta precisión:
  1.  MACDStochStrategy      — MACD + Stochastic confluencia
  2.  RSIBollingerStrategy    — RSI extremos dentro de BB
  3.  ADXMACDStrategy         — ADX tendencia + MACD dirección
  4.  TripleScreenStrategy    — Elder Triple Screen simplificado
  5.  ScalpingMTFStrategy     — Señal rápida + filtro lento
  6.  MomentumBreakoutStrategy— Momentum + ruptura de rango N velas
  7.  TrendReversalStrategy   — Detección de reversión con confirmación
  8.  VolatilityBreakoutStrategy — ATR expansion + dirección
  9.  SmartMoneyFlowStrategy  — Flujo de dinero inteligente (volumen+precio)
 10.  AdaptiveRSIStrategy     — RSI con umbral adaptativo por volatilidad
 11.  PullbackTrendStrategy   — Pullback en EMA en tendencia fuerte
 12.  DivergenceConfluenceStrategy — Confluencia de divergencias RSI+MACD
"""

import numpy as np
import pandas as pd


def _ema(arr, p):
    a=2/(p+1); out=np.full(len(arr),np.nan)
    if len(arr)<p: return out
    out[p-1]=arr[:p].mean()
    for i in range(p,len(arr)): out[i]=a*arr[i]+(1-a)*out[i-1]
    return out

def _sma(arr,p):
    out=np.full(len(arr),np.nan)
    for i in range(p-1,len(arr)): out[i]=arr[i-p+1:i+1].mean()
    return out

def _atr(h,l,c,p):
    n=len(c); tr=np.zeros(n); tr[0]=h[0]-l[0]
    for i in range(1,n): tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    out=np.full(n,np.nan)
    if n<p: return out
    out[p-1]=tr[:p].mean()
    for i in range(p,n): out[i]=(out[i-1]*(p-1)+tr[i])/p
    return out

def _rsi(c, p=14):
    n=len(c); out=np.full(n,np.nan)
    if n<p+1: return out
    d=np.diff(c); g=np.where(d>0,d,0.); lo=np.where(d<0,-d,0.)
    ag,al=g[:p].mean(),lo[:p].mean()
    for i in range(p,n-1):
        ag=(ag*(p-1)+g[i])/p; al=(al*(p-1)+lo[i])/p
        out[i+1]=100-100/(1+ag/(al+1e-10))
    return out

def _stoch(h,l,c,k=14,d=3):
    n=len(c); pk=np.full(n,np.nan)
    for i in range(k-1,n):
        hh=h[i-k+1:i+1].max(); ll=l[i-k+1:i+1].min()
        pk[i]=(c[i]-ll)/(hh-ll+1e-10)*100
    pd2=_sma(pk,d); return pk,pd2


# 1. MACD + Stochastic
class MACDStochStrategy:
    def __init__(self, macd_fast=12, macd_slow=26, macd_sig=9,
                 stoch_k=14, stoch_d=3, ema_trend=50):
        self.mf=macd_fast; self.ms=macd_slow; self.msig=macd_sig
        self.sk=stoch_k; self.sd=stoch_d; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        c=df["close"].values; h=df["high"].values; l=df["low"].values; n=len(c)
        macd=_ema(c,self.mf)-_ema(c,self.ms)
        sig=_ema(np.nan_to_num(macd),self.msig); hist=macd-sig
        sk,sd=_stoch(h,l,c,self.sk,self.sd); ema=_ema(c,self.ema_trend)
        sigs=["hold"]*n
        for i in range(1,n):
            if any(np.isnan(x) for x in [hist[i],sk[i],ema[i]]): continue
            macd_bull=hist[i]>0 and hist[i]>hist[i-1]
            macd_bear=hist[i]<0 and hist[i]<hist[i-1]
            stoch_bull=sk[i]>sd[i] and sk[i-1]<=sd[i-1] and sk[i]<80
            stoch_bear=sk[i]<sd[i] and sk[i-1]>=sd[i-1] and sk[i]>20
            if macd_bull and stoch_bull and c[i]>ema[i]: sigs[i]="buy"
            elif macd_bear and stoch_bear and c[i]<ema[i]: sigs[i]="sell"
        return sigs

# 2. RSI + Bollinger
class RSIBollingerStrategy:
    def __init__(self, rsi_period=14, bb_period=20, bb_mult=2.0,
                 rsi_os=35, rsi_ob=65, ema_trend=50):
        self.rp=rsi_period; self.bp=bb_period; self.bm=bb_mult
        self.os=rsi_os; self.ob=rsi_ob; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        c=df["close"].values; n=len(c)
        rsi=_rsi(c,self.rp); mid=_sma(c,self.bp)
        std=np.array([c[max(0,i-self.bp+1):i+1].std(ddof=1)
                      if i>=self.bp-1 else np.nan for i in range(n)])
        lo=mid-self.bm*std; hi=mid+self.bm*std; ema=_ema(c,self.ema_trend)
        sigs=["hold"]*n
        for i in range(1,n):
            if any(np.isnan(x) for x in [rsi[i],lo[i],ema[i]]): continue
            if c[i]<lo[i] and rsi[i]<self.os and rsi[i]>rsi[i-1] and c[i]>ema[i]*0.95:
                sigs[i]="buy"
            elif c[i]>hi[i] and rsi[i]>self.ob and rsi[i]<rsi[i-1] and c[i]<ema[i]*1.05:
                sigs[i]="sell"
        return sigs

# 3. ADX + MACD
class ADXMACDStrategy:
    def __init__(self, adx_min=20, macd_fast=12, macd_slow=26, macd_sig=9, ema_trend=50):
        self.adx_min=adx_min; self.mf=macd_fast; self.ms=macd_slow
        self.msig=macd_sig; self.ema_trend=ema_trend
    def _adx(self,h,l,c,p=14):
        n=len(c); tr=np.zeros(n); tr[0]=h[0]-l[0]
        for i in range(1,n): tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
        pdm=np.where((h[1:]-h[:-1])>(l[:-1]-l[1:]),np.maximum(h[1:]-h[:-1],0),0.)
        ndm=np.where((l[:-1]-l[1:])>(h[1:]-h[:-1]),np.maximum(l[:-1]-l[1:],0),0.)
        def ws(a):
            out=np.zeros(len(a)+1); out[p]=a[:p].sum()
            for i in range(p,len(a)): out[i+1]=out[i]-out[i]/p+a[i]
            return out[1:]
        at=ws(tr[1:]); pm=ws(pdm); nm=ws(ndm)
        pdi=np.where(at>0,100*pm/at,0.); ndi=np.where(at>0,100*nm/at,0.)
        dx=np.where(pdi+ndi>0,100*np.abs(pdi-ndi)/(pdi+ndi),0.)
        adx=np.zeros(len(dx))
        if len(dx)>=p: adx[p-1]=dx[:p].mean()
        for i in range(p,len(dx)): adx[i]=(adx[i-1]*(p-1)+dx[i])/p
        pad=np.zeros(1)
        return np.concatenate([pad,adx])[:n], np.concatenate([pad,pdi])[:n], np.concatenate([pad,ndi])[:n]
    def generate_signals_batch(self, df):
        c=df["close"].values; h=df["high"].values; l=df["low"].values; n=len(c)
        adx,pdi,ndi=self._adx(h,l,c)
        macd=_ema(c,self.mf)-_ema(c,self.ms)
        sig=_ema(np.nan_to_num(macd),self.msig); hist=macd-sig
        ema=_ema(c,self.ema_trend); sigs=["hold"]*n
        for i in range(1,n):
            if any(np.isnan(x) for x in [adx[i],hist[i],ema[i]]): continue
            trending=adx[i]>self.adx_min
            if trending and hist[i]>0 and hist[i-1]<=0 and pdi[i]>ndi[i]: sigs[i]="buy"
            elif trending and hist[i]<0 and hist[i-1]>=0 and ndi[i]>pdi[i]: sigs[i]="sell"
        return sigs

# 4. Triple Screen (Elder)
class TripleScreenStrategy:
    def __init__(self, weekly_ema=52, daily_stoch_k=14, intra_macd_fast=12,
                 intra_macd_slow=26):
        self.we=weekly_ema; self.sk=daily_stoch_k
        self.mf=intra_macd_fast; self.ms=intra_macd_slow
    def generate_signals_batch(self, df):
        c=df["close"].values; h=df["high"].values; l=df["low"].values; n=len(c)
        # Screen 1: weekly trend (use longer EMA as proxy)
        trend_ema=_ema(c,self.we)
        # Screen 2: overbought/oversold oscillator
        sk,sd=_stoch(h,l,c,self.sk,3)
        # Screen 3: intraday entry (MACD histogram)
        macd=_ema(c,self.mf)-_ema(c,self.ms)
        sig=_ema(np.nan_to_num(macd),9); hist=macd-sig
        sigs=["hold"]*n
        for i in range(1,n):
            if any(np.isnan(x) for x in [trend_ema[i],sk[i],hist[i]]): continue
            bull_trend=c[i]>trend_ema[i]
            bear_trend=c[i]<trend_ema[i]
            if bull_trend and sk[i]<30 and hist[i]>0 and hist[i-1]<=0: sigs[i]="buy"
            elif bear_trend and sk[i]>70 and hist[i]<0 and hist[i-1]>=0: sigs[i]="sell"
        return sigs

# 5. Scalping MTF
class ScalpingMTFStrategy:
    def __init__(self, fast_ema=5, mid_ema=13, slow_ema=50, rsi_p=7):
        self.fe=fast_ema; self.me=mid_ema; self.se=slow_ema; self.rp=rsi_p
    def generate_signals_batch(self, df):
        c=df["close"].values; n=len(c)
        ef=_ema(c,self.fe); em=_ema(c,self.me); es=_ema(c,self.se)
        rsi=_rsi(c,self.rp); sigs=["hold"]*n
        for i in range(1,n):
            if any(np.isnan(x) for x in [ef[i],em[i],es[i],rsi[i]]): continue
            bull=ef[i]>em[i]>es[i]
            bear=ef[i]<em[i]<es[i]
            bull_p=ef[i-1]>em[i-1]>es[i-1]
            bear_p=ef[i-1]<em[i-1]<es[i-1]
            if bull and not bull_p and rsi[i]>50: sigs[i]="buy"
            elif bear and not bear_p and rsi[i]<50: sigs[i]="sell"
        return sigs

# 6. Momentum Breakout
class MomentumBreakoutStrategy:
    def __init__(self, period=20, mom_period=10, ema_trend=50):
        self.period=period; self.mp=mom_period; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        h=df["high"].values; l=df["low"].values; c=df["close"].values; n=len(c)
        ema=_ema(c,self.ema_trend); sigs=["hold"]*n
        for i in range(max(self.period,self.mp),n):
            if np.isnan(ema[i]): continue
            hh=h[i-self.period:i].max(); ll=l[i-self.period:i].min()
            mom=(c[i]-c[i-self.mp])/c[i-self.mp]*100 if c[i-self.mp]>0 else 0
            if c[i]>hh and mom>0 and c[i]>ema[i]: sigs[i]="buy"
            elif c[i]<ll and mom<0 and c[i]<ema[i]: sigs[i]="sell"
        return sigs

# 7. Trend Reversal
class TrendReversalStrategy:
    def __init__(self, ema_fast=20, ema_slow=50, rsi_p=14, rsi_os=35, rsi_ob=65):
        self.ef=ema_fast; self.es=ema_slow; self.rp=rsi_p
        self.os=rsi_os; self.ob=rsi_ob
    def generate_signals_batch(self, df):
        c=df["close"].values; n=len(c)
        ef=_ema(c,self.ef); es=_ema(c,self.es); rsi=_rsi(c,self.rp)
        sigs=["hold"]*n
        for i in range(2,n):
            if any(np.isnan(x) for x in [ef[i],es[i],rsi[i]]): continue
            # Reversión alcista: precio bajo + RSI rebotando + EMA se cruzan
            was_bear=ef[i-2]<es[i-2]
            now_bull=ef[i]>es[i]
            if was_bear and now_bull and rsi[i]<60 and rsi[i]>rsi[i-1]: sigs[i]="buy"
            was_bull=ef[i-2]>es[i-2]
            now_bear=ef[i]<es[i]
            if was_bull and now_bear and rsi[i]>40 and rsi[i]<rsi[i-1]: sigs[i]="sell"
        return sigs

# 8. Volatility Breakout
class VolatilityBreakoutStrategy:
    def __init__(self, atr_period=14, atr_mult=1.5, lookback=20, ema_trend=50):
        self.ap=atr_period; self.am=atr_mult; self.lb=lookback; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        h=df["high"].values; l=df["low"].values; c=df["close"].values; n=len(c)
        atr=_atr(h,l,c,self.ap); ema=_ema(c,self.ema_trend); sigs=["hold"]*n
        for i in range(self.lb,n):
            if np.isnan(atr[i]) or np.isnan(ema[i]): continue
            prev_atr=atr[i-self.lb:i].mean()
            expanding=atr[i]>prev_atr*self.am
            if expanding:
                if c[i]>c[i-1] and c[i]>ema[i]: sigs[i]="buy"
                elif c[i]<c[i-1] and c[i]<ema[i]: sigs[i]="sell"
        return sigs

# 9. Smart Money Flow
class SmartMoneyFlowStrategy:
    def __init__(self, period=20, flow_thresh=0.6, ema_trend=50):
        self.period=period; self.ft=flow_thresh; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        h=df["high"].values; l=df["low"].values
        c=df["close"].values; v=df["volume"].values; n=len(c)
        # Smart money: grandes velas con volumen alto que cierran en extremos
        body_pct=(c-l)/(h-l+1e-10)  # 1=cierre en máximo, 0=cierre en mínimo
        vol_ma=_sma(v,self.period); ema=_ema(c,self.ema_trend)
        sigs=["hold"]*n
        for i in range(self.period,n):
            if np.isnan(vol_ma[i]) or np.isnan(ema[i]): continue
            big_vol=v[i]>vol_ma[i]*1.5
            bull_candle=body_pct[i]>self.ft and c[i]>c[i-1]
            bear_candle=body_pct[i]<(1-self.ft) and c[i]<c[i-1]
            if big_vol and bull_candle and c[i]>ema[i]: sigs[i]="buy"
            elif big_vol and bear_candle and c[i]<ema[i]: sigs[i]="sell"
        return sigs

# 10. Adaptive RSI
class AdaptiveRSIStrategy:
    def __init__(self, rsi_period=14, atr_period=14, base_os=30, base_ob=70, ema_trend=50):
        self.rp=rsi_period; self.ap=atr_period
        self.os=base_os; self.ob=base_ob; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        h=df["high"].values; l=df["low"].values; c=df["close"].values; n=len(c)
        rsi=_rsi(c,self.rp); atr=_atr(h,l,c,self.ap)
        atr_pct=np.where(c>0,atr/c*100,0.); ema=_ema(c,self.ema_trend)
        sigs=["hold"]*n
        for i in range(1,n):
            if any(np.isnan(x) for x in [rsi[i],atr_pct[i],ema[i]]): continue
            # Ajustar umbrales: más volatilidad → umbrales más amplios
            adj=min(atr_pct[i]*2,15)
            os_adj=self.os-adj; ob_adj=self.ob+adj
            if rsi[i]>os_adj and rsi[i-1]<=os_adj and c[i]>ema[i]: sigs[i]="buy"
            elif rsi[i]<ob_adj and rsi[i-1]>=ob_adj and c[i]<ema[i]: sigs[i]="sell"
        return sigs

# 11. Pullback Trend
class PullbackTrendStrategy:
    def __init__(self, ema_fast=20, ema_slow=50, pullback_pct=0.3, rsi_p=14):
        self.ef=ema_fast; self.es=ema_slow; self.pb=pullback_pct; self.rp=rsi_p
    def generate_signals_batch(self, df):
        c=df["close"].values; n=len(c)
        ef=_ema(c,self.ef); es=_ema(c,self.es); rsi=_rsi(c,self.rp)
        sigs=["hold"]*n
        for i in range(1,n):
            if any(np.isnan(x) for x in [ef[i],es[i],rsi[i]]): continue
            # Pullback alcista: tendencia up + precio retrocede hacia EMA fast
            bull_trend=ef[i]>es[i] and ef[i]>ef[i-1]
            at_ema_fast=abs(c[i]-ef[i])/ef[i]<self.pb/100
            bear_trend=ef[i]<es[i] and ef[i]<ef[i-1]
            if bull_trend and at_ema_fast and rsi[i]>45 and rsi[i]>rsi[i-1]: sigs[i]="buy"
            elif bear_trend and at_ema_fast and rsi[i]<55 and rsi[i]<rsi[i-1]: sigs[i]="sell"
        return sigs

# 12. Divergence Confluence
class DivergenceConfluenceStrategy:
    def __init__(self, period=20, rsi_p=14, macd_fast=12, macd_slow=26, ema_trend=100):
        self.period=period; self.rp=rsi_p
        self.mf=macd_fast; self.ms=macd_slow; self.ema_trend=ema_trend
    def generate_signals_batch(self, df):
        c=df["close"].values; n=len(c)
        rsi=_rsi(c,self.rp)
        macd=_ema(c,self.mf)-_ema(c,self.ms)
        sig=_ema(np.nan_to_num(macd),9); hist=macd-sig
        ema=_ema(c,self.ema_trend); sigs=["hold"]*n
        lb=self.period
        for i in range(lb,n):
            if any(np.isnan(x) for x in [rsi[i],hist[i],ema[i]]): continue
            wc=c[i-lb:i+1]; wr=rsi[i-lb:i+1]; wh=hist[i-lb:i+1]
            if np.any(np.isnan(wr)) or np.any(np.isnan(wh)): continue
            # Divergencia alcista RSI
            rsi_div=wc[-1]==wc.min() and wc[-1]<wc[:-1].min() and wr[-1]>wr[np.argmin(wc[:-1])]
            # Divergencia alcista MACD
            macd_div=wc[-1]==wc.min() and wh[-1]>wh[np.argmin(wc[:-1])]
            if rsi_div and macd_div and c[i]>ema[i]*0.98: sigs[i]="buy"
            # Divergencia bajista
            rsi_div_b=wc[-1]==wc.max() and wc[-1]>wc[:-1].max() and wr[-1]<wr[np.argmax(wc[:-1])]
            macd_div_b=wc[-1]==wc.max() and wh[-1]<wh[np.argmax(wc[:-1])]
            if rsi_div_b and macd_div_b and c[i]<ema[i]*1.02: sigs[i]="sell"
        return sigs
