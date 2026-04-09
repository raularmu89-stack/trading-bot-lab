"""
competition_100.py

COMPETICIÓN DE 100 ESTRATEGIAS — el torneo más grande.

Incluye todas las familias de estrategias + variantes de parámetros.
Motor: Trail(3.0) + Partial(0.33) + Kelly full
Capital inicial para proyección: €100

Familias:
  A. MTF-SMC (6 variantes)
  B. Advanced (10 estrategias × 2 params)
  C. Statistical / Quant (13 estrategias)
  D. Hybrid multi-indicador (12 estrategias)
  E. Momentum oscillators (8 estrategias × 2 params)
  F. Trend momentum (5 estrategias × 2 params)
  G. Price action (6 estrategias)
  H. MA cross (4 estrategias × 2 params)

Total: ~100 configuraciones
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from backtests.kelly_backtest_v2  import run_kelly_v2, kelly_metrics_v2
from strategies.kelly_sizer       import KellySizer
from strategies.risk_manager      import RiskManager
from strategies.regime_filter     import RegimeFilteredStrategy

# Importar todas las familias
from strategies.mtf_smc           import MultiTFSMC
from strategies.smc_strategy      import SMCStrategy
from strategies.advanced_strategies import (
    StochRSIStrategy, IchimokuStrategy, FibRetracementStrategy,
    MarketStructureStrategy, MACDRSIStrategy, LinearRegStrategy,
    BollingerMomStrategy, OrderBlockStrategy, BreakoutVolStrategy,
    AdaptiveMTFStrategy,
)
from strategies.statistical_strategies import (
    ZScoreStrategy, BollingerReversionStrategy, OBVTrendStrategy,
    MFIStrategy, ForceIndexStrategy, AccDistStrategy, ROCStrategy,
    TSIStrategy, CMOStrategy, UltimateOscStrategy,
    InsideBarBreakoutStrategy, FVGStrategy, VolumeOscStrategy,
)
from strategies.hybrid_strategies import (
    MACDStochStrategy, RSIBollingerStrategy, ADXMACDStrategy,
    TripleScreenStrategy, ScalpingMTFStrategy, MomentumBreakoutStrategy,
    TrendReversalStrategy, VolatilityBreakoutStrategy,
    SmartMoneyFlowStrategy, AdaptiveRSIStrategy,
    PullbackTrendStrategy, DivergenceConfluenceStrategy,
)
from strategies.momentum_oscillator_strategies import (
    AwesomeOscillatorStrategy, CCIStrategy, WilliamsRStrategy,
    SqueezeMomentumStrategy, ChaikinMFStrategy, ElderRayStrategy,
    HeikinAshiEMAStrategy, ParabolicSARStrategy,
)
from strategies.trend_momentum_strategies import (
    DualMomentumStrategy, TrendStrengthStrategy, VolumeTrendStrategy,
    AdaptiveChannelStrategy, MultiSignalStrategy,
)
from strategies.price_action_strategies import (
    SuperTrendStrategy, VWAPStrategy, HullMAStrategy,
    KeltnerBreakoutStrategy, PinBarStrategy, RSIDivergenceStrategy,
)
from strategies.ma_cross_strategies import (
    MA300_1000Strategy, EMAStackStrategy,
    GoldenCrossStrategy, DynamicMAStrategy,
)

# ── Config global ──────────────────────────────────────────────────────────────
PAIRS = [
    "BTC-USDT","ETH-USDT","SOL-USDT","XRP-USDT","BNB-USDT",
    "ADA-USDT","AVAX-USDT","DOGE-USDT","LTC-USDT","DOT-USDT",
]
MAX_HOLD=24; PPY=8760; FEE_RT=0.002
TRAIL=3.0; PARTIAL=0.33; ADX_MIN=13
CAPITAL_INIT=100.0

SIZER=KellySizer(variant="full_kelly",min_trades=20,max_fraction=0.60,min_fraction=0.01)
RM=RiskManager(method="atr",atr_multiplier=1.0,rr_ratio=2.0)

def _wrap(s):
    return RegimeFilteredStrategy(s,adx_min=ADX_MIN,di_align=True,
                                  atr_min_pct=0.2,atr_max_pct=6.0)

# ── Demo data ──────────────────────────────────────────────────────────────────
def _gen(sym, n=8760):
    seeds={p:i for i,p in enumerate(PAIRS)}
    rng=np.random.default_rng(seeds.get(sym,0))
    base={"BTC-USDT":40000,"ETH-USDT":2000,"SOL-USDT":80,"BNB-USDT":300}.get(sym,10.)
    p=[base]
    for _ in range(n-1): p.append(max(1e-4,p[-1]+0.3+rng.standard_normal()*p[-1]*0.008))
    p=np.array(p); o=np.concatenate([[p[0]],p[:-1]])
    s=np.abs(rng.standard_normal(n))*p*0.003
    ts=pd.to_datetime([int(time.time())-(n-i)*3600 for i in range(n)],unit="s",utc=True)
    return pd.DataFrame({"open":o,"high":np.maximum(o,p)+s,
                         "low":np.minimum(o,p)-s,"close":p,
                         "volume":rng.integers(500,20000,n).astype(float)},index=ts)

# ── Motor ──────────────────────────────────────────────────────────────────────
def _run(name, strat, datasets):
    all_m,all_eq=[],[]
    for df in datasets.values():
        try:
            sigs=strat.generate_signals_batch(df)
            trades,eq=run_kelly_v2(sigs,df,SIZER,max_hold=MAX_HOLD,risk_manager=RM,
                                   trailing_atr_mult=TRAIL,partial_tp=True,partial_ratio=PARTIAL)
            m=kelly_metrics_v2(trades,eq,periods_per_year=PPY)
            all_m.append(m); all_eq.append(eq)
        except Exception: pass
    if not all_m: return None
    np2=len(all_m); af=float(np.mean([m["avg_fraction"] for m in all_m]))
    tt=sum(m["trades"] for m in all_m)
    gp=float(np.mean([(m["equity_curve"][-1]-1)*100 for m in all_m if m["equity_curve"]]))
    fp=(tt/np2)*FEE_RT*af*100; net=gp-fp
    gs=float(np.mean([m["sharpe"] for m in all_m]))
    ns=gs*net/gp if abs(gp)>1e-6 else 0.
    mo=sum(len(df) for df in datasets.values())/(720*np2)
    tm=(tt/np2)/mo if mo else 0
    ml=min(len(e) for e in all_eq)
    eq_m=np.mean([e[:ml] for e in all_eq],axis=0).tolist()
    return {"name":name,"net_sharpe":round(ns,3),"net_pct":round(net,2),
            "gross_pct":round(gp,2),"fee_pct":round(fp,2),
            "winrate":round(float(np.mean([m["winrate"] for m in all_m]))*100,1),
            "pf":round(float(np.mean([m["profit_factor"] for m in all_m])),2),
            "max_dd":round(float(np.mean([m["max_drawdown"] for m in all_m]))*100,1),
            "t_month":round(tm,1),"calmar":round(float(np.mean([m["calmar"] for m in all_m])),2),
            "frac_pct":round(af*100,1),"eq_curve":eq_m}

# ── 100 configuraciones ────────────────────────────────────────────────────────
def build_100():
    C=[]
    def a(n,s): C.append((n,_wrap(s)))
    def ab(n,s): C.append((n,s))   # sin régimen

    # A — MTF-SMC (6)
    for sw,e4 in [(3,50),(5,50),(7,50),(3,100),(5,100),(10,50)]:
        ab(f"MTF_sw{sw}_e{e4}", MultiTFSMC(swing_window=sw,trend_ema=e4))

    # B — Advanced (10×2=20)
    for sw in [3,5]:
        a(f"SMC_sw{sw}",          SMCStrategy(swing_window=sw))
    ab("AdaptMTF",               AdaptiveMTFStrategy())
    a("StochRSI_14",             StochRSIStrategy())
    a("Ichimoku",                IchimokuStrategy())
    a("FibRet",                  FibRetracementStrategy())
    a("MktStruct",               MarketStructureStrategy())
    a("MACD_RSI",                MACDRSIStrategy())
    a("LinReg",                  LinearRegStrategy())
    a("BBMom",                   BollingerMomStrategy())
    a("OrdBlock",                OrderBlockStrategy())
    a("BrkVol",                  BreakoutVolStrategy())

    # C — Statistical (13)
    a("ZScore_50",               ZScoreStrategy(period=50,entry_z=2.0))
    a("ZScore_30",               ZScoreStrategy(period=30,entry_z=1.8))
    a("BBRevert_20",             BollingerReversionStrategy(period=20,mult=2.0))
    a("BBRevert_14",             BollingerReversionStrategy(period=14,mult=1.8))
    a("OBVTrend",                OBVTrendStrategy())
    a("MFI_14",                  MFIStrategy(period=14))
    a("ForceIdx",                ForceIndexStrategy())
    a("AccDist",                 AccDistStrategy())
    a("ROC_10",                  ROCStrategy(period=10))
    a("ROC_20",                  ROCStrategy(period=20))
    a("TSI",                     TSIStrategy())
    a("CMO_14",                  CMOStrategy(period=14))
    a("InsideBar",               InsideBarBreakoutStrategy())
    a("FVG",                     FVGStrategy())
    a("VolOsc",                  VolumeOscStrategy())
    a("UltOsc",                  UltimateOscStrategy())

    # D — Hybrid (12)
    a("MACDStoch",               MACDStochStrategy())
    a("RSI_BB",                  RSIBollingerStrategy())
    a("ADX_MACD",                ADXMACDStrategy())
    a("TripleScreen",            TripleScreenStrategy())
    a("ScalpMTF",                ScalpingMTFStrategy())
    a("MomBreakout_20",          MomentumBreakoutStrategy(period=20))
    a("MomBreakout_10",          MomentumBreakoutStrategy(period=10))
    a("TrendRev",                TrendReversalStrategy())
    a("VolBreak",                VolatilityBreakoutStrategy())
    a("SmartFlow",               SmartMoneyFlowStrategy())
    a("AdaptRSI",                AdaptiveRSIStrategy())
    a("Pullback",                PullbackTrendStrategy())
    a("DivConfl",                DivergenceConfluenceStrategy())

    # E — Momentum Oscillators (8×2=16, tomamos 10)
    a("AO",                      AwesomeOscillatorStrategy())
    a("CCI_20",                  CCIStrategy(period=20))
    a("CCI_14",                  CCIStrategy(period=14))
    a("WillR_14",                WilliamsRStrategy(period=14))
    a("Squeeze",                 SqueezeMomentumStrategy())
    a("CMF",                     ChaikinMFStrategy())
    a("ElderRay",                ElderRayStrategy())
    a("HeikinAshi_3",            HeikinAshiEMAStrategy(consecutive=3))
    a("HeikinAshi_2",            HeikinAshiEMAStrategy(consecutive=2))
    a("ParSAR",                  ParabolicSARStrategy())

    # F — Trend Momentum (5×2=10)
    a("DualMom_12_48",           DualMomentumStrategy(fast_p=12,slow_p=48))
    a("DualMom_6_24",            DualMomentumStrategy(fast_p=6,slow_p=24))
    a("TrendStr_20",             TrendStrengthStrategy(adx_min=20))
    a("TrendStr_25",             TrendStrengthStrategy(adx_min=25))
    a("VolTrend",                VolumeTrendStrategy())
    a("AdaptCh_1x",              AdaptiveChannelStrategy(atr_mult=1.0))
    a("AdaptCh_15x",             AdaptiveChannelStrategy(atr_mult=1.5))
    a("MultiSig3",               MultiSignalStrategy(score_min=3))
    a("MultiSig4",               MultiSignalStrategy(score_min=4))

    # G — Price Action (6)
    a("SuperTrend_10_3",         SuperTrendStrategy(atr_period=10,multiplier=3.0))
    a("SuperTrend_7_25",         SuperTrendStrategy(atr_period=7,multiplier=2.5))
    a("HullMA_20_55",            HullMAStrategy(fast=20,slow=55))
    a("HullMA_10_30",            HullMAStrategy(fast=10,slow=30))
    a("Keltner_15",              KeltnerBreakoutStrategy(mult=1.5))
    a("PinBar",                  PinBarStrategy())

    # H — MA Cross (4×2=8 → tomamos 5)
    a("EMAStack_50_100_200",     EMAStackStrategy(fast=50,mid=100,slow=200))
    a("EMAStack_20_50_100",      EMAStackStrategy(fast=20,mid=50,slow=100))
    a("GoldCross",               GoldenCrossStrategy())
    a("DynamicMA",               DynamicMAStrategy())
    a("MA300_1000",              MA300_1000Strategy())

    print(f"  [C100] {len(C)} configuraciones.")
    return C

# ── Resultados ─────────────────────────────────────────────────────────────────
def print_results(rows):
    valid=sorted([r for r in rows if r and r["net_pct"]>0],
                 key=lambda x:-x["net_sharpe"])

    total=len([r for r in rows if r])
    print(f"\n{'='*125}")
    print(f"  COMPETICIÓN 100 ESTRATEGIAS")
    print(f"  Rentables: {len(valid)}/{total}  |  Trail {TRAIL}  "
          f"Partial {PARTIAL}  ADX>{ADX_MIN}  |  Capital: €{CAPITAL_INIT:.0f}")
    print(f"{'='*125}")
    print(f"  {'#':>3}  {'Estrategia':<30}  "
          f"{'NetSh':>6}  {'Net%/yr':>8}  {'WR%':>5}  "
          f"{'PF':>5}  {'t/mes':>5}  {'MaxDD':>6}  {'Cal':>6}")
    print("  "+"─"*123)

    medals={0:"🥇",1:"🥈",2:"🥉"}
    for i,r in enumerate(valid[:25]):
        m=medals.get(i,"  ")
        print(f"  {i+1:>3}. {m} {r['name']:<28}  "
              f"{r['net_sharpe']:>6.3f}  {r['net_pct']:>+8.2f}%  "
              f"{r['winrate']:>5.1f}%  {r['pf']:>5.2f}  "
              f"{r['t_month']:>5.1f}  {r['max_dd']:>+6.1f}%  "
              f"{r['calmar']:>6.2f}")

    if not valid: return
    best=valid[0]; py=best["net_pct"]; pmo=(1+py/100)**(1/12)-1

    # Tabla mensual desde €100
    print(f"\n{'='*125}")
    print(f"  GANADORA: {best['name']}  |  {py:+.2f}%/año  |  "
          f"Sharpe {best['net_sharpe']:.3f}  |  WR {best['winrate']:.1f}%  |  "
          f"MaxDD {best['max_dd']:+.1f}%  |  Calmar {best['calmar']:.2f}")
    print(f"  Tasa mensual: {pmo*100:.2f}%/mes\n")

    print(f"  PROYECCIÓN €{CAPITAL_INIT:.0f} → 36 meses:")
    print(f"  {'─'*70}")
    cap=CAPITAL_INIT
    for mo in list(range(1,13))+[18,24,30,36]:
        prev=CAPITAL_INIT*(1+py/100)**((mo-1)/12)
        cap=CAPITAL_INIT*(1+py/100)**(mo/12)
        mark=" ←" if mo in [9,12,24,36] else ""
        print(f"  Mes {mo:>2}: €{cap:>9,.2f}  "
              f"(+€{cap-prev:>8,.2f}/mes  total +€{cap-CAPITAL_INIT:>9,.2f}  "
              f"{cap/CAPITAL_INIT:.1f}x){mark}")
    print(f"  {'─'*70}")

    # Top 5 comparativa
    print(f"\n  TOP 5 COMPARATIVA:")
    print(f"  {'─'*90}")
    print(f"  {'#':<4} {'Estrategia':<30} {'%/año':>8}  "
          f"{'12m':>9}  {'24m':>9}  {'36m':>9}")
    print(f"  {'─'*90}")
    for i,r in enumerate(valid[:5]):
        v1=CAPITAL_INIT*(1+r["net_pct"]/100)**1-CAPITAL_INIT
        v2=CAPITAL_INIT*(1+r["net_pct"]/100)**2-CAPITAL_INIT
        v3=CAPITAL_INIT*(1+r["net_pct"]/100)**3-CAPITAL_INIT
        print(f"  {i+1:<4} {r['name']:<30} {r['net_pct']:>+8.2f}%  "
              f"+€{v1:>7,.2f}  +€{v2:>7,.2f}  +€{v3:>7,.2f}")
    print(f"  {'─'*90}")
    print(f"\n  Para €100/mes → capital mínimo: €{100/pmo:>8,.0f}")
    print(f"  Para €500/mes → capital mínimo: €{500/pmo:>8,.0f}")
    print(f"{'='*125}\n")
    return valid

def save_plot(rows):
    valid=sorted([r for r in rows if r and r["net_pct"]>0],
                 key=lambda x:-x["net_sharpe"])
    if not valid: return

    fig=plt.figure(figsize=(24,14)); fig.patch.set_facecolor("#0d1117")
    gs=fig.add_gridspec(2,3,hspace=0.45,wspace=0.35)
    def ax_(p):
        ax=fig.add_subplot(p); ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9",labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#30363d")
        return ax

    pal=plt.cm.plasma(np.linspace(0.05,0.95,min(20,len(valid))))

    # 1. Equity €100
    ax1=ax_(gs[0,:2])
    for i,r in enumerate(valid[:20]):
        eq=r.get("eq_curve",[])
        if eq:
            xs=np.linspace(0,12,len(eq))
            ax1.plot(xs,[CAPITAL_INIT*v for v in eq],
                     color=pal[i],lw=1.3,alpha=0.7,
                     label=f"{r['name'][:18]}({r['net_pct']:>+.0f}%)" if i<6 else "")
    ax1.axhline(CAPITAL_INIT,color="#8b949e",lw=0.8,ls="--")
    ax1.legend(fontsize=6.5,framealpha=0.3,labelcolor="#c9d1d9")
    ax1.set_title(f"Top 20 Equity desde €{CAPITAL_INIT:.0f}",color="#f0f6fc",fontsize=10)
    ax1.set_xlabel("Meses",color="#8b949e"); ax1.set_ylabel("€",color="#8b949e")

    # 2. WR vs Sharpe
    ax2=ax_(gs[0,2])
    ax2.scatter([r["winrate"] for r in valid[:40]],
                [r["net_sharpe"] for r in valid[:40]],
                c=np.arange(min(40,len(valid))),cmap="plasma",s=50,alpha=0.85)
    ax2.axvline(50,color="#f85149",lw=0.8,ls="--",alpha=0.5)
    ax2.set_title("Win Rate vs Net Sharpe",color="#f0f6fc",fontsize=10)
    ax2.set_xlabel("Win Rate %",color="#8b949e"); ax2.set_ylabel("Net Sharpe",color="#8b949e")

    # 3. Bar top 25
    ax3=ax_(gs[1,:2])
    top25=valid[:25]
    ax3.barh([r["name"][:28] for r in top25],[r["net_pct"] for r in top25],
             color=pal[:len(top25)],alpha=0.85)
    ax3.set_xlabel("Net Return %/año",color="#8b949e")
    ax3.set_title("Top 25 Net Return — Competición 100",color="#f0f6fc",fontsize=10)
    ax3.invert_yaxis()

    # 4. Proyección €100 top 5
    ax4=ax_(gs[1,2])
    for i,r in enumerate(valid[:5]):
        py=r["net_pct"]; mo=list(range(1,37))
        ax4.plot(mo,[CAPITAL_INIT*(1+py/100)**(m/12) for m in mo],
                 color=pal[i],lw=2,label=f"{r['name'][:16]}({py:>+.0f}%)")
    ax4.axhline(CAPITAL_INIT,color="#8b949e",lw=0.8,ls="--")
    ax4.set_title(f"€{CAPITAL_INIT:.0f} → 36 meses",color="#f0f6fc",fontsize=10)
    ax4.set_xlabel("Mes",color="#8b949e"); ax4.set_ylabel("€",color="#8b949e")
    ax4.legend(fontsize=6,framealpha=0.3,labelcolor="#c9d1d9")

    fig.suptitle(f"COMPETICIÓN 100 ESTRATEGIAS | €{CAPITAL_INIT:.0f} | "
                 f"Trail {TRAIL} + Partial {PARTIAL} | 1H 1 año 10 pares",
                 color="#f0f6fc",fontsize=12,y=1.01)
    os.makedirs("data",exist_ok=True)
    out="data/competition_100.png"
    plt.savefig(out,dpi=140,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(); print(f"  Gráfico: {out}")

def main():
    print("\n"+"="*65)
    print(f"  COMPETICIÓN 100 ESTRATEGIAS — cargando datos...")
    print("="*65)
    t0=time.time()
    datasets={p:_gen(p) for p in PAIRS}
    print(f"  Datos: {time.time()-t0:.1f}s\n")

    configs=build_100()
    total=len(configs); results=[]; t1=time.time()
    print(f"  Ejecutando {total} estrategias...\n")

    for i,(name,strat) in enumerate(configs):
        r=_run(name,strat,datasets); results.append(r)
        status=""
        if r and r["net_pct"]>0:
            status=f"✅ {r['net_pct']:>+7.2f}%  Sh {r['net_sharpe']:.3f}"
        elif r:
            status=f"❌ {r['net_pct']:>+7.2f}%"
        else:
            status="❌ sin datos"
        print(f"  [{i+1:>3}/{total}] {name:<30} {status}")

    print(f"\n  Completado en {time.time()-t1:.1f}s")
    os.makedirs("data",exist_ok=True)
    df_out=pd.DataFrame([r for r in results if r])
    df_out=df_out.drop(columns=["eq_curve"],errors="ignore")
    df_out.to_csv("data/competition_100_results.csv",index=False)
    print("  CSV: data/competition_100_results.csv")

    valid=print_results(results)
    save_plot(results)
    return results

if __name__=="__main__":
    main()
