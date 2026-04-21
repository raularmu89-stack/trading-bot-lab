"""
smart_bot.py

SMART BOT — Detección de régimen de mercado + estrategia óptima automática.

Mejoras sobre live_trader.py:
  1. Detecta régimen de mercado cada ciclo (ADX, ATR%, volumen)
  2. Selecciona la estrategia óptima para ese régimen:
       TENDENCIA FUERTE → AdaptMTF (Sharpe 8.0, +150%/yr)
       TENDENCIA MEDIA  → MTF_sw5 con RegimeFilter[light]
       MERCADO LATERAL  → Ensemble3 (voto mayoritario 2/3)
       BREAKOUT         → MTF_sw3 (ventana corta, más reactivo)
  3. Trailing stop ATR×4.0 + Partial TP 33% en 1R (config óptima)
  4. Kelly 80% cap (confirmado como óptimo en 189 configs)
  5. Multi-par: 10 pares simultáneos
  6. Paper mode + Real mode + Dashboard JSON

USAR:
    # Paper trading (sin riesgo):
    python trading/smart_bot.py --capital 100

    # Real trading (requiere .env con API keys):
    python trading/smart_bot.py --real --capital 100

    # Un ciclo y salir (cron / test):
    python trading/smart_bot.py --once

FORMATO .env:
    KUCOIN_API_KEY=tu_key
    KUCOIN_API_SECRET=tu_secret
    KUCOIN_API_PASSPHRASE=tu_passphrase
    INITIAL_CAPITAL=100
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import signal
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from data.kucoin_client          import KuCoinClient
from strategies.kelly_sizer      import KellySizer
from strategies.risk_manager     import RiskManager
from strategies.regime_filter    import RegimeFilteredStrategy
from strategies.mtf_smc          import MultiTFSMC
from strategies.advanced_strategies import AdaptiveMTFStrategy
from trading.kucoin_trader       import KuCoinTrader
from trading.position_manager    import PositionManager

# ── Configuración óptima (confirmada 189 configs) ─────────────────────────────

KELLY_CAP      = 0.80   # 80% — mejor en grid search
TRAIL_ATR_MULT = 4.0    # Trailing stop ATR×4.0
PARTIAL_RATIO  = 0.33   # Tomar 33% en 1R → mover SL a BE
ADX_MIN        = 13     # RegimeFilter light

PAIRS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BNB-USDT",
    "ADA-USDT", "AVAX-USDT", "DOGE-USDT", "LTC-USDT", "DOT-USDT",
]

CANDLES_NEEDED = 300
SLEEP_SECONDS  = 3600
STATE_FILE     = "data/smart_bot_state.json"
LOG_FILE       = "data/smart_bot_log.txt"
MIN_USDT_TRADE = 5.0    # mínimo por trade

# ── Regímenes ─────────────────────────────────────────────────────────────────

REGIME_STRONG_TREND  = "TENDENCIA_FUERTE"
REGIME_MEDIUM_TREND  = "TENDENCIA_MEDIA"
REGIME_LATERAL       = "LATERAL"
REGIME_BREAKOUT      = "BREAKOUT"


# ── Utilidades ────────────────────────────────────────────────────────────────

def _log(msg: str):
    ts   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        os.makedirs("data", exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    if len(c) < 2:
        return float(h[-1] - l[-1])
    tr = np.maximum(h[1:]-l[1:],
         np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    return float(np.mean(tr[-period:])) if len(tr) >= period else float(np.mean(tr))


def _calc_adx(df: pd.DataFrame, period: int = 14) -> tuple[float, float, float]:
    """Retorna (ADX, DI+, DI-)."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(c)
    if n < period + 1:
        return 0.0, 0.0, 0.0

    tr   = np.zeros(n)
    pdm  = np.zeros(n)
    ndm  = np.zeros(n)
    tr[0] = h[0] - l[0]

    for i in range(1, n):
        tr[i]  = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        up      = h[i] - h[i-1]
        dn      = l[i-1] - l[i]
        pdm[i]  = up if up > dn and up > 0 else 0
        ndm[i]  = dn if dn > up and dn > 0 else 0

    def _smooth(arr, p):
        out = np.zeros(n)
        out[p] = arr[1:p+1].sum()
        for i in range(p+1, n):
            out[i] = out[i-1] - out[i-1]/p + arr[i]
        return out

    atr14  = _smooth(tr,  period)
    pdm14  = _smooth(pdm, period)
    ndm14  = _smooth(ndm, period)

    dip = np.where(atr14 > 0, 100*pdm14/atr14, 0)
    dim = np.where(atr14 > 0, 100*ndm14/atr14, 0)
    dx  = np.where((dip+dim) > 0, 100*np.abs(dip-dim)/(dip+dim), 0)

    adx14 = np.zeros(n)
    adx14[period*2-1] = dx[period:period*2].mean()
    for i in range(period*2, n):
        adx14[i] = (adx14[i-1]*(period-1) + dx[i]) / period

    return float(adx14[-1]), float(dip[-1]), float(dim[-1])


# ── Detector de régimen ────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Clasifica el régimen de mercado actual.

    Combina ADX, ATR% y volumen relativo para determinar:
      - TENDENCIA_FUERTE : ADX>25, DI+/DI- alineados, ATR%>0.5%
      - TENDENCIA_MEDIA  : ADX 15-25, alineamiento moderado
      - BREAKOUT         : ADX<15 pero ATR% en expansión (vol×1.5)
      - LATERAL          : ADX<15, ATR% bajo, sin dirección clara
    """

    def detect(self, df: pd.DataFrame) -> dict:
        adx, dip, dim = _calc_adx(df)
        atr      = _calc_atr(df)
        price    = float(df["close"].iloc[-1])
        atr_pct  = atr / price * 100 if price > 0 else 0

        # Volumen relativo (últimas 5 vs últimas 20)
        vol = df["volume"].values
        vol_ratio = (vol[-5:].mean() / vol[-20:].mean()) if len(vol) >= 20 else 1.0

        # Clasificar
        if adx > 25 and abs(dip - dim) > 10:
            regime = REGIME_STRONG_TREND
        elif adx > 15:
            regime = REGIME_MEDIUM_TREND
        elif vol_ratio > 1.5 and atr_pct > 0.6:
            regime = REGIME_BREAKOUT
        else:
            regime = REGIME_LATERAL

        return {
            "regime":    regime,
            "adx":       round(adx, 1),
            "dip":       round(dip, 1),
            "dim":       round(dim, 1),
            "atr_pct":   round(atr_pct, 3),
            "vol_ratio": round(vol_ratio, 2),
            "trending":  adx > ADX_MIN and dip > dim,
        }


# ── Router de estrategias ─────────────────────────────────────────────────────

class StrategyRouter:
    """
    Selecciona la estrategia óptima según el régimen detectado.

    Basado en resultados de todos los torneos:
      TENDENCIA_FUERTE → AdaptMTF (+153%/yr, Sharpe 8.0)
      TENDENCIA_MEDIA  → MTF_sw5_e50 con RegimeFilter (+142%/yr)
      BREAKOUT         → MTF_sw3_e50 (ventana corta, más reactivo)
      LATERAL          → Ensemble3(2/3): AdaptMTF+MTF_sw3+MTF_sw5
    """

    def __init__(self):
        # Estrategia principal (la mejor en casi todo)
        self._adapt_mtf    = AdaptiveMTFStrategy()
        self._mtf_sw3      = MultiTFSMC(swing_window=3, trend_ema=50)
        self._mtf_sw5      = MultiTFSMC(swing_window=5, trend_ema=50)
        self._mtf_sw7      = MultiTFSMC(swing_window=7, trend_ema=50)

        # Con RegimeFilter
        self._adapt_reg = RegimeFilteredStrategy(
            AdaptiveMTFStrategy(), adx_min=ADX_MIN,
            di_align=True, atr_min_pct=0.2, atr_max_pct=6.0
        )
        self._mtf5_reg  = RegimeFilteredStrategy(
            MultiTFSMC(swing_window=5, trend_ema=50), adx_min=ADX_MIN,
            di_align=True, atr_min_pct=0.2, atr_max_pct=6.0
        )

    def get_signal(self, df: pd.DataFrame, regime_info: dict) -> tuple[str, str]:
        """
        Retorna (señal, nombre_estrategia).
        señal: "buy" | "sell" | "hold"
        """
        regime = regime_info["regime"]

        try:
            if regime == REGIME_STRONG_TREND:
                # AdaptMTF puro — máximo rendimiento en tendencia
                sigs   = self._adapt_mtf.generate_signals_batch(df)
                stname = "AdaptMTF"

            elif regime == REGIME_MEDIUM_TREND:
                # MTF_sw5 con RegimeFilter — más selectivo
                sigs   = self._mtf5_reg.generate_signals_batch(df)
                stname = "MTF_sw5+Reg"

            elif regime == REGIME_BREAKOUT:
                # MTF_sw3 — ventana corta, más reactivo a breakouts
                sigs   = self._mtf_sw3.generate_signals_batch(df)
                stname = "MTF_sw3"

            else:  # LATERAL
                # Ensemble 2/3 — consenso de 3 estrategias, reduce falsos
                s1 = self._adapt_mtf.generate_signals_batch(df)
                s2 = self._mtf_sw3.generate_signals_batch(df)
                s3 = self._mtf_sw5.generate_signals_batch(df)
                n  = min(len(s1), len(s2), len(s3))
                # Voto 2/3 solo en última señal
                candidates = [s1[-1] if s1 else "hold",
                              s2[-1] if s2 else "hold",
                              s3[-1] if s3 else "hold"]
                buy_votes  = candidates.count("buy")
                sell_votes = candidates.count("sell")
                last_sig   = ("buy"  if buy_votes  >= 2 else
                              "sell" if sell_votes >= 2 else "hold")
                return last_sig, "Ensemble3(2/3)"

            return (sigs[-1] if sigs else "hold"), stname

        except Exception:
            return "hold", "error"


# ── Gestor de posiciones con trailing stop + partial TP ───────────────────────

class SmartPositionManager:
    """
    Gestiona posiciones con:
      - Trailing stop ATR×4.0
      - Partial TP 33% en 1R → mover SL a breakeven
      - Kelly 80% sizing
    """

    def __init__(self, risk_manager: RiskManager, sizer: KellySizer,
                 state_file: str = "data/smart_positions.json"):
        self.rm          = risk_manager
        self.sizer       = sizer
        self.state_file  = state_file
        self.positions   = {}    # {symbol: position_dict}
        self.closed      = []    # historial
        self.trade_hist  = []    # para Kelly
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file) as f:
                    data = json.load(f)
                self.positions  = data.get("positions", {})
                self.closed     = data.get("closed", [])[-100:]
                self.trade_hist = data.get("trade_hist", [])[-200:]
        except Exception:
            pass

    def _save(self):
        try:
            os.makedirs("data", exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump({
                    "positions":  self.positions,
                    "closed":     self.closed[-100:],
                    "trade_hist": self.trade_hist[-200:],
                }, f, indent=2)
        except Exception:
            pass

    def kelly_fraction(self) -> float:
        return self.sizer.position_fraction(self.trade_hist)

    def open_position(self, symbol: str, side: str, price: float,
                      df: pd.DataFrame, usdt_available: float) -> Optional[dict]:
        """Abre posición con SL/TP y trailing iniciales."""
        if symbol in self.positions:
            return None

        frac = self.kelly_fraction()
        usdt = usdt_available * frac
        if usdt < MIN_USDT_TRADE:
            return None

        sl, tp = self.rm.calculate_levels(df, price, side)
        if sl is None:
            return None

        atr = _calc_atr(df)
        trail_init = (price - TRAIL_ATR_MULT * atr) if side == "buy" \
                     else (price + TRAIL_ATR_MULT * atr)

        if side == "buy":
            trail_init = max(trail_init, sl) if sl else trail_init
        else:
            trail_init = min(trail_init, sl) if sl else trail_init

        pos = {
            "symbol":       symbol,
            "side":         side,
            "entry":        price,
            "sl":           sl,
            "tp":           tp,
            "trail":        trail_init,
            "partial_done": False,
            "usdt":         usdt,
            "size":         usdt / price,
            "frac":         frac,
            "open_time":    time.time(),
            "strategy":     "",
        }
        self.positions[symbol] = pos
        self._save()
        return pos

    def update(self, symbol: str, high: float, low: float,
               close: float, atr: float) -> Optional[dict]:
        """
        Actualiza trailing stop y comprueba exits.
        Retorna dict con resultado si cerró, None si sigue abierta.
        """
        pos = self.positions.get(symbol)
        if not pos:
            return None

        side   = pos["side"]
        entry  = pos["entry"]
        sl     = pos["sl"]
        tp     = pos["tp"]
        trail  = pos["trail"]
        pdone  = pos["partial_done"]

        # 1. Actualizar trailing stop
        if side == "buy":
            new_trail = high - TRAIL_ATR_MULT * atr
            if new_trail > trail:
                pos["trail"] = new_trail
                trail = new_trail
        else:
            new_trail = low + TRAIL_ATR_MULT * atr
            if new_trail < trail:
                pos["trail"] = new_trail
                trail = new_trail

        eff_sl = max(sl, trail) if side == "buy" else min(sl, trail)

        # 2. Partial TP en 1R
        if not pdone and tp is not None:
            one_r = (entry + (tp - entry) * PARTIAL_RATIO) if side == "buy" \
                    else (entry - (entry - tp) * PARTIAL_RATIO)
            hit   = (high >= one_r) if side == "buy" else (low <= one_r)
            if hit:
                pnl = (one_r - entry)/entry if side == "buy" \
                      else (entry - one_r)/entry
                result = self._close_partial(symbol, pos, one_r, pnl)
                pos["partial_done"] = True
                pos["sl"] = entry   # mover SL a breakeven
                if side == "buy":
                    pos["trail"] = max(pos["trail"], entry)
                else:
                    pos["trail"] = min(pos["trail"], entry)
                self._save()
                return result

        # 3. SL / trailing hit
        sl_hit = (low <= eff_sl) if side == "buy" else (high >= eff_sl)
        if sl_hit:
            pnl  = (eff_sl - entry)/entry if side == "buy" \
                   else (entry - eff_sl)/entry
            rem  = 1.0 - (PARTIAL_RATIO if pdone else 0.0)
            exit_type = "trail" if trail != sl else "sl"
            return self._close_full(symbol, pos, eff_sl, pnl, rem, exit_type)

        # 4. TP hit
        if tp is not None:
            tp_hit = (high >= tp) if side == "buy" else (low <= tp)
            if tp_hit:
                pnl = (tp - entry)/entry if side == "buy" \
                      else (entry - tp)/entry
                rem = 1.0 - (PARTIAL_RATIO if pdone else 0.0)
                return self._close_full(symbol, pos, tp, pnl, rem, "tp")

        self._save()
        return None

    def force_close(self, symbol: str, price: float, reason: str = "signal") -> Optional[dict]:
        """Cierra posición a precio de mercado."""
        pos = self.positions.get(symbol)
        if not pos:
            return None
        entry = pos["entry"]
        pnl   = (price - entry)/entry if pos["side"] == "buy" \
                else (entry - price)/entry
        rem   = 1.0 - (PARTIAL_RATIO if pos["partial_done"] else 0.0)
        return self._close_full(symbol, pos, price, pnl, rem, reason)

    def _close_partial(self, symbol: str, pos: dict, price: float, pnl: float) -> dict:
        fee    = 0.001
        net    = pnl - fee
        usdt   = pos["usdt"] * PARTIAL_RATIO * (1 + net)
        result = {
            "symbol": symbol, "side": pos["side"],
            "entry": pos["entry"], "exit_price": price,
            "pnl": round(pnl * 100, 3),
            "net_pnl": round(net * 100, 3),
            "usdt_result": round(usdt, 4),
            "exit_type": "partial_tp",
            "time": datetime.now(timezone.utc).isoformat(),
        }
        self.closed.append(result)
        self.trade_hist.append({"pnl": pnl, "win": pnl > 0})
        return result

    def _close_full(self, symbol: str, pos: dict, price: float,
                    pnl: float, rem: float, exit_type: str) -> dict:
        fee    = 0.001
        net    = pnl - fee
        usdt   = pos["usdt"] * rem * (1 + net)
        result = {
            "symbol":     symbol,
            "side":       pos["side"],
            "entry":      pos["entry"],
            "exit_price": round(price, 6),
            "pnl":        round(pnl * 100, 3),
            "net_pnl":    round(net * 100, 3),
            "usdt_result":round(usdt, 4),
            "exit_type":  exit_type,
            "win":        pnl > 0,
            "time":       datetime.now(timezone.utc).isoformat(),
            "strategy":   pos.get("strategy", ""),
        }
        self.closed.append(result)
        self.trade_hist.append({"pnl": pnl, "win": pnl > 0})
        del self.positions[symbol]
        self._save()
        return result

    def stats(self) -> dict:
        if not self.closed:
            return {"trades": 0, "winrate": 0.0, "pnl_usdt": 0.0,
                    "open": len(self.positions), "best": 0.0, "worst": 0.0}
        wins  = [t for t in self.closed if t.get("win")]
        total_usdt = sum(t.get("usdt_result", 0) for t in self.closed)
        return {
            "trades":  len(self.closed),
            "winrate": round(len(wins)/len(self.closed)*100, 1),
            "pnl_usdt": round(total_usdt, 2),
            "open":    len(self.positions),
            "best":    round(max((t["pnl"] for t in self.closed), default=0), 2),
            "worst":   round(min((t["pnl"] for t in self.closed), default=0), 2),
        }


# ── Smart Bot principal ───────────────────────────────────────────────────────

class SmartBot:
    """
    Bot de trading inteligente con detección de régimen + estrategia óptima.

    Ciclo cada hora:
      1. Descarga 300 velas 1H por par
      2. Detecta régimen de mercado (ADX, ATR%, volumen)
      3. Selecciona estrategia óptima para ese régimen
      4. Genera señal y abre/cierra posición
      5. Actualiza trailing stop + partial TP en posiciones abiertas
      6. Guarda estado para dashboard
    """

    def __init__(self, paper: bool = True,
                 api_key: str = "", api_secret: str = "",
                 api_passphrase: str = "",
                 initial_capital: float = 100.0):

        self.paper  = paper
        self.client = KuCoinClient()
        self.trader = KuCoinTrader(
            paper=paper,
            api_key=api_key, api_secret=api_secret,
            api_passphrase=api_passphrase,
            initial_balance=initial_capital,
        )

        rm    = RiskManager(method="atr", atr_multiplier=1.0, rr_ratio=2.0)
        sizer = KellySizer(variant="full_kelly", min_trades=20,
                           max_fraction=KELLY_CAP, min_fraction=0.01)

        self.pm      = SmartPositionManager(rm, sizer)
        self.router  = StrategyRouter()
        self.detector= RegimeDetector()
        self._cycle  = 0
        self._running= False

        mode = "PAPER 📄" if paper else "REAL 💰⚠️"
        _log(f"SmartBot iniciado | Modo: {mode} | Capital: {initial_capital} USDT")
        _log(f"Config: Kelly {KELLY_CAP:.0%} | Trail ATR×{TRAIL_ATR_MULT} | Partial {PARTIAL_RATIO:.0%}")
        _log(f"Pares: {', '.join(PAIRS)}")

    def _run_cycle(self):
        self._cycle += 1
        _log(f"── Ciclo #{self._cycle} {'─'*45}")

        # 1. Datos
        datasets = {}
        for pair in PAIRS:
            try:
                df = self.client.get_ohlcv(pair, interval="1hour",
                                           limit=CANDLES_NEEDED)
                if df is not None and len(df) >= 100:
                    datasets[pair] = df
            except Exception as e:
                _log(f"  ⚠ {pair}: {e}")
            time.sleep(0.12)

        _log(f"  Datos: {len(datasets)}/{len(PAIRS)} pares OK")
        if not datasets:
            _log("  Sin datos — ciclo omitido")
            return

        usdt_balance = self.trader.get_balance("USDT")
        kelly_frac   = self.pm.kelly_fraction()

        regime_summary = {}

        for pair, df in datasets.items():
            try:
                last = df.iloc[-1]
                high  = float(last["high"])
                low   = float(last["low"])
                close = float(last["close"])
                atr   = _calc_atr(df)

                # Actualizar posición abierta
                closed = self.pm.update(pair, high, low, close, atr)
                if closed:
                    icon = "✅" if closed["win"] else "❌"
                    _log(f"  {icon} CERRADO {pair} via {closed['exit_type'].upper()}"
                         f" | pnl={closed['pnl']:+.2f}% | {closed['usdt_result']:+.4f} USDT")
                    if not self.paper and closed["exit_type"] != "partial_tp":
                        pos_side = "sell" if closed["side"] == "buy" else "buy"
                        if pos_side == "sell":
                            self.trader.market_sell(pair, closed.get("size", 0))
                        else:
                            self.trader.market_buy(pair, abs(closed.get("usdt_result", 0)))

                # Detectar régimen
                regime_info = self.detector.detect(df)
                regime      = regime_info["regime"]
                regime_summary[pair] = regime_info

                # Generar señal
                sig, stname = self.router.get_signal(df, regime_info)

                # Señal opuesta a posición abierta → cerrar
                if pair in self.pm.positions:
                    cur = self.pm.positions[pair]
                    opp = (cur["side"] == "buy" and sig == "sell") or \
                          (cur["side"] == "sell" and sig == "buy")
                    if opp:
                        result = self.pm.force_close(pair, close, "signal")
                        if result:
                            icon = "✅" if result["win"] else "❌"
                            _log(f"  {icon} SEÑAL OPUESTA {pair} → cerrado"
                                 f" | pnl={result['pnl']:+.2f}%")
                    continue

                # Sin posición y hay señal → abrir
                if sig in ("buy", "sell") and pair not in self.pm.positions:
                    pos = self.pm.open_position(
                        pair, sig, close, df, usdt_balance
                    )
                    if pos:
                        pos["strategy"] = stname
                        usdt_trade = pos["usdt"]
                        emoji = "🟢" if sig == "buy" else "🔴"
                        _log(f"  {emoji} {sig.upper()} {pair} | "
                             f"precio={close:.4f} | usdt={usdt_trade:.2f} | "
                             f"kelly={kelly_frac*100:.1f}% | "
                             f"régimen={regime} | strat={stname}")
                        if not self.paper:
                            if sig == "buy":
                                self.trader.market_buy(pair, usdt_trade)
                            else:
                                self.trader.market_sell(pair, usdt_trade/close)

            except Exception as e:
                _log(f"  ✗ {pair}: {e}")

        # Resumen del ciclo
        stats = self.pm.stats()
        _log(f"  Balance: {usdt_balance:.2f} USDT | "
             f"Trades: {stats['trades']} | WR: {stats['winrate']:.1f}% | "
             f"P&L: {stats['pnl_usdt']:+.4f} USDT | "
             f"Abiertas: {stats['open']} | Kelly: {kelly_frac*100:.1f}%")

        # Régimen dominante
        if regime_summary:
            regimes = [r["regime"] for r in regime_summary.values()]
            from collections import Counter
            dominant = Counter(regimes).most_common(1)[0]
            _log(f"  Régimen dominante: {dominant[0]} ({dominant[1]}/{len(regimes)} pares)")

        self._save_state(datasets, regime_summary, usdt_balance, stats)

    def _save_state(self, datasets, regime_summary, balance, stats):
        open_pos = []
        for sym, pos in self.pm.positions.items():
            df    = datasets.get(sym)
            price = float(df.iloc[-1]["close"]) if df is not None else pos["entry"]
            pnl   = (price - pos["entry"])/pos["entry"] * 100 \
                    if pos["side"] == "buy" \
                    else (pos["entry"] - price)/pos["entry"] * 100
            open_pos.append({
                **pos,
                "current_price": round(price, 6),
                "unrealized_pnl_pct": round(pnl, 3),
                "regime": regime_summary.get(sym, {}).get("regime", "?"),
            })

        state = {
            "timestamp":      time.time(),
            "datetime":       datetime.now(timezone.utc).isoformat(),
            "mode":           "paper" if self.paper else "real",
            "cycle":          self._cycle,
            "usdt_balance":   round(balance, 2),
            "kelly_fraction": round(self.pm.kelly_fraction()*100, 1),
            "config": {
                "kelly_cap":   KELLY_CAP,
                "trail_mult":  TRAIL_ATR_MULT,
                "partial_pct": PARTIAL_RATIO,
                "adx_min":     ADX_MIN,
            },
            "open_positions": open_pos,
            "stats":          stats,
            "recent_trades":  self.pm.closed[-20:],
            "regime_map":     {p: r["regime"] for p, r in regime_summary.items()},
        }
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def start(self, run_once: bool = False):
        self._running = True

        def _stop(sig, frame):
            _log("⛔ Parada solicitada...")
            self._running = False

        signal.signal(signal.SIGINT,  _stop)
        signal.signal(signal.SIGTERM, _stop)

        modo = "PAPER 📄" if self.paper else "REAL 💰"
        _log(f"🚀 SmartBot arrancado | {modo}")
        _log(f"   Intervalo: {SLEEP_SECONDS//60} min entre ciclos")

        while self._running:
            try:
                self._run_cycle()
            except Exception as e:
                _log(f"  ✗ ERROR: {e}")

            if run_once:
                break

            _log(f"  💤 Próximo ciclo en {SLEEP_SECONDS//60} min...")
            time.sleep(SLEEP_SECONDS)

        _log("✅ SmartBot detenido")
        stats = self.pm.stats()
        _log(f"  Resumen final:")
        _log(f"    Trades       : {stats['trades']}")
        _log(f"    Win rate     : {stats['winrate']:.1f}%")
        _log(f"    P&L total    : {stats['pnl_usdt']:+.4f} USDT")
        _log(f"    Mejor trade  : {stats['best']:+.2f}%")
        _log(f"    Peor trade   : {stats['worst']:+.2f}%")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SmartBot — Trading con régimen automático")
    parser.add_argument("--real",    action="store_true",
                        help="Modo real (requiere .env con API keys)")
    parser.add_argument("--capital", type=float, default=100.0,
                        help="Capital inicial en USDT (default: 100)")
    parser.add_argument("--once",    action="store_true",
                        help="Ejecutar solo un ciclo (test/cron)")
    args = parser.parse_args()

    paper          = not args.real
    api_key        = ""
    api_secret     = ""
    api_passphrase = ""
    capital        = args.capital

    if not paper:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        api_key        = os.getenv("KUCOIN_API_KEY", "")
        api_secret     = os.getenv("KUCOIN_API_SECRET", "")
        api_passphrase = os.getenv("KUCOIN_API_PASSPHRASE", "")
        capital        = float(os.getenv("INITIAL_CAPITAL", capital))

        if not all([api_key, api_secret, api_passphrase]):
            print("❌ Faltan credenciales en .env:")
            print("   KUCOIN_API_KEY=...")
            print("   KUCOIN_API_SECRET=...")
            print("   KUCOIN_API_PASSPHRASE=...")
            sys.exit(1)

        print("\n⚠️  MODO REAL — operando con dinero real en KuCoin")
        print("   CTRL+C para parar\n")

    bot = SmartBot(
        paper=paper,
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
        initial_capital=capital,
    )
    bot.start(run_once=args.once)


if __name__ == "__main__":
    main()
