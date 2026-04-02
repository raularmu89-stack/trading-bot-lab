"""
live_trader.py

Bucle principal de trading en vivo — estrategia MTF-SMC + RM1×2.

Flujo cada hora:
  1. Descarga últimas ~200 velas 1H por par
  2. Genera señal MTF-SMC (con filtro 4H)
  3. Si señal BUY/SELL y no hay posición abierta → ejecuta
  4. Si señal opuesta a posición abierta → cierra y abre nueva
  5. Actualiza PositionManager con la última vela (SL/TP)
  6. Guarda estado para el dashboard

MODO PAPER (default — sin riesgo real):
    python trading/live_trader.py

MODO REAL (requiere .env con API keys):
    python trading/live_trader.py --real

.env formato:
    KUCOIN_API_KEY=tu_key
    KUCOIN_API_SECRET=tu_secret
    KUCOIN_API_PASSPHRASE=tu_passphrase
    INITIAL_CAPITAL=1000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import json
import signal
import threading
from datetime import datetime, timezone

from data.kucoin_client     import KuCoinClient
from strategies.mtf_smc     import MultiTFSMC
from strategies.risk_manager import RiskManager
from strategies.kelly_sizer  import KellySizer
from trading.kucoin_trader   import KuCoinTrader
from trading.position_manager import PositionManager


# ── Configuración de la estrategia ganadora ───────────────────────────────────

STRATEGY   = MultiTFSMC(swing_window=5, trend_ema=50)
RISK_MGR   = RiskManager(method="atr", atr_multiplier=1.0, rr_ratio=2.0)
SIZER      = KellySizer(variant="full_kelly", min_trades=20,
                        max_fraction=0.40, min_fraction=0.02)

PAIRS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "BNB-USDT",
    "ADA-USDT", "AVAX-USDT", "DOGE-USDT", "LTC-USDT", "DOT-USDT",
]

CANDLES_NEEDED = 300    # velas 1H para señal fiable (incluye 4H resample)
SLEEP_SECONDS  = 3600   # revisar cada hora
DATA_FILE      = "data/live_state.json"
LOG_FILE       = "data/live_log.txt"


# ── Utilidades ────────────────────────────────────────────────────────────────

def _log(msg: str):
    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _kelly_fraction(trade_history: list) -> float:
    return SIZER.position_fraction(trade_history)


# ── Motor de trading ──────────────────────────────────────────────────────────

class LiveTrader:
    """
    Bucle de trading en vivo para MTF-SMC.

    Parámetros
    ----------
    paper         : True = paper trading, False = real
    api_key/secret/passphrase : credenciales KuCoin (solo real mode)
    initial_capital : capital inicial USDT para paper
    """

    def __init__(self, paper: bool = True,
                 api_key="", api_secret="", api_passphrase="",
                 initial_capital: float = 1000.0):

        self.paper   = paper
        self.client  = KuCoinClient()
        self.trader  = KuCoinTrader(
            paper=paper,
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
            initial_balance=initial_capital,
        )
        self.pm      = PositionManager(
            risk_manager=RISK_MGR,
            state_file="data/positions.json",
        )
        self.pm.load_state()

        self.trade_history = []   # historial para Kelly
        self._running      = False
        self._cycle        = 0

        _log(f"LiveTrader iniciado | paper={paper} | capital={initial_capital} USDT")
        _log(f"Estrategia: MTF-SMC(sw=5,ema4h=50) + RM ATR×1.0 RR=2.0")
        _log(f"Pares: {', '.join(PAIRS)}")

    # ── Un ciclo completo ─────────────────────────────────────────────────

    def _run_cycle(self):
        self._cycle += 1
        _log(f"── Ciclo #{self._cycle} ──────────────────────")

        # 1. Obtener datos
        datasets = {}
        for pair in PAIRS:
            try:
                df = self.client.get_ohlcv(pair, interval="1hour",
                                           limit=CANDLES_NEEDED)
                if df is not None and len(df) >= 100:
                    datasets[pair] = df
            except Exception as e:
                _log(f"  ⚠ {pair} datos error: {e}")
            time.sleep(0.15)

        _log(f"  Datos: {len(datasets)}/{len(PAIRS)} pares OK")

        # 2. Actualizar SL/TP con la última vela
        latest_candles = {}
        for pair, df in datasets.items():
            last = df.iloc[-1]
            latest_candles[pair] = {
                "high":  float(last["high"]),
                "low":   float(last["low"]),
                "close": float(last["close"]),
            }

        closed = self.pm.update_all(latest_candles)
        for c in closed:
            self.trade_history.append({
                "pnl": c["pnl_pct"] / 100,
                "win": c["pnl_net_usdt"] > 0,
            })
            _log(f"  {'✅' if c['pnl_net_usdt'] > 0 else '❌'} CLOSED "
                 f"{c['symbol']} via {c['status'].upper()} | "
                 f"pnl={c['pnl_net_usdt']:+.4f} USDT ({c['pnl_pct']:+.2f}%)")

        # 3. Generar señales y ejecutar
        kelly_frac = _kelly_fraction(self.trade_history)
        usdt_total = self.trader.get_balance("USDT")

        for pair, df in datasets.items():
            try:
                sigs = STRATEGY.generate_signals_batch(df)
                sig  = sigs[-1] if sigs else "hold"

                if sig == "hold":
                    continue

                current_pos = self.pm.open_positions.get(pair)

                # Señal opuesta → cerrar posición existente
                if current_pos:
                    if (sig == "buy"  and current_pos.side == "sell") or \
                       (sig == "sell" and current_pos.side == "buy"):
                        price  = latest_candles[pair]["close"]
                        result = self.pm.force_close(pair, price, "signal")
                        if result:
                            # Ejecutar orden de cierre
                            if current_pos.side == "buy":
                                self.trader.market_sell(pair, current_pos.size)
                            else:
                                self.trader.market_buy(pair,
                                    current_pos.usdt_risked)
                            self.trade_history.append({
                                "pnl": result["pnl_pct"] / 100,
                                "win": result["pnl_net_usdt"] > 0,
                            })
                    continue   # no abrir nueva en el mismo ciclo

                # Sin posición → abrir nueva
                if pair not in self.pm.open_positions:
                    usdt_to_risk = usdt_total * kelly_frac
                    if usdt_to_risk < 5.0:
                        _log(f"  ⚠ {pair}: capital insuficiente ({usdt_to_risk:.2f} USDT)")
                        continue

                    price = latest_candles[pair]["close"]
                    size  = usdt_to_risk / price

                    _log(f"  📶 SEÑAL {sig.upper():4s} {pair}  "
                         f"price={price:.4f}  usdt={usdt_to_risk:.2f}  "
                         f"kelly={kelly_frac*100:.1f}%")

                    # Ejecutar orden
                    if sig == "buy":
                        order = self.trader.market_buy(pair, usdt_to_risk)
                    else:
                        order = self.trader.market_sell(pair,
                            usdt_to_risk / price)

                    order_id = order.get("order_id", "")
                    actual_size  = order.get("size", size)
                    actual_price = order.get("price", price)

                    self.pm.open(
                        symbol=pair, side=sig,
                        entry_price=actual_price,
                        size=actual_size,
                        data=df,
                        usdt_risked=usdt_to_risk,
                        order_id=order_id,
                    )

            except Exception as e:
                _log(f"  ✗ {pair} error: {e}")

        # 4. Guardar estado del dashboard
        self._save_live_state(datasets)

        # 5. Resumen del ciclo
        stats = self.pm.stats()
        bal   = self.trader.get_balance("USDT")
        _log(f"  Balance USDT: {bal:.2f}  |  "
             f"Trades: {stats['total_trades']}  |  "
             f"WR: {stats['winrate']:.1f}%  |  "
             f"P&L neto: {stats['total_pnl_usdt']:>+.4f} USDT  |  "
             f"Posiciones abiertas: {stats['open_positions']}")

    def _save_live_state(self, datasets: dict):
        """Guarda estado completo para el dashboard Flask."""
        stats = self.pm.stats()
        bal   = self.trader.get_balance("USDT")

        open_pos = []
        for sym, pos in self.pm.open_positions.items():
            price = datasets.get(sym, None)
            if price is not None:
                cur_price = float(price.iloc[-1]["close"])
                upnl      = pos.unrealized_pnl(cur_price)
            else:
                cur_price = pos.entry_price
                upnl      = 0.0
            d = pos.to_dict()
            d["current_price"] = round(cur_price, 6)
            d["unrealized_pnl_usdt"] = round(upnl, 4)
            open_pos.append(d)

        state = {
            "timestamp":      time.time(),
            "datetime":       datetime.now(timezone.utc).isoformat(),
            "mode":           "paper" if self.paper else "real",
            "strategy":       "MTF-SMC(sw5,ema4h50)+RM ATR×1 RR=2",
            "cycle":          self._cycle,
            "usdt_balance":   round(bal, 2),
            "open_positions": open_pos,
            "stats":          stats,
            "recent_trades":  self.pm.closed_trades[-20:],
            "kelly_fraction": round(_kelly_fraction(self.trade_history) * 100, 1),
        }
        try:
            with open(DATA_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    # ── Loop principal ────────────────────────────────────────────────────

    def start(self, run_once: bool = False):
        """
        Inicia el bucle de trading.

        run_once=True : ejecuta un solo ciclo (útil para tests/cron)
        run_once=False: bucle infinito cada SLEEP_SECONDS
        """
        self._running = True

        def _handle_stop(sig, frame):
            _log("⛔ Señal de parada recibida. Cerrando...")
            self._running = False

        signal.signal(signal.SIGINT,  _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

        _log("🚀 Trading loop iniciado")
        _log(f"   Intervalo: cada {SLEEP_SECONDS//60} minutos")
        _log(f"   Modo: {'PAPER 📄' if self.paper else '⚠️  REAL 💰'}")

        while self._running:
            try:
                self._run_cycle()
            except Exception as e:
                _log(f"  ✗ ERROR en ciclo: {e}")

            if run_once:
                break

            next_run = datetime.now(timezone.utc)
            _log(f"  💤 Próximo ciclo en {SLEEP_SECONDS//60} min...")
            time.sleep(SLEEP_SECONDS)

        _log("✅ Trading loop detenido.")
        self.pm.print_stats()

        if self.paper:
            summary = self.trader.paper_summary()
            _log(f"\n  PAPER SUMMARY:")
            _log(f"    Balance USDT  : {summary['usdt_balance']:.2f}")
            _log(f"    Open pos value: {summary['open_positions_value']:.2f}")
            _log(f"    Total value   : {summary['total_value']:.2f}")
            _log(f"    Total trades  : {summary['total_trades']}")
            _log(f"    Total fees    : {summary['total_fees']:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live trader MTF-SMC")
    parser.add_argument("--real",    action="store_true",
                        help="Activar modo real (requiere .env)")
    parser.add_argument("--capital", type=float, default=1000.0,
                        help="Capital inicial USDT para paper (default: 1000)")
    parser.add_argument("--once",    action="store_true",
                        help="Ejecutar solo un ciclo y salir")
    args = parser.parse_args()

    paper = not args.real
    api_key = api_secret = api_passphrase = ""

    if not paper:
        # Cargar credenciales desde .env
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        api_key        = os.getenv("KUCOIN_API_KEY", "")
        api_secret     = os.getenv("KUCOIN_API_SECRET", "")
        api_passphrase = os.getenv("KUCOIN_API_PASSPHRASE", "")
        capital        = float(os.getenv("INITIAL_CAPITAL", args.capital))

        if not all([api_key, api_secret, api_passphrase]):
            print("❌ ERROR: Faltan credenciales KuCoin en .env")
            print("   Crea .env con:")
            print("   KUCOIN_API_KEY=...")
            print("   KUCOIN_API_SECRET=...")
            print("   KUCOIN_API_PASSPHRASE=...")
            sys.exit(1)

        print("\n⚠️  MODO REAL ACTIVADO — operando con dinero real")
        print("   Pulsa CTRL+C para parar en cualquier momento.\n")
    else:
        capital = args.capital

    bot = LiveTrader(
        paper=paper,
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
        initial_capital=capital,
    )
    bot.start(run_once=args.once)


if __name__ == "__main__":
    main()
