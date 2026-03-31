"""
paper_trader.py

Paper trading en tiempo real: simula operaciones sin dinero real.

Flujo cada ciclo:
  1. Descarga las ultimas N velas de Binance
  2. Genera señal con SMCStrategy
  3. Si hay señal de entrada y no hay posicion abierta → abre trade simulado
  4. Si hay posicion abierta → comprueba SL/TP y señal de salida
  5. Registra todo en logs/paper_trades.csv y muestra estado por consola

Uso:
    python -m bot.paper_trader
    python -m bot.paper_trader --symbol ETHUSDT --tf 1h --interval 60
"""

import argparse
import csv
import os
import time
from datetime import datetime

from bot.data_fetcher import fetch_klines
from strategies.smc_strategy import SMCStrategy
from strategies.risk_manager import RiskManager

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "paper_trades.csv")
LOG_FIELDS = [
    "timestamp", "symbol", "timeframe", "event",
    "price", "sl", "tp", "pnl_pct", "balance",
]

_SEPARATOR = "─" * 60


def _log_event(writer, symbol, tf, event, price, sl, tp, pnl_pct, balance):
    writer.writerow({
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol":    symbol,
        "timeframe": tf,
        "event":     event,
        "price":     round(price, 6),
        "sl":        round(sl, 6) if sl else "",
        "tp":        round(tp, 6) if tp else "",
        "pnl_pct":   round(pnl_pct * 100, 4) if pnl_pct is not None else "",
        "balance":   round(balance, 4),
    })


def _print_status(cycle, symbol, tf, price, signal, position, balance):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    pos_str = "---"
    if position:
        side = position["side"].upper()
        pnl = (price - position["entry"]) / position["entry"]
        if side == "SELL":
            pnl = -pnl
        pos_str = (f"{side} @ {position['entry']:.4f}  "
                   f"SL={position['sl']:.4f}  TP={position['tp']:.4f}  "
                   f"PnL={pnl*100:+.2f}%")
    print(f"[{ts}] #{cycle:3d}  {symbol}/{tf}  precio={price:.4f}  "
          f"señal={signal:<5}  pos={pos_str}  balance={balance:.4f}€")


def run_paper_trader(
    symbol="BTCUSDT",
    timeframe="1h",
    initial_balance=100.0,
    interval_sec=60,
    candles=200,
    swing_window=7,
    sl_pct=0.02,
    rr_ratio=2.0,
    risk_method="atr",
    max_hold=20,
    max_cycles=None,
):
    strategy = SMCStrategy(
        swing_window=swing_window,
        require_fvg=False,
        use_choch_filter=True,
    )
    risk_manager = RiskManager(sl_pct=sl_pct, rr_ratio=rr_ratio, method=risk_method)

    balance = initial_balance
    position = None
    cycle = 0
    total_trades = 0
    winning_trades = 0

    os.makedirs(os.path.dirname(os.path.abspath(LOG_PATH)), exist_ok=True)
    log_file = open(LOG_PATH, "a", newline="")
    writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    if os.path.getsize(LOG_PATH) == 0:
        writer.writeheader()

    print(_SEPARATOR)
    print(f"  PAPER TRADER  |  {symbol}  |  {timeframe}  |  Balance: {balance:.2f}€")
    print(f"  SL={sl_pct*100:.1f}%  RR={rr_ratio}:1  swing={swing_window}  "
          f"hold={max_hold}  metodo={risk_method}")
    print(_SEPARATOR)

    try:
        while max_cycles is None or cycle < max_cycles:
            cycle += 1

            # 1. Descargar datos
            data = fetch_klines(symbol=symbol, timeframe=timeframe, limit=candles)
            if data is None or data.empty:
                print(f"[ciclo {cycle}] Error al obtener datos, reintentando...")
                time.sleep(interval_sec)
                continue

            current_price = float(data["close"].iloc[-1])
            current_high  = float(data["high"].iloc[-1])
            current_low   = float(data["low"].iloc[-1])

            # 2. Comprobar SL/TP si hay posicion abierta
            if position is not None:
                entry = position["entry"]
                sl    = position["sl"]
                tp    = position["tp"]
                side  = position["side"]

                if side == "buy":
                    if current_low <= sl:
                        pnl = (sl - entry) / entry
                        balance *= (1 + pnl)
                        total_trades += 1
                        print(f"  *** STOP LOSS ({symbol})  PnL={pnl*100:+.2f}%  "
                              f"Balance={balance:.4f}€")
                        _log_event(writer, symbol, timeframe, "SL_HIT",
                                   sl, sl, tp, pnl, balance)
                        log_file.flush()
                        position = None
                        continue
                    elif current_high >= tp:
                        pnl = (tp - entry) / entry
                        balance *= (1 + pnl)
                        total_trades += 1
                        winning_trades += 1
                        print(f"  *** TAKE PROFIT ({symbol})  PnL={pnl*100:+.2f}%  "
                              f"Balance={balance:.4f}€")
                        _log_event(writer, symbol, timeframe, "TP_HIT",
                                   tp, sl, tp, pnl, balance)
                        log_file.flush()
                        position = None
                        continue
                else:  # sell
                    if current_high >= sl:
                        pnl = (entry - sl) / entry
                        balance *= (1 + pnl)
                        total_trades += 1
                        print(f"  *** STOP LOSS ({symbol})  PnL={pnl*100:+.2f}%  "
                              f"Balance={balance:.4f}€")
                        _log_event(writer, symbol, timeframe, "SL_HIT",
                                   sl, sl, tp, pnl, balance)
                        log_file.flush()
                        position = None
                        continue
                    elif current_low <= tp:
                        pnl = (entry - tp) / entry
                        balance *= (1 + pnl)
                        total_trades += 1
                        winning_trades += 1
                        print(f"  *** TAKE PROFIT ({symbol})  PnL={pnl*100:+.2f}%  "
                              f"Balance={balance:.4f}€")
                        _log_event(writer, symbol, timeframe, "TP_HIT",
                                   tp, sl, tp, pnl, balance)
                        log_file.flush()
                        position = None
                        continue

                # Comprobar max_hold
                if cycle - position["open_cycle"] >= max_hold:
                    pnl = (current_price - entry) / entry if side == "buy" \
                          else (entry - current_price) / entry
                    balance *= (1 + pnl)
                    total_trades += 1
                    if pnl > 0:
                        winning_trades += 1
                    print(f"  --- CIERRE MAX_HOLD ({symbol})  PnL={pnl*100:+.2f}%  "
                          f"Balance={balance:.4f}€")
                    _log_event(writer, symbol, timeframe, "MAX_HOLD",
                               current_price, sl, tp, pnl, balance)
                    log_file.flush()
                    position = None

            # 3. Generar señal
            signal_result = strategy.generate_signal(data)
            signal = signal_result["signal"]

            # 4. Abrir posicion si no hay ninguna
            if position is None and signal in ("buy", "sell"):
                sl_price, tp_price = risk_manager.calculate_levels(
                    data, current_price, signal
                )
                position = {
                    "side":       signal,
                    "entry":      current_price,
                    "sl":         sl_price,
                    "tp":         tp_price,
                    "open_cycle": cycle,
                }
                print(f"  >>> ENTRADA {signal.upper()} @ {current_price:.4f}  "
                      f"SL={sl_price:.4f}  TP={tp_price:.4f}  "
                      f"({signal_result.get('reason','')})")
                _log_event(writer, symbol, timeframe,
                           f"OPEN_{signal.upper()}",
                           current_price, sl_price, tp_price, None, balance)
                log_file.flush()

            _print_status(cycle, symbol, timeframe, current_price,
                          signal, position, balance)

            if total_trades > 0:
                wr = winning_trades / total_trades
                print(f"  Stats: {total_trades} trades  WR={wr:.1%}  "
                      f"Balance={balance:.4f}€  "
                      f"({(balance/initial_balance-1)*100:+.2f}%)")

            if max_cycles and cycle >= max_cycles:
                break

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print(f"\n{_SEPARATOR}")
        print("Paper trader detenido por el usuario.")
    finally:
        log_file.close()

    print(_SEPARATOR)
    print(f"Sesion finalizada: {cycle} ciclos  |  {total_trades} trades")
    if total_trades > 0:
        print(f"WR={winning_trades/total_trades:.1%}  "
              f"Balance final={balance:.4f}€  "
              f"({(balance/initial_balance-1)*100:+.2f}%)")
    print(f"Log guardado en: {os.path.abspath(LOG_PATH)}")


def main():
    parser = argparse.ArgumentParser(description="SMC Paper Trader")
    parser.add_argument("--symbol",   default="BTCUSDT")
    parser.add_argument("--tf",       default="1h",    help="Timeframe Binance")
    parser.add_argument("--balance",  default=100.0,   type=float)
    parser.add_argument("--interval", default=60,      type=int,
                        help="Segundos entre ciclos")
    parser.add_argument("--candles",  default=200,     type=int)
    parser.add_argument("--swing",    default=7,       type=int)
    parser.add_argument("--sl",       default=0.02,    type=float)
    parser.add_argument("--rr",       default=2.0,     type=float)
    parser.add_argument("--method",   default="atr",   choices=["fixed","atr"])
    parser.add_argument("--hold",     default=20,      type=int)
    parser.add_argument("--cycles",   default=None,    type=int,
                        help="Numero maximo de ciclos (None = infinito)")
    args = parser.parse_args()

    run_paper_trader(
        symbol=args.symbol,
        timeframe=args.tf,
        initial_balance=args.balance,
        interval_sec=args.interval,
        candles=args.candles,
        swing_window=args.swing,
        sl_pct=args.sl,
        rr_ratio=args.rr,
        risk_method=args.method,
        max_hold=args.hold,
        max_cycles=args.cycles,
    )


if __name__ == "__main__":
    main()
