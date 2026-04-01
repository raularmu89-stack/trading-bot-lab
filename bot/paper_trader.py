"""
paper_trader.py

Paper trading en tiempo real con todos los indicadores del Pine Script 'raul smc'.

Flujo cada ciclo:
  1. Descarga las últimas N velas de Binance (o mock si --mock / API bloqueada)
  2. Calcula estructura dual (Internal + Swing), EMA/RSI, P/D zone
  3. Genera señal con SMCStrategy (con todos los filtros activos)
  4. Gestiona posición abierta: SL/TP via high/low o max_hold
  5. Registra todo en logs/paper_trades.csv y muestra estado por consola
  6. Al finalizar imprime métricas de sesión (Sharpe, MaxDD, WR)

Uso:
    python -m bot.paper_trader                               # BTC 1h, 100€
    python -m bot.paper_trader --symbol ETHUSDT --tf 4h
    python -m bot.paper_trader --style swing --dual --ema --pd
    python -m bot.paper_trader --mock --cycles 50            # offline con datos sintéticos
"""

import argparse
import csv
import os
import time
from datetime import datetime

from bot.data_fetcher import fetch_klines
from strategies.smc_strategy import SMCStrategy
from strategies.risk_manager import RiskManager
from indicators.market_structure import detect_dual_structure
from indicators.ema_rsi import compute_ema_rsi, TRADING_STYLES
from indicators.premium_discount import detect_premium_discount
from backtests.metrics import compute_all

LOG_PATH   = os.path.join(os.path.dirname(__file__), "..", "logs", "paper_trades.csv")
LOG_FIELDS = [
    "timestamp", "symbol", "timeframe", "event",
    "price", "sl", "tp", "pnl_pct", "balance",
    "swing_trend", "internal_trend", "pd_zone", "ema_signal",
]

SEP  = "─" * 70
SEP2 = "═" * 70


# ─── Logging ─────────────────────────────────────────────────────────────────

def _log(writer, logfile, symbol, tf, event, price, sl, tp,
         pnl_pct, balance, ctx):
    writer.writerow({
        "timestamp":      datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol":         symbol,
        "timeframe":      tf,
        "event":          event,
        "price":          round(price, 6),
        "sl":             round(sl, 6)  if sl  else "",
        "tp":             round(tp, 6)  if tp  else "",
        "pnl_pct":        round(pnl_pct * 100, 4) if pnl_pct is not None else "",
        "balance":        round(balance, 4),
        "swing_trend":    ctx.get("swing_trend", ""),
        "internal_trend": ctx.get("internal_trend", ""),
        "pd_zone":        ctx.get("pd_zone", ""),
        "ema_signal":     ctx.get("ema_signal", ""),
    })
    logfile.flush()


# ─── Indicadores de contexto ─────────────────────────────────────────────────

def _get_context(data, style, swing_window, internal_window):
    """
    Calcula contexto de mercado extra para mostrar en consola y loguear.
    Devuelve dict con swing_trend, internal_trend, pd_zone, ema_signal,
    strong_high, strong_low, zone_pct.
    """
    ctx = {
        "swing_trend":    "?",
        "internal_trend": "?",
        "pd_zone":        "?",
        "ema_signal":     "?",
        "strong_high":    None,
        "strong_low":     None,
        "zone_pct":       None,
    }
    try:
        ds = detect_dual_structure(data, internal_window=internal_window,
                                   swing_window=swing_window)
        if ds:
            ctx["swing_trend"]    = ds["swing"]["trend"]    if ds["swing"]    else "?"
            ctx["internal_trend"] = ds["internal"]["trend"] if ds["internal"] else "?"
            ctx["strong_high"]    = ds["strong_high"]
            ctx["strong_low"]     = ds["strong_low"]
    except Exception:
        pass

    try:
        pd = detect_premium_discount(data, swing_window=swing_window)
        if pd:
            ctx["pd_zone"]   = pd["zone"]
            ctx["zone_pct"]  = pd["zone_pct"]
    except Exception:
        pass

    try:
        ema = compute_ema_rsi(data, style=style)
        ctx["ema_signal"] = ema["signal"]
    except Exception:
        pass

    return ctx


# ─── Consola ─────────────────────────────────────────────────────────────────

_ZONE_ICON = {"premium": "▲", "discount": "▼", "equilibrium": "◆", "?": "?"}
_TREND_ICON = {"bullish": "↑", "bearish": "↓", "neutral": "→", "?": "?"}


def _print_header(symbol, tf, balance, style, swing_window, sl_pct,
                  rr_ratio, risk_method, use_dual, use_ema, use_pd):
    print(SEP2)
    print(f"  PAPER TRADER  ·  {symbol}  ·  {timeframe_label(tf)}")
    print(f"  Balance inicial: {balance:.2f}€  |  Style: {style}")
    print(f"  swing={swing_window}  SL={sl_pct*100:.1f}%  RR={rr_ratio}  "
          f"método={risk_method}  dual={use_dual}  EMA={use_ema}  PD={use_pd}")
    print(SEP2)


def _print_cycle(cycle, symbol, tf, price, signal, position,
                 balance, ctx, total_trades, winning_trades):
    ts  = datetime.utcnow().strftime("%H:%M:%S")
    sw  = _TREND_ICON.get(ctx["swing_trend"], "?")
    it  = _TREND_ICON.get(ctx["internal_trend"], "?")
    pd  = _ZONE_ICON.get(ctx["pd_zone"], "?")
    pct = f"{ctx['zone_pct']*100:.0f}%" if ctx["zone_pct"] is not None else "?%"

    sig_icon = {"buy": "▶ BUY ", "sell": "◀ SELL", "hold": " hold "}.get(signal, signal)
    pos_str  = "sin posición"
    if position:
        side = position["side"].upper()
        pnl  = ((price - position["entry"]) / position["entry"]
                if side == "BUY"
                else (position["entry"] - price) / position["entry"])
        pos_str = (f"{side} @ {position['entry']:.4f}  "
                   f"SL={position['sl']:.4f}  TP={position['tp']:.4f}  "
                   f"PnL={pnl*100:+.2f}%")

    wr_str = ""
    if total_trades > 0:
        wr_str = f"  WR={winning_trades/total_trades:.0%} ({total_trades}T)"

    print(f"[{ts}] #{cycle:3d}  {symbol}/{tf}  {price:.4f}"
          f"  Sw:{sw} In:{it}  P/D:{pd}{pct}"
          f"  EMA:{ctx['ema_signal']:<5}"
          f"  [{sig_icon}]"
          f"  {pos_str}"
          f"  {balance:.2f}€{wr_str}")


def timeframe_label(tf):
    return {"1m": "1 min", "5m": "5 min", "15m": "15 min",
            "1h": "1 hora", "4h": "4 horas", "1d": "1 día"}.get(tf, tf)


# ─── Core ─────────────────────────────────────────────────────────────────────

def run_paper_trader(
    symbol="BTCUSDT",
    timeframe="1h",
    initial_balance=100.0,
    interval_sec=60,
    candles=300,
    swing_window=50,
    internal_window=5,
    sl_pct=0.02,
    rr_ratio=2.0,
    risk_method="atr",
    max_hold=20,
    trading_style="swing",
    use_dual=True,
    use_ema_filter=False,
    use_pd_filter=False,
    use_ob_filter=False,
    use_fvg_filter=False,
    max_cycles=None,
    use_mock=False,
):
    # ── Estrategia con todos los filtros ────────────────────────────────────
    strategy = SMCStrategy(
        swing_window=swing_window,
        internal_window=internal_window,
        use_dual_structure=use_dual,
        trading_style=trading_style,
        use_ema_filter=use_ema_filter,
        use_pd_filter=use_pd_filter,
        require_ob=use_ob_filter,
        require_fvg=use_fvg_filter,
        use_choch_filter=True,
        ob_mitigation="highlow",
    )
    risk_manager = RiskManager(sl_pct=sl_pct, rr_ratio=rr_ratio, method=risk_method)

    balance    = initial_balance
    position   = None
    cycle      = 0
    total_trades   = 0
    winning_trades = 0
    equity_curve   = [balance]
    session_trades = []   # para métricas al final

    os.makedirs(os.path.dirname(os.path.abspath(LOG_PATH)), exist_ok=True)
    logfile = open(LOG_PATH, "a", newline="")
    writer  = csv.DictWriter(logfile, fieldnames=LOG_FIELDS)
    if os.path.getsize(LOG_PATH) == 0:
        writer.writeheader()

    _print_header(symbol, timeframe, initial_balance, trading_style,
                  swing_window, sl_pct, rr_ratio, risk_method,
                  use_dual, use_ema_filter, use_pd_filter)

    # ── Fuente de datos ──────────────────────────────────────────────────────
    if use_mock:
        from bot.mock_data import generate_mock_klines
        print("  [MODO MOCK] Usando datos sintéticos (sin API)")
        _mock_all = generate_mock_klines(symbol, n_candles=candles + (max_cycles or 100),
                                          timeframe=timeframe, seed=42)

    def _get_data(cycle_idx):
        if use_mock:
            end = candles + cycle_idx - 1
            return _mock_all.iloc[cycle_idx - 1: end].reset_index(drop=True)
        try:
            return fetch_klines(symbol=symbol, timeframe=timeframe, limit=candles)
        except Exception:
            # Fallback silencioso a mock si Binance no responde
            from bot.mock_data import generate_mock_klines
            return generate_mock_klines(symbol, n_candles=candles, timeframe=timeframe,
                                         seed=cycle_idx)

    print(SEP)
    try:
        while max_cycles is None or cycle < max_cycles:
            cycle += 1

            # ── 1. Datos ────────────────────────────────────────────────────
            data = _get_data(cycle)
            if data is None or len(data) < 20:
                print(f"[ciclo {cycle}] Sin datos, reintentando...")
                time.sleep(interval_sec)
                continue

            price  = float(data["close"].iloc[-1])
            c_high = float(data["high"].iloc[-1])
            c_low  = float(data["low"].iloc[-1])

            # ── 2. Contexto de mercado ───────────────────────────────────────
            ctx = _get_context(data, trading_style, swing_window, internal_window)

            # ── 3. Gestionar posición abierta ────────────────────────────────
            if position is not None:
                entry = position["entry"]
                sl    = position["sl"]
                tp    = position["tp"]
                side  = position["side"]
                closed = False

                if side == "buy":
                    if c_low <= sl:
                        pnl = (sl - entry) / entry
                        _close_position("SL_HIT", pnl, price, sl, tp,
                                        position, writer, logfile, ctx,
                                        symbol, timeframe, balance)
                        balance *= (1 + pnl)
                        closed = True
                    elif c_high >= tp:
                        pnl = (tp - entry) / entry
                        _close_position("TP_HIT", pnl, price, sl, tp,
                                        position, writer, logfile, ctx,
                                        symbol, timeframe, balance)
                        balance *= (1 + pnl)
                        winning_trades += 1
                        closed = True
                else:  # sell
                    if c_high >= sl:
                        pnl = (entry - sl) / entry
                        _close_position("SL_HIT", pnl, price, sl, tp,
                                        position, writer, logfile, ctx,
                                        symbol, timeframe, balance)
                        balance *= (1 + pnl)
                        closed = True
                    elif c_low <= tp:
                        pnl = (entry - tp) / entry
                        _close_position("TP_HIT", pnl, price, sl, tp,
                                        position, writer, logfile, ctx,
                                        symbol, timeframe, balance)
                        balance *= (1 + pnl)
                        winning_trades += 1
                        closed = True

                if closed:
                    total_trades += 1
                    session_trades.append(pnl)
                    equity_curve.append(balance)
                    position = None
                elif cycle - position["open_cycle"] >= max_hold:
                    # Cierre por max_hold
                    pnl = ((price - entry) / entry if side == "buy"
                           else (entry - price) / entry)
                    _close_position("MAX_HOLD", pnl, price, sl, tp,
                                    position, writer, logfile, ctx,
                                    symbol, timeframe, balance)
                    balance *= (1 + pnl)
                    total_trades += 1
                    if pnl > 0:
                        winning_trades += 1
                    session_trades.append(pnl)
                    equity_curve.append(balance)
                    position = None

            # ── 4. Señal ────────────────────────────────────────────────────
            sig_result = strategy.generate_signal(data)
            signal     = sig_result["signal"]

            # ── 5. Abrir posición ────────────────────────────────────────────
            if position is None and signal in ("buy", "sell"):
                sl_p, tp_p = risk_manager.calculate_levels(data, price, signal)
                position = {
                    "side":       signal,
                    "entry":      price,
                    "sl":         sl_p,
                    "tp":         tp_p,
                    "open_cycle": cycle,
                }
                tag = f"OPEN_{signal.upper()}"
                reason = sig_result.get("reason", "")
                print(f"  {'▶▶▶' if signal=='buy' else '◀◀◀'} {tag} @ {price:.4f}"
                      f"  SL={sl_p:.4f}  TP={tp_p:.4f}  [{reason}]")
                _log(writer, logfile, symbol, timeframe, tag,
                     price, sl_p, tp_p, None, balance, ctx)

            # ── 6. Estado ────────────────────────────────────────────────────
            _print_cycle(cycle, symbol, timeframe, price, signal,
                         position, balance, ctx, total_trades, winning_trades)

            if max_cycles and cycle >= max_cycles:
                break

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print(f"\n{SEP}\n  Detenido por el usuario (Ctrl+C)")
    finally:
        # Cerrar posición abierta al final
        if position is not None:
            pnl = ((price - position["entry"]) / position["entry"]
                   if position["side"] == "buy"
                   else (position["entry"] - price) / position["entry"])
            balance *= (1 + pnl)
            session_trades.append(pnl)
            equity_curve.append(balance)
            total_trades += 1
        logfile.close()

    # ── Resumen de sesión ────────────────────────────────────────────────────
    _print_summary(symbol, timeframe, initial_balance, balance,
                   cycle, total_trades, winning_trades,
                   equity_curve, session_trades)


def _close_position(event, pnl, price, sl, tp, position, writer, logfile,
                    ctx, symbol, timeframe, balance):
    icon = "✓ TP" if event == "TP_HIT" else ("✗ SL" if event == "SL_HIT" else "─ MH")
    print(f"  {icon}  {event}  PnL={pnl*100:+.2f}%  "
          f"nuevo balance≈{balance*(1+pnl):.4f}€")
    _log(writer, logfile, symbol, timeframe, event,
         price, sl, tp, pnl, balance, ctx)


def _print_summary(symbol, tf, initial, final, cycles,
                   total_trades, winning_trades, equity_curve, trades):
    print(SEP2)
    print(f"  RESUMEN DE SESIÓN  ·  {symbol}/{tf}")
    print(SEP)
    print(f"  Ciclos:       {cycles}")
    print(f"  Trades:       {total_trades}")
    wr = winning_trades / total_trades if total_trades > 0 else 0
    print(f"  Win Rate:     {wr:.1%}")
    print(f"  Balance:      {initial:.2f}€  →  {final:.2f}€  "
          f"({(final/initial - 1)*100:+.2f}%)")

    if len(equity_curve) > 1:
        m = compute_all([e / initial for e in equity_curve])
        print(f"  Sharpe:       {m['sharpe']:.4f}")
        print(f"  Sortino:      {m['sortino']:.4f}")
        print(f"  Max Drawdown: {m['max_drawdown']:.2%}")
        print(f"  Calmar:       {m['calmar']:.4f}")
    print(SEP2)
    print(f"  Log: {os.path.abspath(LOG_PATH)}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SMC Paper Trader — Pine Script parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python -m bot.paper_trader                              # BTC/1h por defecto
  python -m bot.paper_trader --symbol ETHUSDT --tf 4h
  python -m bot.paper_trader --style swing --dual         # Pine Swing
  python -m bot.paper_trader --style swing --dual --ema --pd --ob  # Pine Full
  python -m bot.paper_trader --mock --cycles 30           # offline sin API
        """
    )
    parser.add_argument("--symbol",   default="BTCUSDT",  help="Par de trading")
    parser.add_argument("--tf",       default="1h",       help="Timeframe")
    parser.add_argument("--balance",  default=100.0,      type=float, help="Balance inicial €")
    parser.add_argument("--interval", default=60,         type=int,   help="Segundos entre ciclos")
    parser.add_argument("--candles",  default=300,        type=int,   help="Velas históricas")
    parser.add_argument("--swing",    default=50,         type=int,   help="swing_window (Pine: 50)")
    parser.add_argument("--internal", default=5,          type=int,   help="internal_window (Pine: 5)")
    parser.add_argument("--sl",       default=0.02,       type=float, help="SL% (fixed)")
    parser.add_argument("--rr",       default=2.0,        type=float, help="Risk/Reward ratio")
    parser.add_argument("--method",   default="atr",      choices=["fixed", "atr"])
    parser.add_argument("--hold",     default=20,         type=int,   help="Max velas en posición")
    parser.add_argument("--style",    default="swing",    choices=list(TRADING_STYLES.keys()),
                        help="Preset EMA/RSI: scalping | intraday | swing")
    parser.add_argument("--dual",     action="store_true", default=True,
                        help="Usar estructura dual Internal + Swing")
    parser.add_argument("--no-dual",  action="store_false", dest="dual")
    parser.add_argument("--ema",      action="store_true", default=False,
                        help="Activar filtro EMA trend")
    parser.add_argument("--pd",       action="store_true", default=False,
                        help="Activar filtro Premium/Discount")
    parser.add_argument("--ob",       action="store_true", default=False,
                        help="Activar filtro Order Block")
    parser.add_argument("--fvg",      action="store_true", default=False,
                        help="Activar filtro FVG")
    parser.add_argument("--mock",     action="store_true", default=False,
                        help="Usar datos sintéticos en lugar de Binance API")
    parser.add_argument("--cycles",   default=None,       type=int,
                        help="Número máximo de ciclos (None = infinito)")
    args = parser.parse_args()

    run_paper_trader(
        symbol=args.symbol,
        timeframe=args.tf,
        initial_balance=args.balance,
        interval_sec=args.interval,
        candles=args.candles,
        swing_window=args.swing,
        internal_window=args.internal,
        sl_pct=args.sl,
        rr_ratio=args.rr,
        risk_method=args.method,
        max_hold=args.hold,
        trading_style=args.style,
        use_dual=args.dual,
        use_ema_filter=args.ema,
        use_pd_filter=args.pd,
        use_ob_filter=args.ob,
        use_fvg_filter=args.fvg,
        max_cycles=args.cycles,
        use_mock=args.mock,
    )


if __name__ == "__main__":
    main()
