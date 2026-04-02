"""
position_manager.py

Gestión de posiciones abiertas con SL/TP automático basado en ATR.

Responsabilidades:
  - Registrar entradas (buy/sell)
  - Calcular SL y TP por ATR en el momento de entrada
  - Monitorizar cada vela si se ha tocado SL o TP
  - Registrar el P&L realizado con fees incluidas
  - Exportar estado a JSON para el dashboard

Uso:
    from trading.position_manager import PositionManager
    from strategies.risk_manager import RiskManager

    pm = PositionManager(risk_manager=RiskManager(method="atr",
                         atr_multiplier=1.0, rr_ratio=2.0))

    pm.open("BTC-USDT", "buy", entry_price=95000, size=0.001,
            data=df, usdt_risked=50)

    # Cada nueva vela:
    closed = pm.update_all(latest_prices)
    for c in closed:
        print(c)
"""

import time
import json
import math
import numpy as np
import pandas as pd
from typing import Optional
from strategies.risk_manager import RiskManager


FEE_TAKER = 0.001   # 0.1% KuCoin taker


def _calc_atr(data: pd.DataFrame, period: int = 14) -> float:
    h = data["high"].values
    l = data["low"].values
    c = data["close"].values
    if len(c) < 2:
        return float(h[-1] - l[-1]) if len(h) else 0.0
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-period:])) if len(tr) >= period else float(np.mean(tr))


class Position:
    """Representa una posición abierta."""

    def __init__(self, symbol: str, side: str, entry_price: float,
                 size: float, sl: float, tp: float,
                 usdt_risked: float, order_id: str = ""):
        self.symbol       = symbol
        self.side         = side          # "buy" o "sell"
        self.entry_price  = entry_price
        self.size         = size          # unidades de la moneda base
        self.sl           = sl
        self.tp           = tp
        self.usdt_risked  = usdt_risked   # USDT comprometidos
        self.order_id     = order_id
        self.open_time    = time.time()
        self.status       = "open"        # "open" | "sl" | "tp" | "signal" | "timeout"
        self.close_price  = None
        self.close_time   = None
        self.pnl_pct      = None          # % sin fees
        self.pnl_net_usdt = None          # USDT neto post-fees

    def value_at(self, price: float) -> float:
        """Valor actual de la posición en USDT."""
        return self.size * price

    def unrealized_pnl(self, price: float) -> float:
        """P&L no realizado en USDT."""
        if self.side == "buy":
            raw = (price - self.entry_price) / self.entry_price
        else:
            raw = (self.entry_price - price) / self.entry_price
        return raw * self.usdt_risked

    def close(self, price: float, reason: str) -> dict:
        """Cierra la posición y calcula P&L."""
        self.close_price = price
        self.close_time  = time.time()
        self.status      = reason

        if self.side == "buy":
            raw_pct = (price - self.entry_price) / self.entry_price
        else:
            raw_pct = (self.entry_price - price) / self.entry_price

        # Fees: 0.1% entrada + 0.1% salida sobre el valor total
        entry_fee = self.usdt_risked * FEE_TAKER
        exit_fee  = self.usdt_risked * (1 + raw_pct) * FEE_TAKER
        total_fee = entry_fee + exit_fee

        self.pnl_pct      = raw_pct * 100
        self.pnl_net_usdt = self.usdt_risked * raw_pct - total_fee

        return self.to_dict()

    def to_dict(self) -> dict:
        age_h = (time.time() - self.open_time) / 3600
        return {
            "symbol":       self.symbol,
            "side":         self.side,
            "entry_price":  round(self.entry_price, 6),
            "close_price":  round(self.close_price, 6) if self.close_price else None,
            "size":         round(self.size, 8),
            "sl":           round(self.sl, 6),
            "tp":           round(self.tp, 6),
            "usdt_risked":  round(self.usdt_risked, 4),
            "status":       self.status,
            "pnl_pct":      round(self.pnl_pct, 3) if self.pnl_pct is not None else None,
            "pnl_net_usdt": round(self.pnl_net_usdt, 4) if self.pnl_net_usdt is not None else None,
            "age_hours":    round(age_h, 2),
            "order_id":     self.order_id,
            "open_time":    self.open_time,
            "close_time":   self.close_time,
        }

    def __repr__(self):
        upnl = ""
        return (f"Position({self.symbol} {self.side.upper()} "
                f"@ {self.entry_price:.4f} | "
                f"SL={self.sl:.4f} TP={self.tp:.4f} | "
                f"status={self.status})")


class PositionManager:
    """
    Gestiona todas las posiciones abiertas y el historial.

    Parámetros
    ----------
    risk_manager   : RiskManager para calcular SL/TP automáticamente
    max_positions  : máximo de posiciones simultáneas por par
    state_file     : ruta donde persistir el estado (JSON)
    """

    def __init__(
        self,
        risk_manager: Optional[RiskManager] = None,
        max_positions: int = 1,
        state_file: str = "data/positions.json",
    ):
        self.rm            = risk_manager or RiskManager(
            method="atr", atr_multiplier=1.0, rr_ratio=2.0
        )
        self.max_positions = max_positions
        self.state_file    = state_file

        self.open_positions: dict[str, Position] = {}   # symbol → Position
        self.closed_trades:  list[dict]           = []
        self._total_fees     = 0.0
        self._total_pnl_usdt = 0.0

    # ── Abrir posición ─────────────────────────────────────────────────────

    def open(self, symbol: str, side: str, entry_price: float,
             size: float, data: pd.DataFrame, usdt_risked: float,
             order_id: str = "") -> Optional[Position]:
        """
        Registra una nueva posición con SL/TP calculados por ATR.

        Retorna None si ya hay max_positions para este símbolo.
        """
        if symbol in self.open_positions:
            print(f"  [PM] Ya hay posición abierta en {symbol}, ignorando.")
            return None

        sl, tp = self.rm.calculate_levels(data, entry_price, side)

        pos = Position(
            symbol=symbol, side=side, entry_price=entry_price,
            size=size, sl=sl, tp=tp,
            usdt_risked=usdt_risked, order_id=order_id,
        )
        self.open_positions[symbol] = pos

        print(f"  [PM] OPEN  {symbol} {side.upper():4s}  "
              f"entry={entry_price:.4f}  SL={sl:.4f}  TP={tp:.4f}  "
              f"usdt={usdt_risked:.2f}")
        self._save_state()
        return pos

    # ── Actualizar posiciones con nueva vela ───────────────────────────────

    def update_all(self, latest_candles: dict) -> list[dict]:
        """
        Comprueba SL/TP para cada posición con los últimos precios.

        latest_candles : {symbol: {"high": float, "low": float, "close": float}}
        Retorna lista de posiciones cerradas en este tick.
        """
        closed = []
        for symbol in list(self.open_positions.keys()):
            pos   = self.open_positions[symbol]
            candle = latest_candles.get(symbol)
            if not candle:
                continue

            high  = candle.get("high",  candle.get("close", 0))
            low   = candle.get("low",   candle.get("close", 0))
            close = candle.get("close", 0)

            reason = None
            price  = close

            if pos.side == "buy":
                if low <= pos.sl:
                    reason, price = "sl", pos.sl
                elif high >= pos.tp:
                    reason, price = "tp", pos.tp
            else:
                if high >= pos.sl:
                    reason, price = "sl", pos.sl
                elif low <= pos.tp:
                    reason, price = "tp", pos.tp

            if reason:
                result = pos.close(price, reason)
                self._total_pnl_usdt += result.get("pnl_net_usdt", 0)
                self.closed_trades.append(result)
                del self.open_positions[symbol]
                closed.append(result)

                sign = "✅" if result["pnl_net_usdt"] > 0 else "❌"
                print(f"  [PM] CLOSE {sign} {symbol} via {reason.upper():6s}  "
                      f"pnl={result['pnl_net_usdt']:>+.4f} USDT  "
                      f"({result['pnl_pct']:>+.2f}%)")

        if closed:
            self._save_state()
        return closed

    def force_close(self, symbol: str, price: float, reason: str = "signal") -> Optional[dict]:
        """Cierra manualmente una posición (p.ej. señal opuesta)."""
        pos = self.open_positions.get(symbol)
        if not pos:
            return None
        result = pos.close(price, reason)
        self._total_pnl_usdt += result.get("pnl_net_usdt", 0)
        self.closed_trades.append(result)
        del self.open_positions[symbol]
        self._save_state()
        return result

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Estadísticas del session completo."""
        if not self.closed_trades:
            return {
                "total_trades": 0, "wins": 0, "losses": 0,
                "winrate": 0.0, "total_pnl_usdt": 0.0,
                "avg_pnl_usdt": 0.0, "profit_factor": 0.0,
                "open_positions": len(self.open_positions),
                "open_symbols": list(self.open_positions.keys()),
                "tp_hits": 0, "sl_hits": 0,
            }

        wins   = [t for t in self.closed_trades if t["pnl_net_usdt"] > 0]
        losses = [t for t in self.closed_trades if t["pnl_net_usdt"] <= 0]
        gross_profit = sum(t["pnl_net_usdt"] for t in wins)
        gross_loss   = abs(sum(t["pnl_net_usdt"] for t in losses))

        return {
            "total_trades":   len(self.closed_trades),
            "wins":           len(wins),
            "losses":         len(losses),
            "winrate":        round(len(wins) / len(self.closed_trades) * 100, 1),
            "total_pnl_usdt": round(self._total_pnl_usdt, 4),
            "avg_pnl_usdt":   round(self._total_pnl_usdt / len(self.closed_trades), 4),
            "profit_factor":  round(gross_profit / gross_loss, 3) if gross_loss > 0 else float("inf"),
            "open_positions": len(self.open_positions),
            "open_symbols":   list(self.open_positions.keys()),
            "tp_hits":        sum(1 for t in self.closed_trades if t.get("status") == "tp"),
            "sl_hits":        sum(1 for t in self.closed_trades if t.get("status") == "sl"),
        }

    def print_stats(self):
        s = self.stats()
        print(f"\n  {'─'*55}")
        print(f"  POSITION MANAGER — Stats de sesión")
        print(f"  {'─'*55}")
        print(f"  Trades cerrados : {s['total_trades']}")
        print(f"  Wins / Losses   : {s['wins']} / {s['losses']}  ({s['winrate']:.1f}% WR)")
        print(f"  P&L neto        : {s['total_pnl_usdt']:>+.4f} USDT")
        print(f"  Avg por trade   : {s['avg_pnl_usdt']:>+.4f} USDT")
        print(f"  Profit Factor   : {s['profit_factor']:.3f}")
        print(f"  TP hits / SL hits: {s['tp_hits']} / {s['sl_hits']}")
        print(f"  Posiciones abiertas: {s['open_positions']}")
        print(f"  {'─'*55}\n")

    # ── Persistencia ──────────────────────────────────────────────────────

    def _save_state(self):
        """Guarda estado en JSON para el dashboard."""
        state = {
            "timestamp":      time.time(),
            "open_positions": [p.to_dict() for p in self.open_positions.values()],
            "closed_trades":  self.closed_trades[-50:],   # últimos 50
            "stats":          self.stats(),
        }
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def load_state(self):
        """Carga estado previo (para continuar tras reinicio)."""
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            self.closed_trades  = state.get("closed_trades", [])
            self._total_pnl_usdt = sum(
                t.get("pnl_net_usdt", 0) for t in self.closed_trades
            )
            print(f"  [PM] Estado cargado: {len(self.closed_trades)} trades históricos")
        except FileNotFoundError:
            pass
