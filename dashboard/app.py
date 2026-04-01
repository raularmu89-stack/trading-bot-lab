"""
dashboard/app.py

Dashboard web para el trading bot — solo visualización, sin operar.

Características:
  - Gráfico de velas OHLCV en tiempo real con indicadores
  - Régimen de mercado actual detectado por RegimeDetector
  - Señal actual de cada estrategia disponible
  - Equity curve del último backtest
  - Hiperparámetros óptimos
  - Soporte para datos reales (KuCoin) o simulados

Uso:
    python dashboard/app.py
    # Abre http://localhost:5000

Variables de entorno:
    DASHBOARD_USE_REAL_DATA=1   → usar KuCoin API real
    DASHBOARD_SYMBOL=BTC-USDT   → par por defecto
    DASHBOARD_INTERVAL=15min    → timeframe
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from flask import Flask, render_template, jsonify, request

from indicators.regime_detector import RegimeDetector
from strategies.smc_strategy        import SMCStrategy
from strategies.scenario_router     import ScenarioRouter
from strategies.macd_divergence     import MACDDivergenceStrategy
from strategies.bollinger_squeeze   import BollingerSqueezeStrategy
from strategies.momentum_burst      import MomentumBurstStrategy


# ── Configuración ─────────────────────────────────────────────────────────────

USE_REAL_DATA = os.getenv("DASHBOARD_USE_REAL_DATA", "0") == "1"
DEFAULT_SYMBOL   = os.getenv("DASHBOARD_SYMBOL",   "BTC-USDT")
DEFAULT_INTERVAL = os.getenv("DASHBOARD_INTERVAL", "15min")
DEFAULT_LIMIT    = int(os.getenv("DASHBOARD_LIMIT", "200"))

SYMBOLS_AVAILABLE = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT",
    "BNB-USDT", "LINK-USDT", "XRP-USDT",
]

STRATEGIES_AVAILABLE = {
    "smc":              SMCStrategy(swing_window=5),
    "scenario_router":  ScenarioRouter(verbose=False),
    "macd_divergence":  MACDDivergenceStrategy(),
    "bollinger_squeeze":BollingerSqueezeStrategy(),
    "momentum_burst":   MomentumBurstStrategy(),
}

REGIME_DETECTOR = RegimeDetector()

app = Flask(__name__)

# ── Cache de datos ────────────────────────────────────────────────────────────

_cache: dict = {}
_cache_lock = threading.Lock()
CACHE_TTL = 60  # segundos


def _cache_key(symbol, interval, limit):
    return f"{symbol}:{interval}:{limit}"


def _get_data(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    key = _cache_key(symbol, interval, limit)
    now = time.time()

    with _cache_lock:
        if key in _cache and now - _cache[key]["ts"] < CACHE_TTL:
            return _cache[key]["df"]

    if USE_REAL_DATA:
        from data.kucoin_client import KuCoinClient
        client = KuCoinClient(verbose=False)
        df = client.get_ohlcv(symbol, interval=interval, limit=limit)
    else:
        df = _gen_demo_data(symbol, limit)

    with _cache_lock:
        _cache[key] = {"df": df, "ts": now}

    return df


def _gen_demo_data(symbol: str, n: int = 200) -> pd.DataFrame:
    """Genera datos OHLCV demo para modo offline."""
    seeds = {"BTC-USDT": 0, "ETH-USDT": 1, "SOL-USDT": 2,
             "BNB-USDT": 3, "LINK-USDT": 4, "XRP-USDT": 5}
    seed  = seeds.get(symbol, 0)
    rng   = np.random.default_rng(seed)
    now   = int(time.time())

    prices = [30000.0 if "BTC" in symbol else
              (2000.0 if "ETH" in symbol else
               (100.0 if "SOL" in symbol else 50.0))]
    slope  = {"BTC-USDT": 0.3, "ETH-USDT": 0.4}.get(symbol, 0.2)
    for _ in range(n - 1):
        prices.append(max(1.0, prices[-1] + slope + rng.standard_normal() * prices[-1] * 0.005))

    prices = np.array(prices)
    opens  = np.concatenate([[prices[0]], prices[:-1]])
    spread = abs(rng.standard_normal(n)) * prices * 0.002

    timestamps = pd.to_datetime(
        [now - (n - i) * 900 for i in range(n)],
        unit="s", utc=True
    )
    df = pd.DataFrame({
        "open":   opens,
        "high":   np.maximum(opens, prices) + spread,
        "low":    np.minimum(opens, prices) - spread,
        "close":  prices,
        "volume": rng.integers(1000, 50000, n).astype(float),
    }, index=timestamps)
    return df


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           symbols=SYMBOLS_AVAILABLE,
                           strategies=list(STRATEGIES_AVAILABLE.keys()),
                           default_symbol=DEFAULT_SYMBOL,
                           default_interval=DEFAULT_INTERVAL,
                           use_real_data=USE_REAL_DATA)


@app.route("/api/ohlcv")
def api_ohlcv():
    """Devuelve OHLCV en formato JSON para el gráfico de velas."""
    symbol   = request.args.get("symbol",   DEFAULT_SYMBOL)
    interval = request.args.get("interval", DEFAULT_INTERVAL)
    limit    = int(request.args.get("limit", DEFAULT_LIMIT))

    try:
        df = _get_data(symbol, interval, limit)
        if df is None or df.empty:
            return jsonify({"error": "no_data"}), 404

        candles = []
        for ts, row in df.iterrows():
            candles.append({
                "t": int(ts.timestamp() * 1000),
                "o": round(float(row["open"]),   4),
                "h": round(float(row["high"]),   4),
                "l": round(float(row["low"]),    4),
                "c": round(float(row["close"]),  4),
                "v": round(float(row["volume"]), 2),
            })

        return jsonify({
            "symbol":   symbol,
            "interval": interval,
            "candles":  candles,
            "count":    len(candles),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/regime")
def api_regime():
    """Detecta el régimen actual de mercado."""
    symbol   = request.args.get("symbol",   DEFAULT_SYMBOL)
    interval = request.args.get("interval", DEFAULT_INTERVAL)

    try:
        df = _get_data(symbol, interval, 200)
        if df is None or df.empty:
            return jsonify({"error": "no_data"}), 404

        regime_info = REGIME_DETECTOR.detect(df)
        return jsonify({
            "symbol":       symbol,
            "regime":       regime_info.get("regime", "unknown"),
            "regime_label": RegimeDetector.regime_label(regime_info.get("regime", "")),
            "adx":          round(float(regime_info.get("adx", 0) or 0), 2),
            "atr_ratio":    round(float(regime_info.get("atr_ratio", 0) or 0), 4),
            "rsi":          round(float(regime_info.get("rsi", 50) or 50), 1),
            "ema_slope":    round(float(regime_info.get("ema_slope", 0) or 0), 6),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals")
def api_signals():
    """Genera señales de todas las estrategias disponibles."""
    symbol   = request.args.get("symbol",   DEFAULT_SYMBOL)
    interval = request.args.get("interval", DEFAULT_INTERVAL)

    try:
        df = _get_data(symbol, interval, 200)
        if df is None or df.empty:
            return jsonify({"error": "no_data"}), 404

        results = {}
        for name, strategy in STRATEGIES_AVAILABLE.items():
            try:
                sig = strategy.generate_signal(df)
                results[name] = {
                    "signal": sig.get("signal", "hold"),
                    "reason": sig.get("reason", ""),
                    **{k: v for k, v in sig.items()
                       if k not in ("signal", "reason")
                       and isinstance(v, (int, float, str, bool, type(None)))},
                }
            except Exception as ex:
                results[name] = {"signal": "error", "reason": str(ex)}

        return jsonify({
            "symbol":    symbol,
            "interval":  interval,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals":   results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ticker")
def api_ticker():
    """Stats actuales del mercado."""
    symbol = request.args.get("symbol", DEFAULT_SYMBOL)

    try:
        if USE_REAL_DATA:
            from data.kucoin_client import KuCoinClient
            client = KuCoinClient()
            ticker = client.get_ticker(symbol)
        else:
            # Demo: usa el último precio del OHLCV
            df = _get_data(symbol, DEFAULT_INTERVAL, 50)
            last_close = float(df["close"].iloc[-1]) if df is not None and not df.empty else 0.0
            ticker = {
                "symbol": symbol, "price": last_close,
                "high_24h": last_close * 1.02, "low_24h": last_close * 0.98,
                "volume_24h": 5000.0, "change_pct": 1.5,
                "bid": last_close * 0.9999, "ask": last_close * 1.0001,
            }
        return jsonify(ticker)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest_summary")
def api_backtest_summary():
    """Carga el último resultado de backtest si existe."""
    try:
        path = "data/ml_backtest_results.csv"
        if not os.path.exists(path):
            path = "data/hyperparam_results.csv"
        if not os.path.exists(path):
            return jsonify({"error": "no_results_file"})

        df = pd.read_csv(path)
        summary = df.head(5).to_dict(orient="records")
        return jsonify({"results": summary, "source": os.path.basename(path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
def api_status():
    """Estado del sistema."""
    return jsonify({
        "status":        "ok",
        "use_real_data": USE_REAL_DATA,
        "strategies":    list(STRATEGIES_AVAILABLE.keys()),
        "symbols":       SYMBOLS_AVAILABLE,
        "cache_entries": len(_cache),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    })


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "5000"))
    print(f"\n  Trading Bot Dashboard")
    print(f"  Modo: {'REAL (KuCoin)' if USE_REAL_DATA else 'DEMO (datos simulados)'}")
    print(f"  URL:  http://localhost:{port}")
    print(f"  Estrategias: {', '.join(STRATEGIES_AVAILABLE.keys())}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
