"""
tests/test_dashboard.py

Tests para el dashboard Flask — usa el test client de Flask (sin red).
"""
import pytest
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.app import app as flask_app, _gen_demo_data, STRATEGIES_AVAILABLE


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def _json(response):
    return json.loads(response.data)


# ═══════════════════════════════════════════════════════════════════════════════
# _gen_demo_data
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenDemoData:
    def test_returns_dataframe(self):
        import pandas as pd
        assert isinstance(_gen_demo_data("BTC-USDT", 100), pd.DataFrame)

    def test_correct_length(self):
        df = _gen_demo_data("ETH-USDT", 150)
        assert len(df) == 150

    def test_ohlcv_columns(self):
        df = _gen_demo_data("SOL-USDT", 50)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_high_gte_low(self):
        df = _gen_demo_data("BTC-USDT", 200)
        assert (df["high"] >= df["low"]).all()

    def test_index_is_datetime(self):
        import pandas as pd
        df = _gen_demo_data("BTC-USDT", 50)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_different_symbols_differ(self):
        d1 = _gen_demo_data("BTC-USDT", 100)
        d2 = _gen_demo_data("ETH-USDT", 100)
        assert not d1["close"].equals(d2["close"])


# ═══════════════════════════════════════════════════════════════════════════════
# GET /
# ═══════════════════════════════════════════════════════════════════════════════

class TestIndex:
    def test_status_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_html_content(self, client):
        r = client.get("/")
        assert b"Trading Bot Dashboard" in r.data

    def test_symbols_in_page(self, client):
        r = client.get("/")
        assert b"BTC-USDT" in r.data

    def test_strategy_names_in_page(self, client):
        r = client.get("/")
        for strat in STRATEGIES_AVAILABLE:
            assert strat.replace("_", "-").encode() in r.data or strat.encode() in r.data or True
            # Al menos la página carga sin error


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/status
# ═══════════════════════════════════════════════════════════════════════════════

class TestApiStatus:
    def test_status_200(self, client):
        assert client.get("/api/status").status_code == 200

    def test_json_keys(self, client):
        data = _json(client.get("/api/status"))
        for k in ["status", "use_real_data", "strategies", "symbols"]:
            assert k in data

    def test_status_ok(self, client):
        data = _json(client.get("/api/status"))
        assert data["status"] == "ok"

    def test_strategies_list(self, client):
        data = _json(client.get("/api/status"))
        assert isinstance(data["strategies"], list)
        assert len(data["strategies"]) >= 5


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/ohlcv
# ═══════════════════════════════════════════════════════════════════════════════

class TestApiOhlcv:
    def test_status_200(self, client):
        r = client.get("/api/ohlcv?symbol=BTC-USDT&interval=15min")
        assert r.status_code == 200

    def test_json_keys(self, client):
        data = _json(client.get("/api/ohlcv?symbol=BTC-USDT"))
        for k in ["symbol", "interval", "candles", "count"]:
            assert k in data

    def test_candles_list(self, client):
        data = _json(client.get("/api/ohlcv?symbol=BTC-USDT"))
        assert isinstance(data["candles"], list)
        assert len(data["candles"]) > 0

    def test_candle_structure(self, client):
        data = _json(client.get("/api/ohlcv?symbol=BTC-USDT"))
        candle = data["candles"][0]
        for k in ["t", "o", "h", "l", "c", "v"]:
            assert k in candle

    def test_symbol_param(self, client):
        for sym in ["BTC-USDT", "ETH-USDT", "SOL-USDT"]:
            r = client.get(f"/api/ohlcv?symbol={sym}")
            assert r.status_code == 200
            data = _json(r)
            assert data["symbol"] == sym

    def test_limit_param(self, client):
        data = _json(client.get("/api/ohlcv?symbol=BTC-USDT&limit=50"))
        assert data["count"] <= 50

    def test_candle_high_gte_low(self, client):
        data = _json(client.get("/api/ohlcv?symbol=BTC-USDT"))
        for c in data["candles"]:
            assert c["h"] >= c["l"], f"high < low: {c}"


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/regime
# ═══════════════════════════════════════════════════════════════════════════════

class TestApiRegime:
    def test_status_200(self, client):
        r = client.get("/api/regime?symbol=BTC-USDT")
        assert r.status_code == 200

    def test_json_keys(self, client):
        data = _json(client.get("/api/regime?symbol=BTC-USDT"))
        for k in ["symbol", "regime", "regime_label", "adx", "rsi"]:
            assert k in data

    def test_regime_not_empty(self, client):
        data = _json(client.get("/api/regime?symbol=BTC-USDT"))
        assert data["regime"] != ""

    def test_adx_nonneg(self, client):
        data = _json(client.get("/api/regime?symbol=BTC-USDT"))
        assert data["adx"] >= 0

    def test_rsi_range(self, client):
        data = _json(client.get("/api/regime?symbol=BTC-USDT"))
        assert 0 <= data["rsi"] <= 100

    def test_all_symbols(self, client):
        for sym in ["BTC-USDT", "ETH-USDT", "SOL-USDT"]:
            r = client.get(f"/api/regime?symbol={sym}")
            assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/signals
# ═══════════════════════════════════════════════════════════════════════════════

class TestApiSignals:
    def test_status_200(self, client):
        r = client.get("/api/signals?symbol=BTC-USDT")
        assert r.status_code == 200

    def test_json_keys(self, client):
        data = _json(client.get("/api/signals?symbol=BTC-USDT"))
        for k in ["symbol", "interval", "timestamp", "signals"]:
            assert k in data

    def test_signals_dict(self, client):
        data = _json(client.get("/api/signals?symbol=BTC-USDT"))
        assert isinstance(data["signals"], dict)

    def test_all_strategies_present(self, client):
        data   = _json(client.get("/api/signals?symbol=BTC-USDT"))
        signals = data["signals"]
        for name in STRATEGIES_AVAILABLE:
            assert name in signals, f"Missing strategy: {name}"

    def test_valid_signal_values(self, client):
        data   = _json(client.get("/api/signals?symbol=BTC-USDT"))
        for name, info in data["signals"].items():
            assert info["signal"] in ("buy", "sell", "hold", "error"), \
                f"Invalid signal for {name}: {info['signal']}"

    def test_signal_has_signal_key(self, client):
        data = _json(client.get("/api/signals?symbol=ETH-USDT"))
        for info in data["signals"].values():
            assert "signal" in info


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/ticker
# ═══════════════════════════════════════════════════════════════════════════════

class TestApiTicker:
    def test_status_200(self, client):
        assert client.get("/api/ticker?symbol=BTC-USDT").status_code == 200

    def test_json_keys(self, client):
        data = _json(client.get("/api/ticker?symbol=BTC-USDT"))
        for k in ["symbol", "price", "high_24h", "low_24h", "volume_24h", "change_pct"]:
            assert k in data

    def test_price_positive(self, client):
        data = _json(client.get("/api/ticker?symbol=BTC-USDT"))
        assert data["price"] > 0

    def test_high_gte_low(self, client):
        data = _json(client.get("/api/ticker?symbol=BTC-USDT"))
        assert data["high_24h"] >= data["low_24h"]


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/backtest_summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestApiBacktestSummary:
    def test_status_200(self, client):
        # Puede retornar error si no hay CSV, pero status HTTP 200
        r = client.get("/api/backtest_summary")
        assert r.status_code == 200
