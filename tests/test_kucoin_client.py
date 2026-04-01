"""
tests/test_kucoin_client.py

Tests para KuCoinClient — usan mocks para no depender de red.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from data.kucoin_client import (
    KuCoinClient, VALID_INTERVALS, INTERVAL_ALIASES, INTERVAL_SECONDS,
    MAX_CANDLES_PER_REQUEST,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_candles(n=50, base_price=30000.0):
    """Genera datos de velas simulados en formato KuCoin (más reciente primero)."""
    import time
    now = int(time.time())
    candles = []
    for i in range(n):
        ts = now - i * 3600
        p  = base_price + i * 10.0
        candles.append([str(ts), str(p), str(p + 100), str(p + 200), str(p - 100), "500.0", "15000000.0"])
    return candles   # newest first, as KuCoin returns


def _fake_ticker():
    return {
        "last": "30000.0", "high": "31000.0", "low": "29000.0",
        "vol": "5000.0", "changeRate": "0.02",
        "buy": "29990.0", "sell": "30010.0",
    }


def _fake_orderbook():
    return {
        "price": "30000.0", "bestBid": "29990.0",
        "bestAsk": "30010.0", "time": "1700000000000",
    }


def _fake_symbols():
    return [
        {"symbol": "BTC-USDT", "quoteCurrency": "USDT", "enableTrading": True},
        {"symbol": "ETH-USDT", "quoteCurrency": "USDT", "enableTrading": True},
        {"symbol": "BTC-BTC",  "quoteCurrency": "BTC",  "enableTrading": True},  # debe excluirse
        {"symbol": "XRP-USDT", "quoteCurrency": "USDT", "enableTrading": False}, # debe excluirse
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Constantes y configuración
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_valid_intervals_not_empty(self):
        assert len(VALID_INTERVALS) > 0

    def test_common_intervals_present(self):
        for iv in ["1min", "15min", "1hour", "4hour", "1day"]:
            assert iv in VALID_INTERVALS

    def test_interval_seconds_complete(self):
        for iv in VALID_INTERVALS:
            assert iv in INTERVAL_SECONDS

    def test_interval_aliases_map_to_valid(self):
        for alias, kucoin in INTERVAL_ALIASES.items():
            assert kucoin in VALID_INTERVALS

    def test_max_candles(self):
        assert MAX_CANDLES_PER_REQUEST == 1500


# ═══════════════════════════════════════════════════════════════════════════════
# Inicialización
# ═══════════════════════════════════════════════════════════════════════════════

class TestInit:
    def test_default_params(self):
        c = KuCoinClient()
        assert c.timeout  == 10
        assert c.retries  == 3
        assert c.verbose  is False

    def test_custom_params(self):
        c = KuCoinClient(timeout=5, retries=1, verbose=True)
        assert c.timeout == 5
        assert c.retries == 1
        assert c.verbose is True

    def test_session_created(self):
        c = KuCoinClient()
        assert c._session is not None

    def test_repr(self):
        c = KuCoinClient(timeout=7, retries=2)
        r = repr(c)
        assert "KuCoinClient" in r
        assert "7" in r


# ═══════════════════════════════════════════════════════════════════════════════
# get_ohlcv
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetOhlcv:
    def _client(self, candles=None, n=50):
        c = KuCoinClient()
        c._get = MagicMock(return_value=candles if candles is not None else _fake_candles(n))
        return c

    def test_returns_dataframe(self):
        df = self._client().get_ohlcv("BTC-USDT")
        assert isinstance(df, pd.DataFrame)

    def test_columns(self):
        df = self._client().get_ohlcv("BTC-USDT")
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_index_is_datetime(self):
        df = self._client().get_ohlcv("BTC-USDT")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_index_utc(self):
        df = self._client().get_ohlcv("BTC-USDT")
        assert str(df.index.tz) == "UTC"

    def test_sorted_ascending(self):
        df = self._client().get_ohlcv("BTC-USDT")
        assert df.index.is_monotonic_increasing

    def test_dtype_float(self):
        df = self._client().get_ohlcv("BTC-USDT")
        for col in ["open", "high", "low", "close", "volume"]:
            assert df[col].dtype == np.float64 or df[col].dtype == np.float32

    def test_interval_alias_resolved(self):
        c = self._client()
        c.get_ohlcv("BTC-USDT", interval="1h")
        call_kwargs = c._get.call_args
        assert call_kwargs[0][1]["type"] == "1hour"  # alias resuelto

    def test_invalid_interval_raises(self):
        c = self._client()
        c._get = MagicMock(return_value=_fake_candles())
        with pytest.raises(ValueError):
            c.get_ohlcv("BTC-USDT", interval="99min")

    def test_limit_caps_at_max(self):
        c = self._client(n=100)
        df = c.get_ohlcv("BTC-USDT", limit=9999)
        # El limit se pasa a _get, pero el resultado se trunca
        assert len(df) <= MAX_CANDLES_PER_REQUEST

    def test_empty_response_returns_empty_df(self):
        c = self._client(candles=[])
        df = c.get_ohlcv("BTC-USDT")
        assert df.empty
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_limit_truncates(self):
        c = self._client(n=100)
        df = c.get_ohlcv("BTC-USDT", limit=20)
        assert len(df) <= 20

    def test_start_end_passed_to_get(self):
        c = self._client()
        c.get_ohlcv("BTC-USDT", start_time=1000, end_time=2000)
        params = c._get.call_args[0][1]
        assert params["startAt"] == 1000
        assert params["endAt"]   == 2000

    def test_symbol_passed(self):
        c = self._client()
        c.get_ohlcv("ETH-USDT")
        params = c._get.call_args[0][1]
        assert params["symbol"] == "ETH-USDT"


# ═══════════════════════════════════════════════════════════════════════════════
# get_multi_ohlcv
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetMultiOhlcv:
    def test_returns_dict(self):
        c = KuCoinClient()
        c._get = MagicMock(return_value=_fake_candles(30))
        result = c.get_multi_ohlcv(["BTC-USDT", "ETH-USDT"])
        assert isinstance(result, dict)

    def test_keys_match_symbols(self):
        c = KuCoinClient()
        c._get = MagicMock(return_value=_fake_candles(30))
        result = c.get_multi_ohlcv(["BTC-USDT", "ETH-USDT"])
        assert set(result.keys()) == {"BTC-USDT", "ETH-USDT"}

    def test_values_are_dataframes(self):
        c = KuCoinClient()
        c._get = MagicMock(return_value=_fake_candles(30))
        result = c.get_multi_ohlcv(["BTC-USDT"])
        assert isinstance(result["BTC-USDT"], pd.DataFrame)

    def test_error_symbol_returns_none(self):
        c = KuCoinClient()
        c._get = MagicMock(side_effect=ConnectionError("fallo"))
        result = c.get_multi_ohlcv(["BTC-USDT"])
        assert result["BTC-USDT"] is None

    def test_empty_list(self):
        c = KuCoinClient()
        assert c.get_multi_ohlcv([]) == {}


# ═══════════════════════════════════════════════════════════════════════════════
# get_ticker
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetTicker:
    def _client(self):
        c = KuCoinClient()
        c._get = MagicMock(return_value=_fake_ticker())
        return c

    def test_returns_dict(self):
        assert isinstance(self._client().get_ticker("BTC-USDT"), dict)

    def test_keys_present(self):
        t = self._client().get_ticker("BTC-USDT")
        for k in ["symbol", "price", "high_24h", "low_24h", "volume_24h", "change_pct", "bid", "ask"]:
            assert k in t

    def test_symbol_set(self):
        t = self._client().get_ticker("ETH-USDT")
        assert t["symbol"] == "ETH-USDT"

    def test_price_float(self):
        t = self._client().get_ticker("BTC-USDT")
        assert isinstance(t["price"], float)
        assert t["price"] == 30000.0

    def test_change_pct_multiplied(self):
        t = self._client().get_ticker("BTC-USDT")
        # changeRate=0.02 → change_pct = 2.0
        assert abs(t["change_pct"] - 2.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# get_orderbook_top
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetOrderbookTop:
    def _client(self):
        c = KuCoinClient()
        c._get = MagicMock(return_value=_fake_orderbook())
        return c

    def test_returns_dict(self):
        assert isinstance(self._client().get_orderbook_top("BTC-USDT"), dict)

    def test_keys(self):
        ob = self._client().get_orderbook_top("BTC-USDT")
        for k in ["symbol", "price", "bid", "ask", "time"]:
            assert k in ob

    def test_bid_ask_floats(self):
        ob = self._client().get_orderbook_top("BTC-USDT")
        assert ob["bid"] == 29990.0
        assert ob["ask"] == 30010.0


# ═══════════════════════════════════════════════════════════════════════════════
# list_usdt_symbols
# ═══════════════════════════════════════════════════════════════════════════════

class TestListUsdtSymbols:
    def _client(self):
        c = KuCoinClient()
        c._get = MagicMock(return_value=_fake_symbols())
        return c

    def test_returns_list(self):
        assert isinstance(self._client().list_usdt_symbols(), list)

    def test_only_usdt(self):
        symbols = self._client().list_usdt_symbols()
        for s in symbols:
            assert s.endswith("-USDT") or "USDT" in s

    def test_excludes_disabled(self):
        symbols = self._client().list_usdt_symbols()
        assert "XRP-USDT" not in symbols

    def test_excludes_non_usdt(self):
        symbols = self._client().list_usdt_symbols()
        assert "BTC-BTC" not in symbols

    def test_sorted(self):
        symbols = self._client().list_usdt_symbols()
        assert symbols == sorted(symbols)


# ═══════════════════════════════════════════════════════════════════════════════
# ping
# ═══════════════════════════════════════════════════════════════════════════════

class TestPing:
    def test_ping_true_on_success(self):
        c = KuCoinClient()
        c._get = MagicMock(return_value=_fake_ticker())
        assert c.ping() is True

    def test_ping_false_on_error(self):
        c = KuCoinClient()
        c._get = MagicMock(side_effect=ConnectionError("sin red"))
        assert c.ping() is False


# ═══════════════════════════════════════════════════════════════════════════════
# get_live_snapshot
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetLiveSnapshot:
    def test_returns_dict(self):
        c = KuCoinClient()
        c.get_ohlcv   = MagicMock(return_value=pd.DataFrame({"open":[], "close":[], "high":[], "low":[], "volume":[]}))
        c.get_ticker  = MagicMock(return_value={"symbol": "BTC-USDT", "price": 30000.0})
        result = c.get_live_snapshot(["BTC-USDT"])
        assert "BTC-USDT" in result

    def test_contains_ohlcv_and_ticker(self):
        c = KuCoinClient()
        c.get_ohlcv  = MagicMock(return_value=pd.DataFrame())
        c.get_ticker = MagicMock(return_value={"price": 1.0})
        result = c.get_live_snapshot(["ETH-USDT"])
        assert "ohlcv"  in result["ETH-USDT"]
        assert "ticker" in result["ETH-USDT"]

    def test_error_captured(self):
        c = KuCoinClient()
        c.get_ohlcv  = MagicMock(side_effect=ConnectionError("red caída"))
        c.get_ticker = MagicMock(return_value={})
        result = c.get_live_snapshot(["BTC-USDT"])
        assert result["BTC-USDT"]["ohlcv"] is None
        assert "error" in result["BTC-USDT"]


# ═══════════════════════════════════════════════════════════════════════════════
# _get — lógica de reintentos
# ═══════════════════════════════════════════════════════════════════════════════

class TestInternalGet:
    def test_raises_connection_error_after_retries(self):
        import requests as _req
        c = KuCoinClient(retries=2, timeout=1)
        with patch.object(c._session, "get", side_effect=_req.exceptions.ConnectionError("timeout")):
            with pytest.raises(ConnectionError):
                c._get("/api/v1/market/stats", {"symbol": "BTC-USDT"})

    def test_raises_on_kucoin_error_code(self):
        c = KuCoinClient(retries=1, timeout=1)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"code": "400100", "msg": "symbol not found"}
        with patch.object(c._session, "get", return_value=mock_resp):
            with pytest.raises(ValueError, match="KuCoin error"):
                c._get("/api/v1/market/stats", {"symbol": "XXX-USDT"})

    def test_returns_data_on_success(self):
        c = KuCoinClient(retries=1, timeout=1)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"code": "200000", "data": {"price": "100"}}
        with patch.object(c._session, "get", return_value=mock_resp):
            result = c._get("/api/v1/market/stats", {"symbol": "BTC-USDT"})
        assert result == {"price": "100"}
