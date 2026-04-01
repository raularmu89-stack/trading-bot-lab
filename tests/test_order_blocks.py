"""
test_order_blocks.py — Tests para indicators/order_blocks.py (versión Pine-faithful)
"""

import numpy as np
import pandas as pd
import pytest
from indicators.order_blocks import (
    detect_order_blocks,
    detect_dual_order_blocks,
    get_active_order_blocks,
    price_in_order_block,
)


def _make_df(highs, lows, closes, opens=None):
    n = len(highs)
    opens = opens or [c * 0.999 for c in closes]
    return pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": [1000.0] * n,
    })


def _trending_up(n=120, start=100.0, step=0.8):
    """Subida sostenida → genera swing highs → OBs bajistas (antes del impulso)."""
    prices = [start + i * step for i in range(n)]
    highs  = [p + 0.5 for p in prices]
    lows   = [p - 0.5 for p in prices]
    return _make_df(highs, lows, prices)


def _trending_down(n=120, start=200.0, step=0.8):
    """Bajada sostenida → genera swing lows → OBs alcistas."""
    prices = [start - i * step for i in range(n)]
    highs  = [p + 0.5 for p in prices]
    lows   = [p - 0.5 for p in prices]
    return _make_df(highs, lows, prices)


def _impulse_up(n=80, base=100.0):
    """
    Datos con un impulso alcista claro en la mitad:
      1. Lateral (40 velas) → establece swing low
      2. Impulso alcista (40 velas) → rompe swing high → genera bullish OB
    """
    lateral  = [base + np.sin(i * 0.3) for i in range(40)]
    impulse  = [base + i * 1.2 for i in range(40)]
    prices   = lateral + impulse
    highs    = [p + 1.0 for p in prices]
    lows     = [p - 1.0 for p in prices]
    return _make_df(highs, lows, prices)


def _mock_data(n=200):
    from bot.mock_data import generate_mock_klines
    return generate_mock_klines("BTCUSDT", n_candles=n, seed=99)


# ─── Datos insuficientes ─────────────────────────────────────────────────────

def test_returns_empty_with_none():
    assert detect_order_blocks(None) == []


def test_returns_empty_with_too_few_bars():
    df = _trending_up(n=5)
    assert detect_order_blocks(df, swing_window=5) == []


# ─── Estructura de resultado ──────────────────────────────────────────────────

def test_ob_entry_has_required_keys():
    df  = _mock_data(200)
    obs = detect_order_blocks(df, swing_window=5)
    if obs:
        ob = obs[0]
        for key in ("type", "ob_high", "ob_low", "ob_index",
                    "pivot_index", "mitigated", "broken", "index"):
            assert key in ob, f"Falta clave: {key}"


def test_ob_type_is_bullish_or_bearish():
    df  = _mock_data(200)
    obs = detect_order_blocks(df, swing_window=5)
    for ob in obs:
        assert ob["type"] in ("bullish", "bearish")


def test_ob_high_greater_than_ob_low():
    df  = _mock_data(200)
    obs = detect_order_blocks(df, swing_window=5)
    for ob in obs:
        assert ob["ob_high"] >= ob["ob_low"], \
            f"ob_high={ob['ob_high']} < ob_low={ob['ob_low']}"


def test_ob_index_within_bounds():
    df  = _mock_data(200)
    obs = detect_order_blocks(df, swing_window=5)
    for ob in obs:
        assert 0 <= ob["ob_index"] < len(df)


def test_broken_equals_mitigated():
    """Backward compat: 'broken' == 'mitigated'."""
    df  = _mock_data(200)
    obs = detect_order_blocks(df, swing_window=5)
    for ob in obs:
        assert ob["broken"] == ob["mitigated"]


# ─── Mitigación ───────────────────────────────────────────────────────────────

def test_mitigation_close_vs_highlow_differ():
    """Los dos modos de mitigación pueden dar resultados distintos."""
    df      = _mock_data(300)
    obs_hl  = detect_order_blocks(df, swing_window=5, mitigation="highlow")
    obs_cl  = detect_order_blocks(df, swing_window=5, mitigation="close")
    # No tienen por qué ser iguales (highlow mitiga antes)
    mitigated_hl = sum(1 for o in obs_hl if o["mitigated"])
    mitigated_cl = sum(1 for o in obs_cl if o["mitigated"])
    # highlow mitiga con más frecuencia que close
    assert mitigated_hl >= mitigated_cl


# ─── get_active_order_blocks ─────────────────────────────────────────────────

def test_active_obs_are_not_mitigated():
    df      = _mock_data(300)
    active  = get_active_order_blocks(df, swing_window=5)
    for ob in active:
        assert not ob["mitigated"]


def test_active_obs_subset_of_all():
    df      = _mock_data(300)
    all_obs = detect_order_blocks(df, swing_window=5)
    active  = get_active_order_blocks(df, swing_window=5)
    assert len(active) <= len(all_obs)


# ─── price_in_order_block ────────────────────────────────────────────────────

def test_price_inside_ob():
    obs   = [{"type": "bullish", "ob_high": 110.0, "ob_low": 100.0}]
    assert price_in_order_block(105.0, obs) is True


def test_price_outside_ob():
    obs   = [{"type": "bullish", "ob_high": 110.0, "ob_low": 100.0}]
    assert price_in_order_block(90.0, obs) is False


def test_price_on_ob_boundary():
    obs   = [{"type": "bearish", "ob_high": 110.0, "ob_low": 100.0}]
    assert price_in_order_block(110.0, obs) is True
    assert price_in_order_block(100.0, obs) is True


def test_price_in_ob_with_type_filter():
    obs = [
        {"type": "bullish", "ob_high": 110.0, "ob_low": 100.0},
        {"type": "bearish", "ob_high": 200.0, "ob_low": 190.0},
    ]
    assert price_in_order_block(105.0, obs, ob_type="bullish") is True
    assert price_in_order_block(105.0, obs, ob_type="bearish") is False
    assert price_in_order_block(195.0, obs, ob_type="bearish") is True


def test_price_in_empty_ob_list():
    assert price_in_order_block(100.0, []) is False


# ─── detect_dual_order_blocks ────────────────────────────────────────────────

def test_dual_ob_has_internal_and_swing_keys():
    df     = _mock_data(300)
    result = detect_dual_order_blocks(df, internal_window=5, swing_window=30)
    assert "internal" in result
    assert "swing"    in result


def test_dual_ob_internal_and_swing_are_lists():
    df     = _mock_data(300)
    result = detect_dual_order_blocks(df, internal_window=5, swing_window=30)
    assert isinstance(result["internal"], list)
    assert isinstance(result["swing"],    list)


def test_dual_ob_swing_window_larger_gives_fewer_obs():
    """Ventana más grande → pivots más escasos → menos OBs esperables."""
    df      = _mock_data(500)
    small   = detect_dual_order_blocks(df, internal_window=5, swing_window=10)
    large   = detect_dual_order_blocks(df, internal_window=5, swing_window=40)
    # Con ventana grande hay menos pivots disponibles
    assert len(large["swing"]) <= len(small["internal"])
