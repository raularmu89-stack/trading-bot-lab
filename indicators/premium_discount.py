"""
premium_discount.py

Conversión desde Pine Script 'raul smc' (TradingView).

Zonas Premium, Descuento y Equilibrio basadas en el rango swing high/low:

  Premium     : parte superior del rango (> 50%)  → zona de oferta / venta
  Equilibrium : mitad del rango (45%–55%)          → zona neutral
  Discount    : parte inferior del rango (< 50%)   → zona de demanda / compra

Pine equivalent:
    drawPremiumDiscountZones():
        Premium zone:     trailing.top  → 0.95*top + 0.05*bottom
        Equilibrium zone: avg(top,bot) ± 2.5% del rango
        Discount zone:    0.95*bot + 0.05*top → trailing.bottom
"""

import numpy as np


def detect_premium_discount(data, swing_window=50):
    """
    Calcula las zonas Premium / Equilibrium / Discount.

    Pine equivalent:
        trailing.top    = max(high) del rango actual
        trailing.bottom = min(low)  del rango actual

    Parametros:
      swing_window : velas atrás para determinar el rango (0 = todo el dataset)

    Devuelve dict:
      swing_high   : máximo del rango
      swing_low    : mínimo del rango
      premium_top  : límite superior Premium
      premium_bot  : límite inferior Premium (= 95% rango desde arriba)
      eq_top       : límite superior Equilibrium
      eq_bot       : límite inferior Equilibrium
      discount_top : límite superior Discount (= 5% rango desde arriba)
      discount_bot : límite inferior Discount
      equilibrium  : precio punto medio exacto
      zone         : zona donde está el precio actual
                     "premium" | "equilibrium" | "discount" | "unknown"
      zone_pct     : posición porcentual del precio en el rango (0=min, 1=max)
    """
    if data is None or len(data) < 2:
        return None

    h = data["high"].values.astype(float)
    l = data["low"].values.astype(float)
    c = data["close"].values.astype(float)

    window_slice = slice(-swing_window, None) if swing_window > 0 else slice(None)
    top    = float(h[window_slice].max())
    bottom = float(l[window_slice].min())
    rng    = top - bottom

    if rng == 0:
        return None

    # Pine exact percentages:
    #   Premium top    = top
    #   Premium bot    = 0.95*top + 0.05*bottom
    #   Eq top         = 0.525*top + 0.475*bottom
    #   Eq bot         = 0.525*bottom + 0.475*top  (= 0.475*top + 0.525*bottom)
    #   Discount top   = 0.05*top + 0.95*bottom    (= 0.95*bot + 0.05*top)
    #   Discount bot   = bottom
    premium_top  = top
    premium_bot  = 0.95 * top   + 0.05 * bottom
    eq_top       = 0.525 * top  + 0.475 * bottom
    eq_bot       = 0.475 * top  + 0.525 * bottom
    discount_top = 0.05 * top   + 0.95 * bottom
    discount_bot = bottom
    equilibrium  = (top + bottom) / 2

    current_price = float(c[-1])
    zone_pct = (current_price - bottom) / rng

    if current_price >= premium_bot:
        zone = "premium"
    elif current_price <= discount_top:
        zone = "discount"
    elif eq_bot <= current_price <= eq_top:
        zone = "equilibrium"
    elif current_price > eq_top:
        zone = "premium"
    else:
        zone = "discount"

    return {
        "swing_high":   top,
        "swing_low":    bottom,
        "premium_top":  premium_top,
        "premium_bot":  premium_bot,
        "eq_top":       eq_top,
        "eq_bot":       eq_bot,
        "discount_top": discount_top,
        "discount_bot": discount_bot,
        "equilibrium":  equilibrium,
        "zone":         zone,
        "zone_pct":     round(zone_pct, 4),
    }


def price_zone(price, pd_result):
    """
    Devuelve la zona de un precio dado un resultado de detect_premium_discount.

    Devuelve: "premium" | "equilibrium" | "discount" | "unknown"
    """
    if pd_result is None:
        return "unknown"
    if price >= pd_result["premium_bot"]:
        return "premium"
    if price <= pd_result["discount_top"]:
        return "discount"
    if pd_result["eq_bot"] <= price <= pd_result["eq_top"]:
        return "equilibrium"
    return "premium" if price > pd_result["equilibrium"] else "discount"
