"""
order_blocks.py

Detecta Order Blocks (OB) en SMC:

- Bullish OB: ultima vela bajista antes de un movimiento alcista fuerte
  que rompe un swing high. El cuerpo de esa vela es la zona de demanda.

- Bearish OB: ultima vela alcista antes de un movimiento bajista fuerte
  que rompe un swing low. El cuerpo es la zona de oferta.

Un OB es "activo" mientras el precio no lo haya cruzado completamente.
"""


def detect_order_blocks(data, swing_window=5, min_move=0.005):
    """
    Detecta Order Blocks en el dataset.

    Parametros:
      swing_window : velas a cada lado para identificar swings
      min_move     : movimiento minimo (fraccion del precio) para
                     considerar que el OB genero un impulso valido

    Devuelve lista de dicts:
      type        : "bullish" o "bearish"
      ob_high     : limite superior del OB
      ob_low      : limite inferior del OB
      index       : indice de la vela OB
      broken      : True si el precio ha vuelto y roto el OB
    """
    if data is None or len(data) < swing_window * 2 + 3:
        return []

    opens  = data["open"].values
    highs  = data["high"].values
    lows   = data["low"].values
    closes = data["close"].values
    n = len(data)

    # Detectar swing highs y lows
    swing_highs = []
    swing_lows = []
    for i in range(swing_window, n - swing_window):
        if highs[i] > highs[i - swing_window:i].max() and \
           highs[i] >= highs[i + 1:i + swing_window + 1].max():
            swing_highs.append((i, highs[i]))
        if lows[i] < lows[i - swing_window:i].min() and \
           lows[i] <= lows[i + 1:i + swing_window + 1].min():
            swing_lows.append((i, lows[i]))

    order_blocks = []

    # Bullish OB: buscar la ultima vela bajista antes de romper un swing high
    for sh_idx, sh_price in swing_highs:
        # Buscar la ultima vela bajista en las 5 velas previas al swing
        for j in range(sh_idx - 1, max(sh_idx - 6, 0) - 1, -1):
            if closes[j] < opens[j]:  # vela bajista
                # Verificar que el movimiento desde ese punto fue significativo
                move = (sh_price - closes[j]) / closes[j]
                if move >= min_move:
                    ob_high = max(opens[j], closes[j])
                    ob_low  = min(opens[j], closes[j])
                    # Comprobar si ya fue roto (precio cerro por debajo del OB)
                    broken = any(closes[k] < ob_low for k in range(sh_idx, n))
                    order_blocks.append({
                        "type":    "bullish",
                        "ob_high": ob_high,
                        "ob_low":  ob_low,
                        "index":   j,
                        "broken":  broken,
                    })
                break

    # Bearish OB: buscar la ultima vela alcista antes de romper un swing low
    for sl_idx, sl_price in swing_lows:
        for j in range(sl_idx - 1, max(sl_idx - 6, 0) - 1, -1):
            if closes[j] > opens[j]:  # vela alcista
                move = (closes[j] - sl_price) / closes[j]
                if move >= min_move:
                    ob_high = max(opens[j], closes[j])
                    ob_low  = min(opens[j], closes[j])
                    broken = any(closes[k] > ob_high for k in range(sl_idx, n))
                    order_blocks.append({
                        "type":    "bearish",
                        "ob_high": ob_high,
                        "ob_low":  ob_low,
                        "index":   j,
                        "broken":  broken,
                    })
                break

    return order_blocks


def get_active_order_blocks(data, swing_window=5, min_move=0.005):
    """Devuelve solo los OBs que no han sido rotos."""
    obs = detect_order_blocks(data, swing_window, min_move)
    return [ob for ob in obs if not ob["broken"]]


def price_in_order_block(price, order_blocks, ob_type=None):
    """
    Comprueba si un precio esta dentro de algun OB activo.

    ob_type : "bullish", "bearish" o None (cualquiera)
    """
    for ob in order_blocks:
        if ob_type and ob["type"] != ob_type:
            continue
        if ob["ob_low"] <= price <= ob["ob_high"]:
            return True
    return False
