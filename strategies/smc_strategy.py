from indicators.market_structure import detect_market_structure
from indicators.fvg import detect_fvg
from indicators.order_blocks import get_active_order_blocks, price_in_order_block
from indicators.liquidity import last_sweep


class SMCStrategy:
    def __init__(
        self,
        swing_window=5,
        require_fvg=False,
        use_choch_filter=True,
        require_ob=False,
        require_sweep=False,
    ):
        """
        swing_window     : velas a cada lado para detectar swing highs/lows
        require_fvg      : exige FVG del mismo tipo para confirmar la entrada
        use_choch_filter : si True, CHoCH devuelve hold en lugar de operar
        require_ob       : exige que el precio este en un Order Block activo
        require_sweep    : exige un Liquidity Sweep reciente en la misma direccion
        """
        self.swing_window = swing_window
        self.require_fvg = require_fvg
        self.use_choch_filter = use_choch_filter
        self.require_ob = require_ob
        self.require_sweep = require_sweep

    def generate_signal(self, data):
        if data is None or len(data) < 5:
            return {"signal": "hold", "reason": "Not enough data"}

        structure = detect_market_structure(data, window=self.swing_window)
        fvgs = detect_fvg(data)

        if structure is None:
            return {"signal": "hold", "reason": "No market structure detected"}

        trend = structure.get("trend", "neutral")
        bos = structure.get("bos", False)
        choch = structure.get("choch", False)

        if self.use_choch_filter and choch:
            return {
                "signal": "hold",
                "reason": "CHoCH detected, waiting confirmation",
                "structure": structure,
                "fvg_count": len(fvgs),
            }

        current_price = float(data["close"].iloc[-1])

        # Order Blocks activos
        active_obs = get_active_order_blocks(data, self.swing_window) if self.require_ob else []

        # Ultimo liquidity sweep
        sweep = last_sweep(data, self.swing_window) if self.require_sweep else None

        if trend == "bullish" and bos:
            if self.require_fvg:
                if not any(f["type"] == "bullish" for f in fvgs):
                    return {"signal": "hold", "reason": "No bullish FVG",
                            "structure": structure, "fvg_count": len(fvgs)}

            if self.require_ob:
                if not price_in_order_block(current_price, active_obs, ob_type="bullish"):
                    return {"signal": "hold", "reason": "Price not in bullish OB",
                            "structure": structure, "fvg_count": len(fvgs)}

            if self.require_sweep:
                if sweep is None or sweep["type"] != "bullish":
                    return {"signal": "hold", "reason": "No bullish liquidity sweep",
                            "structure": structure, "fvg_count": len(fvgs)}

            return {
                "signal": "buy",
                "reason": "Bullish BOS" +
                          (" + OB" if self.require_ob else "") +
                          (" + Sweep" if self.require_sweep else ""),
                "structure": structure,
                "fvg_count": len(fvgs),
            }

        if trend == "bearish" and bos:
            if self.require_fvg:
                if not any(f["type"] == "bearish" for f in fvgs):
                    return {"signal": "hold", "reason": "No bearish FVG",
                            "structure": structure, "fvg_count": len(fvgs)}

            if self.require_ob:
                if not price_in_order_block(current_price, active_obs, ob_type="bearish"):
                    return {"signal": "hold", "reason": "Price not in bearish OB",
                            "structure": structure, "fvg_count": len(fvgs)}

            if self.require_sweep:
                if sweep is None or sweep["type"] != "bearish":
                    return {"signal": "hold", "reason": "No bearish liquidity sweep",
                            "structure": structure, "fvg_count": len(fvgs)}

            return {
                "signal": "sell",
                "reason": "Bearish BOS" +
                          (" + OB" if self.require_ob else "") +
                          (" + Sweep" if self.require_sweep else ""),
                "structure": structure,
                "fvg_count": len(fvgs),
            }

        return {
            "signal": "hold",
            "reason": "No clear setup",
            "structure": structure,
            "fvg_count": len(fvgs),
        }
