from indicators.market_structure import detect_market_structure
from indicators.fvg import detect_fvg


class SMCStrategy:
    def __init__(self, swing_window=5, require_fvg=False, use_choch_filter=True):
        """
        swing_window     : velas a cada lado para detectar swing highs/lows
        require_fvg      : exige FVG del mismo tipo para confirmar la entrada
        use_choch_filter : si True, CHoCH devuelve hold en lugar de operar
        """
        self.swing_window = swing_window
        self.require_fvg = require_fvg
        self.use_choch_filter = use_choch_filter

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

        if trend == "bullish" and bos:
            if self.require_fvg:
                bullish_fvgs = [f for f in fvgs if f["type"] == "bullish"]
                if not bullish_fvgs:
                    return {
                        "signal": "hold",
                        "reason": "No bullish FVG to confirm entry",
                        "structure": structure,
                        "fvg_count": len(fvgs),
                    }
            return {
                "signal": "buy",
                "reason": "Bullish BOS detected",
                "structure": structure,
                "fvg_count": len(fvgs),
            }

        if trend == "bearish" and bos:
            if self.require_fvg:
                bearish_fvgs = [f for f in fvgs if f["type"] == "bearish"]
                if not bearish_fvgs:
                    return {
                        "signal": "hold",
                        "reason": "No bearish FVG to confirm entry",
                        "structure": structure,
                        "fvg_count": len(fvgs),
                    }
            return {
                "signal": "sell",
                "reason": "Bearish BOS detected",
                "structure": structure,
                "fvg_count": len(fvgs),
            }

        return {
            "signal": "hold",
            "reason": "No clear setup",
            "structure": structure,
            "fvg_count": len(fvgs),
        }
