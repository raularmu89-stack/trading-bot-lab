from indicators.market_structure import detect_market_structure
from indicators.fvg import detect_fvg


class SMCStrategy:
    def __init__(self):
        pass

    def generate_signal(self, data):
        """
        Genera una señal simple basada en estructura de mercado y FVG.
        Devuelve:
        - buy
        - sell
        - hold
        """
        if data is None or len(data) < 5:
            return {
                "signal": "hold",
                "reason": "Not enough data"
            }

        structure = detect_market_structure(data)
        fvgs = detect_fvg(data)

        if structure is None:
            return {
                "signal": "hold",
                "reason": "No market structure detected"
            }

        trend = structure.get("trend", "neutral")
        bos = structure.get("bos", False)
        choch = structure.get("choch", False)

        # Lógica base inicial
        if trend == "bullish" and bos:
            return {
                "signal": "buy",
                "reason": "Bullish BOS detected",
                "structure": structure,
                "fvg_count": len(fvgs)
            }

        if trend == "bearish" and bos:
            return {
                "signal": "sell",
                "reason": "Bearish BOS detected",
                "structure": structure,
                "fvg_count": len(fvgs)
            }

        if choch:
            return {
                "signal": "hold",
                "reason": "CHoCH detected, waiting confirmation",
                "structure": structure,
                "fvg_count": len(fvgs)
            }

        return {
            "signal": "hold",
            "reason": "No clear setup",
            "structure": structure,
            "fvg_count": len(fvgs)
        }
