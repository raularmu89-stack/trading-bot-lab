class SMCStrategy:
    def __init__(self):
        pass

    def generate_signal(self, data):
        """
        Devuelve una señal basada en la estrategia SMC.
        """
        if data is None or len(data) == 0:
            return None

        return {
            "signal": "hold",
            "reason": "No setup detected"
        }
