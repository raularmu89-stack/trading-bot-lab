"""
signal_filter.py

Envuelve cualquier estrategia existente y añade un filtro de ML:
  1. La estrategia subyacente genera su señal normal
  2. Si la señal es buy/sell, el filtro ML predice P(rentable)
  3. Si P < threshold → señal degradada a "hold"
  4. Tras cada trade completado, el modelo aprende del resultado

Uso con paper trader / backtest:
    from strategies.smc_strategy import SMCStrategy
    from ml.signal_filter import MLSignalFilter

    base = SMCStrategy(swing_window=5)
    filtered = MLSignalFilter(base, threshold=0.55)

    signal = filtered.generate_signal(df)
    # → {"signal": "buy", "ml_prob": 0.71, "ml_filtered": False, ...}

    # Tras el trade:
    filtered.record_trade(features, pnl=0.03)   # pnl > 0 → win
"""

import numpy as np
import os
from typing import Optional

from ml.feature_extractor import extract_features, N_FEATURES
from ml.neural_net import TradingNet


# Umbral por defecto: solo pasar señales con P(win) > 55%
DEFAULT_THRESHOLD = 0.52


class MLSignalFilter:
    """
    Filtro ML que envuelve cualquier estrategia del sistema.

    Parámetros
    ----------
    strategy        : estrategia base (cualquiera con generate_signal())
    threshold       : P(rentable) mínimo para dejar pasar la señal (0.52)
    model_path      : ruta para guardar/cargar pesos (None = no persistir)
    min_trades_train: trades mínimos antes de activar el filtro (30)
    retrain_every   : reentrenar batch cada N trades nuevos (20)
    online_learning : activar partial_fit() tras cada trade (True)
    verbose         : incluir detalles ML en la señal devuelta
    """

    def __init__(self,
                 strategy,
                 threshold: float = DEFAULT_THRESHOLD,
                 model_path: Optional[str] = None,
                 min_trades_train: int = 30,
                 retrain_every: int = 20,
                 online_learning: bool = True,
                 verbose: bool = True):

        self.strategy         = strategy
        self.threshold        = threshold
        self.model_path       = model_path
        self.min_trades_train = min_trades_train
        self.retrain_every    = retrain_every
        self.online_learning  = online_learning
        self.verbose          = verbose

        # FastBacktester compat — delega al strategy base
        self.swing_window   = getattr(strategy, "swing_window", 5)
        self.require_fvg    = getattr(strategy, "require_fvg", False)
        self.use_choch_filter = getattr(strategy, "use_choch_filter", False)

        # Historial de entrenamiento: [(features, label), ...]
        self._history: list = []
        self._pending: list = []   # trades con features pero sin resultado todavía

        # Cargar o crear modelo
        if model_path and os.path.exists(model_path + ".npz"):
            try:
                self.model = TradingNet.load(model_path)
            except Exception:
                self.model = TradingNet(input_dim=N_FEATURES)
        else:
            self.model = TradingNet(input_dim=N_FEATURES)

        self._trades_since_retrain = 0
        self._active = False          # False hasta min_trades_train

    # ── Señal filtrada ─────────────────────────────────────────────────────────

    def generate_signal(self, data) -> dict:
        """
        Genera señal de la estrategia base y la filtra con el modelo ML.
        """
        base_result = self.strategy.generate_signal(data)
        signal      = base_result.get("signal", "hold")

        if signal not in ("buy", "sell"):
            return base_result

        # Extraer features para esta señal
        features = extract_features(data, signal)
        if features is None:
            return base_result

        # Guardar features como "pendiente" (se etiquetará cuando cierre el trade)
        self._pending.append({"features": features, "signal": signal})

        # Si el modelo no está activo aún, dejar pasar la señal base
        if not self._active:
            if self.verbose:
                base_result["ml_prob"]     = None
                base_result["ml_active"]   = False
                base_result["ml_filtered"] = False
            return base_result

        # Predicción
        prob = float(self.model.predict_single(features))

        # Filtrar si P < threshold
        if prob < self.threshold:
            filtered_result = dict(base_result)
            filtered_result["signal"]       = "hold"
            filtered_result["ml_filtered"]  = True
            filtered_result["ml_prob"]      = round(prob, 4)
            filtered_result["ml_reason"]    = f"ML filtró señal ({prob:.2%} < {self.threshold:.0%})"
            return filtered_result

        if self.verbose:
            base_result["ml_prob"]     = round(prob, 4)
            base_result["ml_filtered"] = False
            base_result["ml_active"]   = True

        return base_result

    # ── Registro de resultado del trade ──────────────────────────────────────

    def record_trade(self, pnl: float, features: Optional[np.ndarray] = None):
        """
        Registra el resultado de un trade completado.

        Llamar tras cada trade cerrado:
            filter.record_trade(pnl=0.025)   # PnL porcentual

        pnl      : retorno del trade (positivo = ganancia)
        features : vector de features (si None, usa el último pendiente)
        """
        label = 1.0 if pnl > 0 else 0.0

        # Usar features del último pending si no se proporcionan
        if features is None and self._pending:
            feat_data = self._pending.pop(0)
            features  = feat_data["features"]
        elif features is None:
            return  # Sin features, no podemos aprender

        self._history.append((features, label))
        self._trades_since_retrain += 1

        # Online learning: partial_fit inmediato
        if self.online_learning:
            self.model.partial_fit(features, label)

        # Activar el filtro cuando tengamos suficientes trades
        if not self._active and len(self._history) >= self.min_trades_train:
            self._retrain_batch(verbose=True)
            self._active = True

        # Reentrenamiento batch periódico
        elif self._active and self._trades_since_retrain >= self.retrain_every:
            self._retrain_batch()
            self._trades_since_retrain = 0

        # Guardar pesos periódicamente
        if self.model_path and len(self._history) % 50 == 0:
            self.save()

    def record_trade_result(self, entry_price: float, exit_price: float,
                             side: str, features: Optional[np.ndarray] = None):
        """Alternativa: registra con precios de entrada/salida."""
        if side == "buy":
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price
        self.record_trade(pnl, features)

    # ── Reentrenamiento batch ─────────────────────────────────────────────────

    def _retrain_batch(self, epochs: int = 20, verbose: bool = False):
        """Reentrena con todo el historial disponible."""
        if len(self._history) < 10:
            return
        X = np.array([h[0] for h in self._history], dtype=np.float32)
        y = np.array([h[1] for h in self._history], dtype=np.float32)
        self.model.fit(X, y, epochs=epochs, verbose=verbose)
        if verbose:
            s = self.model.score(X, y)
            print(f"  [ML] Reentrenado con {len(y)} trades — "
                  f"acc={s['accuracy']:.3f}  f1={s['f1']:.3f}  "
                  f"steps={s['steps']}")

    # ── Stats y utilidades ────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Estadísticas del filtro ML."""
        n      = len(self._history)
        labels = [h[1] for h in self._history]
        wins   = sum(labels)
        if n == 0:
            return {"trades_seen": 0, "active": False, "threshold": self.threshold}
        if n >= 5:
            X = np.array([h[0] for h in self._history], dtype=np.float32)
            y = np.array(labels, dtype=np.float32)
            score = self.model.score(X, y)
        else:
            score = {}
        return {
            "trades_seen":   n,
            "win_rate_data": round(wins / n, 4),
            "active":        self._active,
            "threshold":     self.threshold,
            "model_steps":   self.model._step,
            "model_lr":      round(self.model.lr, 7),
            **score,
        }

    def save(self, path: Optional[str] = None):
        """Guarda el modelo en disco."""
        p = path or self.model_path
        if p:
            self.model.save(p)

    def load(self, path: Optional[str] = None):
        """Carga el modelo desde disco."""
        p = path or self.model_path
        if p and os.path.exists(p + ".npz"):
            self.model = TradingNet.load(p)
            self._active = self.model._step >= self.min_trades_train

    def reset(self):
        """Reinicia el modelo (para tests o nuevo par)."""
        self.model                = TradingNet(input_dim=N_FEATURES)
        self._history             = []
        self._pending             = []
        self._active              = False
        self._trades_since_retrain = 0

    def __repr__(self):
        return (f"MLSignalFilter(strategy={type(self.strategy).__name__}, "
                f"threshold={self.threshold}, active={self._active}, "
                f"trades={len(self._history)})")
