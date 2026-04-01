"""
neural_net.py

Red neuronal MLP ligera implementada en puro NumPy.
Sin dependencias externas (no torch, no sklearn).

Arquitectura:
  Input(25) → Dense(32, ReLU) → Dropout → Dense(16, ReLU) → Dense(1, Sigmoid)

Aprendizaje:
  - Batch gradient descent (mini-batches de 32)
  - SGD con momentum + learning rate decay
  - Online update: llama a partial_fit() tras cada trade completado
  - Regularización L2 para evitar overfitting con pocos datos

Persistencia:
  - save(path) / load(path) guarda/carga pesos en formato .npz (numpy)
"""

import numpy as np
import json
import os
from typing import Optional


# ── Activaciones ──────────────────────────────────────────────────────────────

def _relu(x):
    return np.maximum(0.0, x)

def _relu_grad(x):
    return (x > 0).astype(np.float32)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

def _sigmoid_grad(x):
    s = _sigmoid(x)
    return s * (1 - s)


# ── Capa Dense ────────────────────────────────────────────────────────────────

class _Dense:
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng     = np.random.default_rng(seed)
        # He initialization para ReLU
        scale   = np.sqrt(2.0 / in_dim)
        self.W  = rng.standard_normal((in_dim, out_dim)).astype(np.float32) * scale
        self.b  = np.zeros(out_dim, dtype=np.float32)
        # Momentum accumulators
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout, lr: float, l2: float, momentum: float):
        dW = self._x.T @ dout + l2 * self.W
        db = dout.sum(axis=0)
        dx = dout @ self.W.T
        # SGD con momentum
        self.vW = momentum * self.vW - lr * dW
        self.vb = momentum * self.vb - lr * db
        self.W  += self.vW
        self.b  += self.vb
        return dx


# ── Red neuronal principal ─────────────────────────────────────────────────────

class TradingNet:
    """
    MLP para predecir P(trade_rentable | features).

    Output: un escalar en [0, 1] donde > 0.5 → señal válida, < 0.5 → filtrar.

    Parámetros
    ----------
    input_dim    : número de features (default: N_FEATURES de feature_extractor)
    hidden1      : neuronas en capa 1 (32)
    hidden2      : neuronas en capa 2 (16)
    lr           : learning rate inicial (0.005)
    lr_decay     : factor de decay por update (0.9999)
    momentum     : momentum SGD (0.9)
    l2           : regularización L2 (0.001)
    dropout_rate : dropout en entrenamiento (0.2)
    min_lr       : learning rate mínimo
    """

    def __init__(self,
                 input_dim: int = 25,
                 hidden1: int = 32,
                 hidden2: int = 16,
                 lr: float = 0.005,
                 lr_decay: float = 0.9999,
                 momentum: float = 0.9,
                 l2: float = 0.001,
                 dropout_rate: float = 0.2,
                 min_lr: float = 1e-5):
        self.input_dim    = input_dim
        self.lr0          = lr
        self.lr           = lr
        self.lr_decay     = lr_decay
        self.momentum     = momentum
        self.l2           = l2
        self.dropout_rate = dropout_rate
        self.min_lr       = min_lr

        self._step        = 0
        self._train_loss  = []

        # Capas
        self.l1   = _Dense(input_dim, hidden1, seed=42)
        self.l2   = _Dense(hidden1,   hidden2, seed=43)
        self.l3   = _Dense(hidden2,   1,       seed=44)
        self.l2_reg = l2   # alias sin conflicto con self.l2 (capa)

    # ── Forward pass ──────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: shape (batch, input_dim) o (input_dim,)
        Retorna: shape (batch,) — probabilidades en [0, 1]
        """
        X = np.atleast_2d(X).astype(np.float32)
        a1 = _relu(self.l1.forward(X))
        a2 = _relu(self.l2.forward(a1))
        out = _sigmoid(self.l3.forward(a2))
        return out.ravel()

    def predict_single(self, x: np.ndarray) -> float:
        """Predice la probabilidad para un único vector de features."""
        return float(self.predict(x[np.newaxis, :])[0])

    # ── Backward pass ─────────────────────────────────────────────────

    def _forward_train(self, X: np.ndarray):
        """Forward con dropout para entrenamiento."""
        a1_pre  = self.l1.forward(X)
        a1      = _relu(a1_pre)
        # Dropout capa 1
        mask1   = (np.random.rand(*a1.shape) > self.dropout_rate).astype(np.float32)
        a1      = a1 * mask1 / (1.0 - self.dropout_rate + 1e-8)
        a2_pre  = self.l2.forward(a1)
        a2      = _relu(a2_pre)
        out_pre = self.l3.forward(a2)
        out     = _sigmoid(out_pre)
        return out, a1_pre, a1, mask1, a2_pre, a2, out_pre

    def _train_step(self, X: np.ndarray, y: np.ndarray):
        """Un paso de backprop sobre un mini-batch."""
        batch_size = X.shape[0]
        out, a1_pre, a1, mask1, a2_pre, a2, out_pre = self._forward_train(X)

        # BCE loss: -[y*log(p) + (1-y)*log(1-p)]
        eps  = 1e-7
        loss = -np.mean(y * np.log(out + eps) + (1 - y) * np.log(1 - out + eps))

        # Gradientes backward
        d_out  = (out - y.reshape(-1, 1)) / batch_size   # (batch, 1)
        d_a2   = self.l3.backward(d_out, self.lr, self.l2_reg, self.momentum)
        d_a2  *= _relu_grad(a2_pre)
        d_a1   = self.l2.backward(d_a2, self.lr, self.l2_reg, self.momentum)
        d_a1  *= _relu_grad(a1_pre) * mask1 / (1.0 - self.dropout_rate + 1e-8)
        self.l1.backward(d_a1, self.lr, self.l2_reg, self.momentum)

        # LR decay
        self.lr = max(self.min_lr, self.lr * self.lr_decay)
        self._step += 1
        return float(loss)

    # ── API pública de entrenamiento ──────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 30, batch_size: int = 32,
            verbose: bool = False):
        """
        Entrena la red con el dataset completo.

        X: (n, input_dim), y: (n,) con valores 0.0 o 1.0
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        n = len(X)
        for ep in range(epochs):
            idx    = np.random.permutation(n)
            X_shuf = X[idx]
            y_shuf = y[idx]
            losses = []
            for start in range(0, n, batch_size):
                xb = X_shuf[start: start + batch_size]
                yb = y_shuf[start: start + batch_size]
                if len(xb) == 0:
                    continue
                losses.append(self._train_step(xb, yb))
            ep_loss = np.mean(losses)
            self._train_loss.append(ep_loss)
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                preds   = (self.predict(X) > 0.5).astype(float)
                acc     = np.mean(preds == y)
                print(f"  Epoch {ep+1:3d}/{epochs}  loss={ep_loss:.4f}  acc={acc:.3f}  lr={self.lr:.6f}")

    def partial_fit(self, x: np.ndarray, y: float):
        """
        Online update: actualiza la red con un único ejemplo.
        Llamar tras cada trade completado.

        x: (input_dim,), y: 1.0 si rentable, 0.0 si no.
        """
        X = x[np.newaxis, :].astype(np.float32)
        Y = np.array([y], dtype=np.float32)
        return self._train_step(X, Y)

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Métricas de evaluación."""
        X  = np.array(X, dtype=np.float32)
        y  = np.array(y, dtype=np.float32)
        p  = self.predict(X)
        pred_bin = (p > 0.5).astype(float)
        acc      = float(np.mean(pred_bin == y))
        tp       = float(np.sum((pred_bin == 1) & (y == 1)))
        fp       = float(np.sum((pred_bin == 1) & (y == 0)))
        fn       = float(np.sum((pred_bin == 0) & (y == 1)))
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        return {
            "accuracy":  round(acc, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "n_samples": len(y),
            "steps":     self._step,
            "lr":        round(self.lr, 7),
        }

    # ── Persistencia ──────────────────────────────────────────────────

    def save(self, path: str):
        """Guarda pesos en formato .npz + metadatos en JSON."""
        np.savez(path,
                 l1_W=self.l1.W, l1_b=self.l1.b,
                 l1_vW=self.l1.vW, l1_vb=self.l1.vb,
                 l2_W=self.l2.W, l2_b=self.l2.b,
                 l2_vW=self.l2.vW, l2_vb=self.l2.vb,
                 l3_W=self.l3.W, l3_b=self.l3.b,
                 l3_vW=self.l3.vW, l3_vb=self.l3.vb)
        meta = {
            "input_dim":    self.input_dim,
            "hidden1":      self.l1.W.shape[1],
            "hidden2":      self.l2.W.shape[1],
            "lr":           self.lr,
            "lr0":          self.lr0,
            "lr_decay":     self.lr_decay,
            "momentum":     self.momentum,
            "l2":           self.l2_reg if hasattr(self, "l2_reg") else self.l2,
            "dropout_rate": self.dropout_rate,
            "min_lr":       self.min_lr,
            "step":         self._step,
            "train_loss":   self._train_loss[-50:],  # últimas 50 épocas
        }
        meta_path = path.replace(".npz", "") + "_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TradingNet":
        """Carga pesos desde .npz."""
        npz_path  = path if path.endswith(".npz") else path + ".npz"
        meta_path = npz_path.replace(".npz", "_meta.json")

        data = np.load(npz_path)

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}

        net = cls(
            input_dim    = meta.get("input_dim", data["l1_W"].shape[0]),
            hidden1      = meta.get("hidden1",   data["l1_W"].shape[1]),
            hidden2      = meta.get("hidden2",   data["l2_W"].shape[1]),
            lr           = meta.get("lr",        0.005),
            lr_decay     = meta.get("lr_decay",  0.9999),
            momentum     = meta.get("momentum",  0.9),
            l2           = meta.get("l2",        0.001),
            dropout_rate = meta.get("dropout_rate", 0.2),
            min_lr       = meta.get("min_lr",    1e-5),
        )
        net.l1.W  = data["l1_W"]; net.l1.b  = data["l1_b"]
        net.l1.vW = data["l1_vW"]; net.l1.vb = data["l1_vb"]
        net.l2.W  = data["l2_W"]; net.l2.b  = data["l2_b"]
        net.l2.vW = data["l2_vW"]; net.l2.vb = data["l2_vb"]
        net.l3.W  = data["l3_W"]; net.l3.b  = data["l3_b"]
        net.l3.vW = data["l3_vW"]; net.l3.vb = data["l3_vb"]
        net._step       = meta.get("step", 0)
        net._train_loss = meta.get("train_loss", [])
        return net

    def __repr__(self):
        h1, h2 = self.l1.W.shape[1], self.l2.W.shape[1]
        return (f"TradingNet({self.input_dim}→{h1}→{h2}→1 | "
                f"steps={self._step} lr={self.lr:.6f})")
