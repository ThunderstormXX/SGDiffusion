from __future__ import annotations

import numpy as np

from .loss import BCELoss


class HessianAnalyzer:
    def __init__(self, loss: BCELoss, eps: float = 1e-4):
        self.loss = loss
        self.eps = eps

    def matrix(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        d = len(theta)
        H = np.empty((d, d), dtype=np.float64)
        for k in range(d):
            step = np.zeros(d)
            step[k] = self.eps
            _, gp = self.loss.value_grad(theta + step, X, y)
            _, gm = self.loss.value_grad(theta - step, X, y)
            H[:, k] = (gp - gm) / (2.0 * self.eps)
        return 0.5 * (H + H.T)

    def spectrum(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict:
        eigvals = np.linalg.eigvalsh(self.matrix(theta, X, y))
        pos = eigvals[eigvals > 1e-6]
        neg = eigvals[eigvals < -1e-6]
        return {
            "eigvals": eigvals.tolist(),
            "min_eig": float(eigvals[0]),
            "max_eig": float(eigvals[-1]),
            "n_pos": int(len(pos)),
            "n_neg": int(len(neg)),
            "n_flat": int(len(eigvals) - len(pos) - len(neg)),
            "condition_pos": float(pos[-1] / pos[0]) if len(pos) else float("nan"),
        }

