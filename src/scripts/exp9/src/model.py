from __future__ import annotations

import numpy as np
from scipy.special import expit

from .config import ExperimentConfig


class SmoothTinyMLP:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.input_dim = 2
        self.hidden_dim = cfg.hidden_dim

    @property
    def dim(self) -> int:
        return self.hidden_dim * self.input_dim + 2 * self.hidden_dim

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self.hidden_dim
        W1 = theta[: 2 * h].reshape(h, 2)
        b1 = theta[2 * h : 3 * h]
        W2 = theta[3 * h : 4 * h]
        return W1, b1, W2

    def unpack_batch(self, Theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = self.hidden_dim
        W1 = Theta[:, : 2 * h].reshape(len(Theta), h, 2)
        b1 = Theta[:, 2 * h : 3 * h]
        W2 = Theta[:, 3 * h : 4 * h]
        return W1, b1, W2

    def forward(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        W1, b1, W2 = self.unpack(theta)
        hidden = expit(X @ W1.T + b1)
        return expit(hidden @ W2)

    def forward_batch(self, Theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        W1, b1, W2 = self.unpack_batch(Theta)
        z1 = np.einsum("ni,bhi->bnh", X, W1) + b1[:, None, :]
        hidden = expit(z1)
        return expit(np.einsum("bnh,bh->bn", hidden, W2))

    def loss_grad(
        self, theta: np.ndarray, X: np.ndarray, y: np.ndarray, weight_decay: float
    ) -> tuple[float, np.ndarray]:
        W1, b1, W2 = self.unpack(theta)
        n = len(X)
        a1 = X @ W1.T + b1
        hidden = expit(a1)
        pred = expit(hidden @ W2)
        eps = 1e-12
        ce = -np.mean(y * np.log(pred + eps) + (1.0 - y) * np.log(1.0 - pred + eps))
        loss = float(ce + 0.5 * weight_decay * theta @ theta)
        dlogit = (pred - y) / n
        grad_W2 = dlogit @ hidden
        dhidden = dlogit[:, None] * W2[None, :]
        da1 = dhidden * hidden * (1.0 - hidden)
        grad_W1 = da1.T @ X
        grad_b1 = da1.sum(axis=0)
        grad = np.concatenate([grad_W1.ravel(), grad_b1, grad_W2])
        return loss, grad + weight_decay * theta

    def accuracy(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((self.forward(theta, X) >= 0.5) == y))

