from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .model import SmoothTinyMLP


class BCELoss:
    def __init__(self, model: SmoothTinyMLP, cfg: ExperimentConfig):
        self.model = model
        self.cfg = cfg

    def value(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        p = self.model.forward(theta, X)
        eps = 1e-12
        ce = -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        return float(ce + 0.5 * self.cfg.weight_decay * theta @ theta)

    def value_grad(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        return self.model.loss_grad(theta, X, y, self.cfg.weight_decay)

    def batch_values(self, Theta: np.ndarray, X: np.ndarray, y: np.ndarray, desc: str) -> np.ndarray:
        losses = np.empty(len(Theta), dtype=np.float64)
        eps = 1e-12
        for start in tqdm(range(0, len(Theta), self.cfg.batch_size), desc=desc):
            stop = min(start + self.cfg.batch_size, len(Theta))
            B = Theta[start:stop]
            p = self.model.forward_batch(B, X)
            ce = -np.mean(y[None, :] * np.log(p + eps), axis=1)
            ce -= np.mean((1.0 - y)[None, :] * np.log(1.0 - p + eps), axis=1)
            reg = 0.5 * self.cfg.weight_decay * np.sum(B * B, axis=1)
            losses[start:stop] = ce + reg
        return losses

