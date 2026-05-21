from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .config import ExperimentConfig


class ParameterSampler:
    def __init__(self, cfg: ExperimentConfig, dim: int):
        self.cfg = cfg
        self.dim = dim
        self.rng = np.random.default_rng(cfg.seed + 1000)

    def sobol_box(self, n: int, scale: float) -> np.ndarray:
        m = int(np.ceil(np.log2(max(n, 2))))
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.cfg.seed + 1001)
        return (2.0 * sampler.random_base2(m)[:n] - 1.0) * scale

    def ball(self, center: np.ndarray, n: int, radius: float) -> np.ndarray:
        z = self.rng.normal(size=(n, self.dim))
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
        r = radius * self.rng.random((n, 1)) ** (1.0 / self.dim)
        return center[None, :] + r * z

    def segment_tube(self, a: np.ndarray, b: np.ndarray, n: int, radius: float) -> np.ndarray:
        t = self.rng.uniform(0.0, 1.0, size=(n, 1))
        centers = (1.0 - t) * a[None, :] + t * b[None, :]
        z = self.rng.normal(size=(n, self.dim))
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
        r = radius * self.rng.random((n, 1)) ** (1.0 / self.dim)
        return centers + r * z

