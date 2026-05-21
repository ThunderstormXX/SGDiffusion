from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig


@dataclass
class BinaryPointDataset:
    cfg: ExperimentConfig
    X: np.ndarray | None = None
    y: np.ndarray | None = None

    def build(self) -> BinaryPointDataset:
        X, y = make_moons(
            n_samples=self.cfg.n_samples,
            noise=self.cfg.moon_noise,
            random_state=self.cfg.seed,
        )
        self.X = StandardScaler().fit_transform(X).astype(np.float64)
        self.y = y.astype(np.float64)
        return self

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.X is None or self.y is None:
            raise RuntimeError("dataset is not built")
        return self.X, self.y
