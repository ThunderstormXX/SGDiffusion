from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .loss import BCELoss
from .symmetry import HiddenPermutationSymmetry


class GDValidator:
    def __init__(self, cfg: ExperimentConfig, loss: BCELoss, sym: HiddenPermutationSymmetry):
        self.cfg = cfg
        self.loss = loss
        self.sym = sym

    def run(self, starts: dict[str, np.ndarray], reps: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict:
        rows, traces = [], {}
        for idx, (name, start) in enumerate(starts.items()):
            final, trace = self._gd(start, reps, X, y)
            traces[f"trace_{idx}"] = trace
            val, grad = self.loss.value_grad(final, X, y)
            cid, dist = self._nearest(final, reps)
            rows.append((name, cid, dist, float(val), float(np.linalg.norm(grad)), len(trace)))
        return {"rows": rows, "traces": traces}

    def _gd(self, theta0: np.ndarray, reps: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = theta0.copy()
        trace = []
        for step in range(self.cfg.gd_validate_steps + 1):
            val, grad = self.loss.value_grad(theta, X, y)
            if step % 25 == 0 or step == self.cfg.gd_validate_steps:
                cid, dist = self._nearest(theta, reps)
                trace.append((step, float(val), float(np.linalg.norm(grad)), cid, dist))
            if np.linalg.norm(grad) < 1e-8:
                break
            theta -= self.cfg.gd_validate_lr * grad
        return theta, np.array(trace, dtype=float)

    def _nearest(self, theta: np.ndarray, reps: np.ndarray) -> tuple[int, float]:
        distances = np.array([self.sym.distance(theta, rep) for rep in reps])
        idx = int(np.argmin(distances))
        return idx, float(distances[idx])

