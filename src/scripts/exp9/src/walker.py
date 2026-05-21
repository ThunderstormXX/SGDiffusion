from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .loss import BCELoss
from .symmetry import HiddenPermutationSymmetry


class SGDWalker:
    def __init__(self, loss: BCELoss, sym: HiddenPermutationSymmetry, cfg: ExperimentConfig):
        self.loss = loss
        self.sym = sym
        self.cfg = cfg

    def run(
        self, start: np.ndarray, reps: np.ndarray, X: np.ndarray, y: np.ndarray, run_id: int,
        lr: float | None = None, steps: int | None = None, eval_every: int | None = None,
    ) -> dict:
        rng = np.random.default_rng(self.cfg.seed + 3000 + run_id)
        theta = start + rng.normal(scale=self.cfg.walk_start_noise, size=start.shape)
        lr = self.cfg.walk_lr if lr is None else lr
        n_steps = self.cfg.walk_steps if steps is None else steps
        eval_every = self.cfg.walk_eval_every if eval_every is None else eval_every
        step_ids, values, clusters, distances, thetas = [], [], [], [], []
        for step in range(n_steps + 1):
            if step % eval_every == 0:
                cid, dist = self.nearest(theta, reps)
                step_ids.append(step)
                values.append(self.loss.value(theta, X, y))
                clusters.append(cid)
                distances.append(dist)
                thetas.append(theta.copy())
            if step == n_steps:
                break
            batch = rng.integers(0, len(X), size=self.cfg.walk_batch_size)
            _, grad = self.loss.value_grad(theta, X[batch], y[batch])
            theta = theta - lr * grad
            if not np.all(np.isfinite(theta)):
                break
        return self._pack(step_ids, values, clusters, distances, thetas)

    def nearest(self, theta: np.ndarray, reps: np.ndarray) -> tuple[int, float]:
        distances = np.array([self.sym.distance(theta, rep) for rep in reps])
        idx = int(np.argmin(distances))
        return idx, float(distances[idx])

    def _pack(self, steps: list, values: list, clusters: list, distances: list, thetas: list) -> dict:
        ids = np.array(clusters, dtype=int)
        jumps = np.flatnonzero(ids[1:] != ids[:-1])
        return {
            "steps": np.array(steps, dtype=int),
            "losses": np.array(values, dtype=float),
            "clusters": ids,
            "distances": np.array(distances, dtype=float),
            "Theta": np.array(thetas, dtype=float),
            "jump_count": int(len(jumps)),
            "jump_steps": np.array(steps, dtype=int)[jumps + 1],
        }
