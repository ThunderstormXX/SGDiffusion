from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .loss import BCELoss
from .minima import MinimaClusterer, MinimaClusters
from .optim import LocalOptimizer
from .sampler import ParameterSampler


class SegmentProbe:
    def __init__(self, loss: BCELoss, cfg: ExperimentConfig):
        self.loss = loss
        self.cfg = cfg

    def scan(self, a: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict:
        ts = np.linspace(-0.2, 1.2, self.cfg.segment_points)
        Theta = np.array([(1.0 - t) * a + t * b for t in ts])
        losses = self.loss.batch_values(Theta, X, y, "segment loss")
        edge = max(losses[np.argmin(np.abs(ts))], losses[np.argmin(np.abs(ts - 1.0))])
        return {"t": ts, "losses": losses, "barrier": float(losses.max() - edge)}

    def closest_pairs(self, clusters: MinimaClusters, n: int = 8) -> list[tuple[int, int, float]]:
        pairs = []
        for i in range(len(clusters.reps)):
            for j in range(i + 1, len(clusters.reps)):
                dist = float(np.linalg.norm(clusters.reps[i] - clusters.reps[j]))
                pairs.append((i, j, dist))
        return sorted(pairs, key=lambda x: x[2])[:n]


class AttractionProbe:
    def __init__(self, opt: LocalOptimizer, sampler: ParameterSampler, clusterer: MinimaClusterer):
        self.opt = opt
        self.sampler = sampler
        self.clusterer = clusterer

    def run(self, clusters: MinimaClusters, X: np.ndarray, y: np.ndarray, radius: float, n: int) -> dict:
        rows, losses = [], []
        for source, center in enumerate(clusters.reps):
            starts = self.sampler.ball(center, n, radius)
            for start in starts:
                theta = self.opt.polish(start, X, y)
                value = self.opt.loss.value(theta, X, y)
                target = self.clusterer.assign_to_reps(theta, clusters.reps)
                rows.append((source, target))
                losses.append(value)
        return {"edges": np.array(rows, dtype=int), "losses": np.array(losses)}
