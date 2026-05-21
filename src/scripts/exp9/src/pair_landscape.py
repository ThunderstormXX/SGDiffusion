from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .geodesic_mds import GeodesicMDS
from .loss import BCELoss
from .pair_landscape_plots import PairLandscapePlots
from .pair_motion_zoom import PairMotionZoomPlot
from .pair_selector import TransitionPairSelector
from .sampler import ParameterSampler


class PairLandscapeStage:
    def __init__(self, cfg: ExperimentConfig, loss: BCELoss, sampler: ParameterSampler, store):
        self.cfg = cfg
        self.loss = loss
        self.sampler = sampler
        self.store = store
        self.plots = PairLandscapePlots()

    def run(
        self, reps: np.ndarray, runs: list[dict], X: np.ndarray, y: np.ndarray,
        valid_ids: np.ndarray | None = None,
    ) -> dict:
        i, j, reason = TransitionPairSelector().pick(reps, runs, valid_ids)
        a, b = reps[i], reps[j]
        radius = max(self.cfg.probe_radius, self.cfg.pair_radius_factor * np.linalg.norm(a - b))
        cloud = self.sampler.segment_tube(a, b, self.cfg.pair_neighborhood_samples, radius)
        traj, run_ids = self._trajectory_points(runs, i, j)
        points = np.vstack([cloud, a[None], b[None], traj])
        labels = np.r_[np.zeros(len(cloud), int), 1, 2, np.full(len(traj), 3, int)]
        point_run_ids = np.r_[np.full(len(cloud) + 2, -1, int), run_ids]
        losses = self.loss.batch_values(points, X, y, "pair landscape loss")
        Z = GeodesicMDS(self.cfg.geodesic_neighbors).fit_transform(points, 2)
        self.store.npz("pair_geodesic_landscape.npz", points=points, Z=Z, losses=losses,
                       labels=labels, run_ids=point_run_ids)
        self.plots.mds2d(Z, losses, labels, self.store.path("pair_geodesic_mds.png"), (i, j))
        self.plots.movement2d(Z, labels, self.store.path("pair_geodesic_motion.png"), (i, j))
        self.plots.surface3d(Z, losses, labels, self.store.path("pair_landscape_3d.png"), (i, j))
        zoom = self.store.path("pair_geodesic_motion_zoom.png")
        PairMotionZoomPlot().plot(Z, labels, point_run_ids, zoom, (i, j))
        return {"pair": [i, j], "reason": reason, "radius": float(radius), "trajectory_points": int(len(traj))}

    def _trajectory_points(self, runs: list[dict], i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        chunks, ids = [], []
        for run_id, run in enumerate(runs):
            mask = np.isin(run["clusters"], [i, j])
            if np.any(mask):
                chunks.append(run["Theta"][mask])
                ids.append(np.full(int(mask.sum()), run_id, dtype=int))
        if chunks:
            return np.vstack(chunks), np.concatenate(ids)
        return np.empty((0, self.sampler.dim), dtype=float), np.empty(0, dtype=int)
