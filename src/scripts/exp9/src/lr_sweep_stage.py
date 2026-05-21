from __future__ import annotations

import math

import numpy as np

from .config import ExperimentConfig
from .lr_sweep_plots import LRSweepPlots
from .optim import LocalOptimizer
from .walker import SGDWalker


class LRSweepStage:
    def __init__(self, cfg: ExperimentConfig, walker: SGDWalker, opt: LocalOptimizer, store):
        self.cfg = cfg
        self.walker = walker
        self.opt = opt
        self.store = store
        self.plots = LRSweepPlots()

    def run(self, reps: np.ndarray, confirmed_ids: np.ndarray, X, y) -> dict:
        valid_ids = confirmed_ids if len(confirmed_ids) else np.arange(len(reps))
        valid_reps = reps[valid_ids]
        start = self._fixed_start(reps.shape[1])
        runs, summaries = [], []
        lrs = list(self.cfg.lr_sweep)
        for idx, lr in enumerate(lrs):
            run = self.walker.run(
                start, valid_reps, X, y, 700 + idx, lr=lr,
                steps=self.cfg.lr_sweep_steps, eval_every=self.cfg.lr_sweep_eval_every,
            )
            run["clusters"] = valid_ids[run["clusters"]]
            polished = self.opt.polish(run["Theta"][-1], X, y)
            run["polished_theta"] = polished
            runs.append(run)
            self.store.npz(f"sgd_lr_sweep_{idx:02d}.npz", **run)
            summaries.append(self._summary(lr, run, reps, valid_ids, X, y))
        visited_ids = self._visited_ids(runs, summaries)
        payload = {"start": "fixed_random_init", "visited_cluster_ids": visited_ids.tolist(), "runs": summaries}
        self.store.json("sgd_lr_sweep_summary.json", payload)
        self.plots.mds(
            reps[visited_ids], visited_ids, runs, lrs,
            self.cfg.geodesic_neighbors, self.store.path("sgd_lr_sweep_mds.png"),
        )
        self.plots.local_mds(
            runs, lrs, summaries, self.cfg.geodesic_neighbors,
            self.store.path("sgd_lr_sweep_local_mds.png"),
        )
        self.plots.timeline(runs, lrs, self.store.path("sgd_lr_sweep_timeline.png"))
        return payload

    def _fixed_start(self, dim: int) -> np.ndarray:
        rng = np.random.default_rng(self.cfg.seed + 7000)
        scale = self.cfg.init_scale / math.sqrt(dim)
        return rng.normal(scale=scale, size=dim)

    def _summary(self, lr: float, run: dict, reps: np.ndarray, valid_ids: np.ndarray, X, y) -> dict:
        finite = bool(np.all(np.isfinite(run["losses"])))
        polished_loss, polished_grad = self.walker.loss.value_grad(run["polished_theta"], X, y)
        polished_cluster, polished_dist = self._nearest(run["polished_theta"], reps, valid_ids)
        sgd_to_polished = self.walker.sym.distance(run["Theta"][-1], run["polished_theta"])
        ratio = sgd_to_polished / max(polished_dist, 1e-12)
        return {
            "lr": float(lr),
            "final_cluster": int(run["clusters"][-1]),
            "final_loss": float(run["losses"][-1]),
            "polished_cluster": int(polished_cluster),
            "polished_loss": float(polished_loss),
            "polished_distance": float(polished_dist),
            "polished_grad_norm": float(np.linalg.norm(polished_grad)),
            "sgd_to_polished_distance": float(sgd_to_polished),
            "distance_ratio_sgd_over_estimate": float(ratio),
            "distance_ratio_orders": float(np.log10(max(ratio, 1e-12))),
            "min_loss": float(np.min(run["losses"])),
            "jump_count": int(run["jump_count"]),
            "unique_clusters": np.unique(run["clusters"]).astype(int).tolist(),
            "finite": finite,
            "checkpoints": int(len(run["steps"])),
        }

    def _nearest(self, theta: np.ndarray, reps: np.ndarray, valid_ids: np.ndarray) -> tuple[int, float]:
        distances = np.array([self.walker.sym.distance(theta, reps[i]) for i in valid_ids])
        pos = int(np.argmin(distances))
        return int(valid_ids[pos]), float(distances[pos])

    def _visited_ids(self, runs: list[dict], summaries: list[dict]) -> np.ndarray:
        ids = set()
        for run in runs:
            ids.update(int(x) for x in np.unique(run["clusters"]))
        ids.update(int(row["polished_cluster"]) for row in summaries)
        return np.array(sorted(ids), dtype=int)
