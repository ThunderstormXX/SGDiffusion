from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .minima import MinimaClusters
from .transition_plots import TransitionPlots
from .walk_plots import TrajectoryPlots
from .walker import SGDWalker


class SGDWalkStage:
    def __init__(self, cfg: ExperimentConfig, walker: SGDWalker, plots: TrajectoryPlots, store):
        self.cfg = cfg
        self.walker = walker
        self.plots = plots
        self.store = store

    def run(self, clusters: MinimaClusters, X, y) -> tuple[list[dict], list[dict]]:
        starts = self._starts(clusters.reps)
        runs = [self.walker.run(start, clusters.reps, X, y, i) for i, start in enumerate(starts)]
        summaries = []
        for i, run in enumerate(runs):
            self.store.npz(f"sgd_walk_{i}.npz", **run)
            summaries.append(self._summary(i, run))
        self.store.json("sgd_walk_summary.json", {"runs": summaries})
        self.plots.timeline(runs, self.store.path("sgd_walk_timeline.png"))
        self.plots.mds_path(clusters.reps, runs, self.store.path("sgd_walk_mds.png"))
        trans = TransitionPlots()
        trans.graph(clusters.reps, runs, self.store.path("cluster_transition_paths.png"))
        trans.matrix(runs, len(clusters.reps), self.store.path("cluster_transition_matrix.png"))
        return runs, summaries

    def _starts(self, reps: np.ndarray) -> list[np.ndarray]:
        starts = [reps[0]]
        if len(reps) > 1 and self.cfg.walk_runs > 1:
            i, j = self._closest_pair(reps)
            starts.append(0.5 * (reps[i] + reps[j]))
        while len(starts) < self.cfg.walk_runs:
            starts.append(reps[len(starts) % len(reps)])
        return starts[: self.cfg.walk_runs]

    def _closest_pair(self, reps: np.ndarray) -> tuple[int, int]:
        best = (0, 1, float("inf"))
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                dist = float(np.linalg.norm(reps[i] - reps[j]))
                best = (i, j, dist) if dist < best[2] else best
        return best[0], best[1]

    def _summary(self, run_id: int, run: dict) -> dict:
        return {
            "run": run_id,
            "start_cluster": int(run["clusters"][0]),
            "end_cluster": int(run["clusters"][-1]),
            "jump_count": int(run["jump_count"]),
            "jump_steps": run["jump_steps"].astype(int).tolist(),
            "unique_clusters": np.unique(run["clusters"]).astype(int).tolist(),
            "final_loss": float(run["losses"][-1]),
        }
