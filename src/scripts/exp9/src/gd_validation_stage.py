from __future__ import annotations

import numpy as np

from .gd_validator import GDValidator
from .minima import MinimaClusters
from .pair_selector import TransitionPairSelector


class GDValidationStage:
    def __init__(self, validator: GDValidator, store):
        self.validator = validator
        self.store = store

    def run(self, clusters: MinimaClusters, runs: list[dict], X, y, valid_ids=None) -> dict:
        i, j, reason = TransitionPairSelector().pick(clusters.reps, runs, valid_ids)
        starts = self._starts(clusters.reps, runs, i, j)
        result = self.validator.run(starts, clusters.reps, X, y)
        self.store.npz("gd_validation_traces.npz", **result["traces"])
        rows = [self._row_dict(row) for row in result["rows"]]
        payload = {"pair": [i, j], "reason": reason, "rows": rows}
        self.store.json("gd_validation_summary.json", payload)
        return payload

    def _starts(self, reps: np.ndarray, runs: list[dict], i: int, j: int) -> dict[str, np.ndarray]:
        starts = {"pair_midpoint": 0.5 * (reps[i] + reps[j])}
        for run_id, run in enumerate(runs):
            mask = np.isin(run["clusters"], [i, j])
            if np.any(mask):
                pts = run["Theta"][mask]
                starts[f"run{run_id}_first"] = pts[0]
                starts[f"run{run_id}_middle"] = pts[len(pts) // 2]
                starts[f"run{run_id}_last"] = pts[-1]
        return starts

    def _row_dict(self, row: tuple) -> dict:
        name, cid, dist, loss, grad, n_trace = row
        return {
            "start": name,
            "final_nearest_cluster": int(cid),
            "final_distance": float(dist),
            "final_loss": float(loss),
            "final_grad_norm": float(grad),
            "trace_points": int(n_trace),
        }
