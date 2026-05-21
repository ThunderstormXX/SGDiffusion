from __future__ import annotations

import numpy as np

from .minima import MinimaClusters


class TextReport:
    def clusters(self, clusters: MinimaClusters, grad_norms: np.ndarray) -> str:
        lines = ["# exp9 local minima report", ""]
        lines.append(f"raw converged points: {len(clusters.assignments)}")
        lines.append(f"symmetry-aware clusters: {len(clusters.reps)}")
        lines.append("")
        lines.append("| id | count | loss | grad_norm | norm |")
        lines.append("|---:|---:|---:|---:|---:|")
        for i, idx in enumerate(clusters.best_indices):
            lines.append(
                f"| {i} | {clusters.counts[i]} | {clusters.rep_losses[i]:.8f} | "
                f"{grad_norms[idx]:.3e} | {np.linalg.norm(clusters.reps[i]):.3f} |"
            )
        return "\n".join(lines) + "\n"

    def summary(self, clusters: MinimaClusters, spectra: list[dict], barriers: list[dict]) -> dict:
        return {
            "n_raw_minima": int(len(clusters.assignments)),
            "n_clusters": int(len(clusters.reps)),
            "cluster_counts": clusters.counts.tolist(),
            "cluster_losses": clusters.rep_losses.tolist(),
            "hessian": spectra,
            "segment_barriers": barriers,
        }

