from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS


class TrajectoryPlots:
    def timeline(self, runs: list[dict], path) -> None:
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        for run_id, run in enumerate(runs):
            label = f"run {run_id}"
            axes[0].plot(run["steps"], run["clusters"], marker=".", label=label)
            axes[1].plot(run["steps"], run["losses"], label=label)
            axes[2].plot(run["steps"], run["distances"], label=label)
        axes[0].set_ylabel("nearest min id")
        axes[1].set_ylabel("full loss")
        axes[2].set_ylabel("distance")
        axes[2].set_xlabel("SGD step")
        axes[0].legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    def mds_path(self, reps: np.ndarray, runs: list[dict], path) -> None:
        path_points = np.vstack([run["Theta"] for run in runs])
        all_points = np.vstack([reps, path_points])
        Z = MDS(n_components=2, random_state=5, normalized_stress="auto").fit_transform(all_points)
        Z_reps = Z[: len(reps)]
        offset = len(reps)
        plt.figure(figsize=(7, 5))
        plt.scatter(Z_reps[:, 0], Z_reps[:, 1], s=170, marker="*", edgecolor="k", label="minima")
        for idx, z in enumerate(Z_reps):
            plt.text(z[0], z[1], f" {idx}", weight="bold")
        for run_id, run in enumerate(runs):
            Z_run = Z[offset : offset + len(run["Theta"])]
            offset += len(run["Theta"])
            plt.plot(Z_run[:, 0], Z_run[:, 1], marker=".", ms=3, label=f"SGD {run_id}")
        plt.title("SGD trajectory projected with found minima")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()

