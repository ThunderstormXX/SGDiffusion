from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .geodesic_mds import GeodesicMDS


class LRSweepPlots:
    def mds(
        self, reps: np.ndarray, cluster_ids: np.ndarray, runs: list[dict],
        lrs: list[float], n_neighbors: int, path,
    ) -> None:
        points = np.vstack([reps, *[run["Theta"] for run in runs]])
        Z = GeodesicMDS(n_neighbors).fit_transform(points)
        fig, ax = plt.subplots(figsize=(8, 6))
        Z_reps = Z[: len(reps)]
        ax.scatter(Z_reps[:, 0], Z_reps[:, 1], s=150, marker="*", c="red", edgecolor="k")
        for cid, z in zip(cluster_ids, Z_reps, strict=False):
            ax.text(z[0], z[1], f" {int(cid)}", weight="bold", fontsize=8)
        offset = len(reps)
        for lr, run in zip(lrs, runs, strict=False):
            Z_run = Z[offset : offset + len(run["Theta"])]
            offset += len(run["Theta"])
            ax.plot(Z_run[:, 0], Z_run[:, 1], marker=".", ms=3, lw=1.8, label=f"lr={lr:g}")
            ax.scatter(Z_run[-1, 0], Z_run[-1, 1], marker="s", s=42, edgecolor="k")
        ax.set_title("SGD LR sweep, geodesic MDS over visited clusters only")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def local_mds(self, runs: list[dict], lrs: list[float], summaries: list[dict], n_neighbors: int, path) -> None:
        points = np.vstack([run["Theta"] for run in runs])
        Z = GeodesicMDS(n_neighbors).fit_transform(points)
        fig, ax = plt.subplots(figsize=(8, 6))
        offset = 0
        for lr, run, summary in zip(lrs, runs, summaries, strict=False):
            Z_run = Z[offset : offset + len(run["Theta"])]
            offset += len(run["Theta"])
            label = f"lr={lr:g} -> c{summary['polished_cluster']}"
            ax.plot(Z_run[:, 0], Z_run[:, 1], marker=".", ms=3, lw=2.0, label=label)
            self._markers(ax, Z_run, summary)
            self._arrows(ax, Z_run)
        ax.set_title("SGD LR sweep, local geodesic MDS of SGD checkpoints only")
        ax.set_xlabel("local geodesic MDS 1")
        ax.set_ylabel("local geodesic MDS 2")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def timeline(self, runs: list[dict], lrs: list[float], path) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for lr, run in zip(lrs, runs, strict=False):
            label = f"lr={lr:g}"
            axes[0].plot(run["steps"], run["clusters"], marker=".", label=label)
            axes[1].plot(run["steps"], run["losses"], label=label)
        axes[0].set_ylabel("nearest confirmed cluster")
        axes[1].set_ylabel("full loss")
        axes[1].set_xlabel("SGD step")
        axes[0].legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _arrows(self, ax, pts: np.ndarray) -> None:
        stride = max(1, len(pts) // 6)
        for k in range(0, len(pts) - 1, stride):
            ax.annotate("", xy=pts[k + 1], xytext=pts[k], arrowprops={"arrowstyle": "->", "lw": 1.2})

    def _markers(self, ax, pts: np.ndarray, summary: dict) -> None:
        ax.scatter(pts[0, 0], pts[0, 1], c="lime", s=70, edgecolor="k", zorder=5)
        ax.scatter(pts[-1, 0], pts[-1, 1], marker="s", c="white", s=70, edgecolor="k", zorder=5)
        ax.text(pts[0, 0], pts[0, 1], " start", fontsize=8, weight="bold")
        ax.text(pts[-1, 0], pts[-1, 1], " last SGD", fontsize=8)
        ax.text(
            pts[-1, 0],
            pts[-1, 1] - 0.25,
            f"polish destination: c{summary['polished_cluster']} (not plotted)",
            fontsize=8,
        )
