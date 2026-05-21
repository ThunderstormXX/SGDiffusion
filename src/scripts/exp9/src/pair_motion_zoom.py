from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


class PairMotionZoomPlot:
    def plot(self, Z: np.ndarray, labels: np.ndarray, run_ids: np.ndarray, path, pair: tuple[int, int]) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        self._panel(axes[0], Z, labels, run_ids, zoom=False)
        self._panel(axes[1], Z, labels, run_ids, zoom=True)
        axes[0].set_title("full geodesic-MDS neighborhood")
        axes[1].set_title("zoom on SGD trajectory")
        fig.suptitle(f"SGD motion between clusters {pair[0]} and {pair[1]}")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _panel(self, ax, Z: np.ndarray, labels: np.ndarray, run_ids: np.ndarray, zoom: bool) -> None:
        cloud = labels == 0
        reps = (labels == 1) | (labels == 2)
        ax.scatter(Z[cloud, 0], Z[cloud, 1], s=9, c="lightgray", alpha=0.42)
        ax.scatter(Z[reps, 0], Z[reps, 1], s=170, marker="*", c="red", edgecolor="k", zorder=4)
        self._paths(ax, Z, labels, run_ids)
        ax.set_xlabel("geodesic MDS 1")
        ax.set_ylabel("geodesic MDS 2")
        if zoom:
            self._zoom_limits(ax, Z[labels == 3])

    def _paths(self, ax, Z: np.ndarray, labels: np.ndarray, run_ids: np.ndarray) -> None:
        traj_runs = [int(x) for x in np.unique(run_ids[labels == 3]) if x >= 0]
        for pos, run_id in enumerate(traj_runs):
            idx = np.where((labels == 3) & (run_ids == run_id))[0]
            pts = Z[idx]
            color = f"C{pos % 10}"
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=3.2, marker="o", ms=4, zorder=5)
            ax.scatter(pts[0, 0], pts[0, 1], c="lime", s=95, edgecolor="k", zorder=6)
            ax.scatter(pts[-1, 0], pts[-1, 1], c="black", s=70, marker="s", zorder=6)
            self._arrows(ax, pts, color)

    def _arrows(self, ax, pts: np.ndarray, color: str) -> None:
        if len(pts) < 2:
            return
        stride = max(1, len(pts) // 10)
        for k in range(0, len(pts) - 1, stride):
            ax.annotate(
                "",
                xy=pts[k + 1],
                xytext=pts[k],
                arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0, "mutation_scale": 13},
                zorder=7,
            )

    def _zoom_limits(self, ax, pts: np.ndarray) -> None:
        if len(pts) == 0:
            return
        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        span = np.maximum(maxs - mins, 1e-3)
        pad = 0.45 * span + 0.015
        ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
        ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])

