from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation


class PairLandscapePlots:
    def mds2d(self, Z: np.ndarray, losses: np.ndarray, labels: np.ndarray, path, pair: tuple[int, int]) -> None:
        plt.figure(figsize=(7, 5))
        sc = plt.scatter(Z[:, 0], Z[:, 1], c=losses, s=12, cmap="viridis", alpha=0.75)
        plt.colorbar(sc, label="loss")
        self._draw_markers(Z, labels)
        plt.title(f"Geodesic MDS neighborhood: clusters {pair[0]} and {pair[1]}")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    def surface3d(self, Z: np.ndarray, losses: np.ndarray, labels: np.ndarray, path, pair: tuple[int, int]) -> None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        mask = labels == 0
        tri = Triangulation(Z[mask, 0], Z[mask, 1])
        ax.plot_trisurf(tri, losses[mask], cmap="viridis", alpha=0.72, linewidth=0.1)
        lift = 0.04 * max(float(losses.max() - losses.min()), 1e-12)
        self._plot_trajectory(ax, Z, losses, labels, lift)
        reps = np.where((labels == 1) | (labels == 2))[0]
        ax.scatter(Z[reps, 0], Z[reps, 1], losses[reps] + lift, s=90, c="red", marker="*", depthshade=False)
        ax.set_title(f"3D loss canvas between clusters {pair[0]} and {pair[1]}")
        ax.set_xlabel("geodesic MDS 1")
        ax.set_ylabel("geodesic MDS 2")
        ax.set_zlabel("loss")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def movement2d(self, Z: np.ndarray, labels: np.ndarray, path, pair: tuple[int, int]) -> None:
        traj = np.where(labels == 3)[0]
        plt.figure(figsize=(7, 5))
        cloud = labels == 0
        plt.scatter(Z[cloud, 0], Z[cloud, 1], s=10, c="lightgray", alpha=0.45)
        self._draw_markers(Z, labels)
        if len(traj):
            time = np.arange(len(traj))
            sc = plt.scatter(Z[traj, 0], Z[traj, 1], c=time, s=24, cmap="plasma", zorder=4)
            plt.scatter(Z[traj[0], 0], Z[traj[0], 1], c="lime", s=95, edgecolor="k", label="start")
            plt.scatter(Z[traj[-1], 0], Z[traj[-1], 1], c="red", s=95, edgecolor="k", label="end")
            self._arrows_2d(Z[traj])
            plt.colorbar(sc, label="trajectory checkpoint order")
        plt.title(f"Point motion in two-minimum neighborhood: {pair[0]} <-> {pair[1]}")
        if len(traj):
            plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    def _draw_markers(self, Z: np.ndarray, labels: np.ndarray) -> None:
        reps = np.where((labels == 1) | (labels == 2))[0]
        traj = np.where(labels == 3)[0]
        plt.scatter(Z[reps, 0], Z[reps, 1], s=180, marker="*", edgecolor="k", c="red")
        if len(traj):
            plt.plot(Z[traj, 0], Z[traj, 1], c="white", lw=3, alpha=0.8)
            plt.plot(Z[traj, 0], Z[traj, 1], c="black", lw=1.4, marker=".", ms=3)

    def _plot_trajectory(self, ax, Z: np.ndarray, losses: np.ndarray, labels: np.ndarray, lift: float) -> None:
        traj = np.where(labels == 3)[0]
        if len(traj):
            ax.plot(Z[traj, 0], Z[traj, 1], losses[traj] + lift, c="red", lw=3.0, marker=".", ms=4)

    def _arrows_2d(self, pts: np.ndarray) -> None:
        stride = max(1, len(pts) // 8)
        for k in range(0, len(pts) - 1, stride):
            plt.annotate(
                "",
                xy=pts[k + 1],
                xytext=pts[k],
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.4},
            )
