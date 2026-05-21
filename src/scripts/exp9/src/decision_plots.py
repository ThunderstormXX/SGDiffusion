from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .minima import MinimaClusters
from .model import SmoothTinyMLP


class DecisionBoundaryPlots:
    def __init__(self, model: SmoothTinyMLP):
        self.model = model

    def plot(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray, path, title: str) -> None:
        x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
        y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
        gx, gy = np.meshgrid(np.linspace(x_min, x_max, 180), np.linspace(y_min, y_max, 180))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        pred = self.model.forward(theta, grid).reshape(gx.shape)

        plt.figure(figsize=(5, 4))
        plt.contourf(gx, gy, pred, levels=35, alpha=0.85)
        plt.contour(gx, gy, pred, levels=[0.5], colors="black", linewidths=1.5)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=24, edgecolor="k", linewidth=0.25)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()

    def top_clusters(self, clusters: MinimaClusters, X: np.ndarray, y: np.ndarray, root, n: int) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for idx, theta in enumerate(clusters.reps[:n]):
            title = f"cluster {idx}: loss={clusters.rep_losses[idx]:.5f}"
            self.plot(theta, X, y, root / f"decision_cluster_{idx:03d}.png", title)

