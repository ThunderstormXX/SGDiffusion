from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS


class LocalMinimaPlots:
    def mds(
        self, reps: np.ndarray, losses: np.ndarray, counts: np.ndarray,
        confirmed_ids: np.ndarray, path, distances: np.ndarray | None = None,
    ) -> np.ndarray:
        Z = self._embed(reps, distances)
        confirmed = np.zeros(len(reps), dtype=bool)
        confirmed[confirmed_ids.astype(int)] = True
        sizes = 90 + 25 * np.sqrt(np.maximum(counts, 1))

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            Z[confirmed, 0], Z[confirmed, 1], c=losses[confirmed],
            s=sizes[confirmed], marker="o", cmap="viridis", edgecolor="k",
            label="confirmed local minima", zorder=3,
        )
        if np.any(~confirmed):
            ax.scatter(
                Z[~confirmed, 0], Z[~confirmed, 1], c="lightgray",
                s=sizes[~confirmed], marker="x", linewidth=2.0,
                label="not confirmed", zorder=2,
            )
        for i, z in enumerate(Z):
            ax.text(z[0], z[1], f" c{i}", fontsize=9, weight="bold")
        fig.colorbar(sc, ax=ax, label="loss")
        ax.set_title("Saved local minima only, MDS")
        ax.set_xlabel("MDS 1")
        ax.set_ylabel("MDS 2")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return Z

    def _embed(self, reps: np.ndarray, distances: np.ndarray | None) -> np.ndarray:
        if len(reps) == 1:
            return np.zeros((1, 2))
        if distances is not None:
            return MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=11,
                normalized_stress="auto",
            ).fit_transform(distances)
        return MDS(
            n_components=2,
            dissimilarity="euclidean",
            random_state=11,
            normalized_stress="auto",
        ).fit_transform(reps)
