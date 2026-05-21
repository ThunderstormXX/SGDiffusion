from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS


class TransitionPlots:
    def graph(self, reps: np.ndarray, runs: list[dict], path) -> None:
        Z = MDS(n_components=2, random_state=7, normalized_stress="auto").fit_transform(reps)
        edges = self._edge_counts(runs)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(Z[:, 0], Z[:, 1], s=150, marker="o", facecolor="white", edgecolor="black")
        for idx, z in enumerate(Z):
            ax.text(z[0], z[1], str(idx), ha="center", va="center", weight="bold", fontsize=8)
        if not edges:
            ax.text(0.5, 0.03, "no cluster switches observed", transform=ax.transAxes, ha="center")
        for k, ((src, dst), count) in enumerate(edges.items()):
            self._arrow(ax, Z[src], Z[dst], count, k)
        ax.set_title("SGD transitions between saved local-minimum clusters")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def matrix(self, runs: list[dict], n_clusters: int, path) -> None:
        M = np.zeros((n_clusters, n_clusters), dtype=int)
        for (src, dst), count in self._edge_counts(runs).items():
            M[src, dst] = count
        active = np.where((M.sum(axis=0) + M.sum(axis=1)) > 0)[0]
        active = active if len(active) else np.arange(min(n_clusters, 1))
        M_small = M[np.ix_(active, active)]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M_small, cmap="magma")
        ax.set_xticks(np.arange(len(active)), labels=active)
        ax.set_yticks(np.arange(len(active)), labels=active)
        ax.set_xlabel("to cluster")
        ax.set_ylabel("from cluster")
        ax.set_title("cluster-switch counts")
        for i in range(M_small.shape[0]):
            for j in range(M_small.shape[1]):
                if M_small[i, j]:
                    ax.text(j, i, str(M_small[i, j]), ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax, label="switch count")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _edge_counts(self, runs: list[dict]) -> Counter:
        edges = Counter()
        for run in runs:
            seq = self._compress(run["clusters"])
            edges.update(zip(seq[:-1], seq[1:], strict=False))
        return edges

    def _compress(self, ids: np.ndarray) -> list[int]:
        out = []
        for cid in ids.astype(int):
            if not out or out[-1] != cid:
                out.append(int(cid))
        return out

    def _arrow(self, ax, start: np.ndarray, stop: np.ndarray, count: int, k: int) -> None:
        rad = 0.16 if k % 2 == 0 else -0.16
        ax.annotate(
            "",
            xy=stop,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "lw": 1.2 + 0.8 * count,
                "color": f"C{k % 10}",
                "shrinkA": 14,
                "shrinkB": 14,
                "connectionstyle": f"arc3,rad={rad}",
            },
        )
        mid = 0.5 * (start + stop)
        ax.text(mid[0], mid[1], str(count), color=f"C{k % 10}", fontsize=9, weight="bold")
