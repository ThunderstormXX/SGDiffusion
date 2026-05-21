from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS


class ExperimentPlots:
    def dataset(self, X: np.ndarray, y: np.ndarray, path) -> None:
        plt.figure(figsize=(5, 4))
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor="k", linewidth=0.3)
        plt.title("two moons dataset")
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()

    def loss_hist(self, losses: np.ndarray, path, title: str) -> None:
        plt.figure(figsize=(6, 4))
        plt.hist(losses, bins=40)
        plt.xlabel("loss")
        plt.ylabel("count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()

    def clusters_mds(self, Theta: np.ndarray, losses: np.ndarray, reps: np.ndarray, path) -> None:
        Z = MDS(n_components=2, random_state=0, normalized_stress="auto").fit_transform(
            np.vstack([Theta, reps])
        )
        pts, rpts = Z[: len(Theta)], Z[len(Theta) :]
        plt.figure(figsize=(7, 5))
        sc = plt.scatter(pts[:, 0], pts[:, 1], c=losses, s=10, alpha=0.55)
        plt.colorbar(sc, label="loss")
        plt.scatter(rpts[:, 0], rpts[:, 1], s=180, marker="*", edgecolor="k")
        for i, z in enumerate(rpts):
            plt.text(z[0], z[1], f" {i}", weight="bold")
        plt.title("local minima clusters, MDS")
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()

    def segment(self, t: np.ndarray, losses: np.ndarray, path, title: str) -> None:
        plt.figure(figsize=(6, 4))
        plt.plot(t, losses, lw=2)
        plt.axvline(0.0, ls="--", color="k", alpha=0.5)
        plt.axvline(1.0, ls="--", color="k", alpha=0.5)
        plt.xlabel("segment coordinate t")
        plt.ylabel("loss")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()

    def spectra(self, spectra: list[dict], path) -> None:
        plt.figure(figsize=(7, 4))
        for i, spec in enumerate(spectra):
            plt.plot(np.sort(spec["eigvals"]), marker="o", ms=3, label=f"c{i}")
        plt.axhline(0.0, color="k", lw=1)
        plt.xlabel("eigenvalue index")
        plt.ylabel("Hessian eigenvalue")
        plt.title("Hessian spectra at cluster representatives")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()

