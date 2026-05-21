from __future__ import annotations

import numpy as np

from .artifacts import ArtifactStore
from .minima import MinimaClusters


class MinimaMemory:
    def __init__(self, store: ArtifactStore):
        self.store = store
        self.root = store.path("minima")
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, clusters: MinimaClusters, confirmed_ids=None) -> list[dict]:
        confirmed = {int(x) for x in (confirmed_ids if confirmed_ids is not None else [])}
        manifest = []
        for idx, theta in enumerate(clusters.reps):
            name = f"cluster_{idx:03d}.npz"
            path = self.root / name
            np.savez_compressed(
                path,
                theta=theta,
                loss=clusters.rep_losses[idx],
                count=clusters.counts[idx],
                best_index=clusters.best_indices[idx],
            )
            manifest.append(
                {
                    "id": idx,
                    "file": f"minima/{name}",
                    "loss": float(clusters.rep_losses[idx]),
                    "count": int(clusters.counts[idx]),
                    "norm": float(np.linalg.norm(theta)),
                    "confirmed_local_minimum": int(idx) in confirmed,
                    "theta": theta.tolist(),
                }
            )
        self.store.json("minima_manifest.json", {"minima": manifest})
        return manifest
