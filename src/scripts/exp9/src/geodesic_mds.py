from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors


class GeodesicMDS:
    def __init__(self, n_neighbors: int):
        self.n_neighbors = n_neighbors

    def fit_transform(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        D = self._geodesic_distances(X)
        return MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=9,
            normalized_stress="auto",
        ).fit_transform(D)

    def _geodesic_distances(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        k = min(self.n_neighbors + 1, n)
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        dist, ind = nbrs.kneighbors(X)
        rows, cols, data = [], [], []
        for i in range(n):
            for d, j in zip(dist[i, 1:], ind[i, 1:], strict=False):
                rows.extend([i, int(j)])
                cols.extend([int(j), i])
                data.extend([float(d), float(d)])
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))
        D = shortest_path(graph, directed=False, unweighted=False)
        return self._repair_disconnected(D)

    def _repair_disconnected(self, D: np.ndarray) -> np.ndarray:
        finite = D[np.isfinite(D)]
        fill = 10.0 * finite.max() if len(finite) else 1.0
        D = np.where(np.isfinite(D), D, fill)
        return 0.5 * (D + D.T)

