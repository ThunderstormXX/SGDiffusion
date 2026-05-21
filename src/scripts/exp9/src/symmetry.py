from __future__ import annotations

from itertools import permutations

import numpy as np

from .model import SmoothTinyMLP


class HiddenPermutationSymmetry:
    def __init__(self, model: SmoothTinyMLP):
        self.model = model
        self.perms = list(permutations(range(model.hidden_dim)))

    def permute(self, theta: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
        W1, b1, W2 = self.model.unpack(theta)
        return np.concatenate([W1[list(perm)].ravel(), b1[list(perm)], W2[list(perm)]])

    def canonical(self, theta: np.ndarray) -> np.ndarray:
        W1, b1, W2 = self.model.unpack(theta)
        keys = np.column_stack([W2, b1, W1])
        order = np.lexsort(keys.T[::-1])
        return self.permute(theta, tuple(int(i) for i in order))

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(min(np.linalg.norm(a - self.permute(b, p)) for p in self.perms))

    def distance_matrix(self, Theta: np.ndarray) -> np.ndarray:
        n = len(Theta)
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = self.distance(Theta[i], Theta[j])
        return D

