from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .symmetry import HiddenPermutationSymmetry


@dataclass
class MinimaClusters:
    reps: np.ndarray
    rep_losses: np.ndarray
    assignments: np.ndarray
    counts: np.ndarray
    best_indices: np.ndarray

    def top(self, n: int) -> MinimaClusters:
        order = np.argsort(self.rep_losses)[:n]
        remap = {old: new for new, old in enumerate(order)}
        keep = np.array([a in remap for a in self.assignments])
        new_assign = np.array([remap[a] for a in self.assignments[keep]], dtype=int)
        return MinimaClusters(
            self.reps[order],
            self.rep_losses[order],
            new_assign,
            self.counts[order],
            self.best_indices[order],
        )


class MinimaClusterer:
    def __init__(self, sym: HiddenPermutationSymmetry, eps: float):
        self.sym = sym
        self.eps = eps

    def fit(self, Theta: np.ndarray, losses: np.ndarray) -> MinimaClusters:
        order = np.argsort(losses)
        reps, rep_losses, counts, best_indices = [], [], [], []
        assignments = np.full(len(Theta), -1, dtype=int)
        for idx in order:
            match = self._match(Theta[idx], reps)
            if match is None:
                match = len(reps)
                reps.append(self.sym.canonical(Theta[idx]))
                rep_losses.append(float(losses[idx]))
                best_indices.append(int(idx))
                counts.append(0)
            assignments[idx] = match
            counts[match] += 1
        return MinimaClusters(
            np.array(reps),
            np.array(rep_losses),
            assignments,
            np.array(counts, dtype=int),
            np.array(best_indices, dtype=int),
        )

    def assign_to_reps(self, theta: np.ndarray, reps: np.ndarray) -> int:
        match = self._match(theta, list(reps))
        return -1 if match is None else match

    def _match(self, theta: np.ndarray, reps: list[np.ndarray]) -> int | None:
        if not reps:
            return None
        distances = [self.sym.distance(theta, rep) for rep in reps]
        best = int(np.argmin(distances))
        return best if distances[best] <= self.eps else None
