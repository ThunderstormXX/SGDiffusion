from __future__ import annotations

from collections import Counter

import numpy as np


class TransitionPairSelector:
    def pick(
        self, reps: np.ndarray, runs: list[dict], valid_ids: np.ndarray | None = None
    ) -> tuple[int, int, str]:
        valid = self._valid_set(reps, valid_ids)
        edges = Counter()
        for run in runs:
            seq = self._compress(run["clusters"])
            for a, b in zip(seq[:-1], seq[1:], strict=False):
                if int(a) in valid and int(b) in valid:
                    edges[tuple(sorted((int(a), int(b))))] += 1
        if edges:
            pair, count = edges.most_common(1)[0]
            return int(pair[0]), int(pair[1]), f"observed_switches_{count}"
        i, j = self._closest_pair(reps, sorted(valid))
        return i, j, "closest_confirmed_pair_fallback"

    def _valid_set(self, reps: np.ndarray, valid_ids: np.ndarray | None) -> set[int]:
        if valid_ids is None or len(valid_ids) < 2:
            return set(range(len(reps)))
        return {int(x) for x in valid_ids}

    def _compress(self, ids: np.ndarray) -> list[int]:
        out = []
        for cid in ids.astype(int):
            if not out or out[-1] != cid:
                out.append(int(cid))
        return out

    def _closest_pair(self, reps: np.ndarray, ids: list[int]) -> tuple[int, int]:
        best = (ids[0], ids[1], float("inf"))
        for pos, i in enumerate(ids):
            for j in ids[pos + 1 :]:
                dist = float(np.linalg.norm(reps[i] - reps[j]))
                best = (i, j, dist) if dist < best[2] else best
        return best[0], best[1]
