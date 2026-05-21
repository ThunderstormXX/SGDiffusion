from __future__ import annotations

from .config import ExperimentConfig
from .hessian import HessianAnalyzer
from .minima import MinimaClusterer, MinimaClusters
from .optim import LocalOptimizer
from .plots import ExperimentPlots
from .probes import AttractionProbe, SegmentProbe
from .sampler import ParameterSampler


class BasinAnalysis:
    def __init__(
        self, cfg: ExperimentConfig, opt: LocalOptimizer, sampler: ParameterSampler,
        clusterer: MinimaClusterer, hess: HessianAnalyzer, plots: ExperimentPlots, store,
    ):
        self.cfg = cfg
        self.opt = opt
        self.sampler = sampler
        self.clusterer = clusterer
        self.hess = hess
        self.plots = plots
        self.store = store

    def run(self, top: MinimaClusters, X, y) -> tuple[list[dict], list[dict]]:
        spectra = [self.hess.spectrum(t, X, y) for t in top.reps[: self.cfg.hessian_clusters]]
        if spectra:
            self.plots.spectra(spectra, self.store.path("hessian_spectra.png"))
        barriers = self._segments(top, X, y)
        attract = AttractionProbe(self.opt, self.sampler, self.clusterer)
        probe = attract.run(top, X, y, self.cfg.probe_radius, self.cfg.probe_per_cluster)
        self.store.npz("attraction_probe.npz", edges=probe["edges"], losses=probe["losses"])
        return spectra, barriers

    def _segments(self, top: MinimaClusters, X, y) -> list[dict]:
        seg = SegmentProbe(self.opt.loss, self.cfg)
        barriers = []
        for rank, (i, j, dist) in enumerate(seg.closest_pairs(top)):
            scan = seg.scan(top.reps[i], top.reps[j], X, y)
            self.store.npz(f"segment_pair_{rank}.npz", t=scan["t"], losses=scan["losses"])
            path = self.store.path(f"segment_pair_{rank}.png")
            self.plots.segment(scan["t"], scan["losses"], path, f"c{i}-c{j}")
            barriers.append({"rank": rank, "i": i, "j": j, "distance": dist, "barrier": scan["barrier"]})
        return barriers
