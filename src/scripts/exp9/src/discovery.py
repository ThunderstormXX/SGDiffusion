from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .loss import BCELoss
from .minima import MinimaClusterer, MinimaClusters
from .optim import LocalOptimizer
from .plots import ExperimentPlots
from .sampler import ParameterSampler


@dataclass
class DiscoveryResult:
    Theta: np.ndarray
    losses: np.ndarray
    grad_norms: np.ndarray
    clusters: MinimaClusters
    top: MinimaClusters


class MinimaDiscovery:
    def __init__(
        self, cfg: ExperimentConfig, loss: BCELoss, opt: LocalOptimizer,
        sampler: ParameterSampler, clusterer: MinimaClusterer, plots: ExperimentPlots, store,
    ):
        self.cfg = cfg
        self.loss = loss
        self.opt = opt
        self.sampler = sampler
        self.clusterer = clusterer
        self.plots = plots
        self.store = store

    def run(self, X: np.ndarray, y: np.ndarray) -> DiscoveryResult:
        rand_Theta, rand_losses, rand_grad_norms = self.opt.run_many(X, y)
        prior = self.sampler.sobol_box(self.cfg.prior_samples, self.cfg.prior_scale)
        prior_losses = self.loss.batch_values(prior, X, y, "prior probe")
        elite = prior[np.argsort(prior_losses)[: self.cfg.prior_refine]]
        prior_Theta, prior_refined_losses, prior_grad_norms = self.opt.run_starts(elite, X, y, "prior refine")
        Theta = np.vstack([rand_Theta, prior_Theta])
        losses = np.r_[rand_losses, prior_refined_losses]
        grad_norms = np.r_[rand_grad_norms, prior_grad_norms]
        source = np.r_[np.zeros(len(rand_Theta), dtype=int), np.ones(len(prior_Theta), dtype=int)]
        self.store.npz("raw_minima.npz", Theta=Theta, losses=losses, grad_norms=grad_norms, source=source)
        self.store.npz("prior_probe.npz", Theta=prior, losses=prior_losses)
        self.plots.loss_hist(losses, self.store.path("minima_loss_hist.png"), "optimized local losses")
        clusters = self.clusterer.fit(Theta, losses)
        top = clusters.top(min(self.cfg.top_clusters, len(clusters.reps)))
        self.store.npz("clusters.npz", reps=clusters.reps, losses=clusters.rep_losses,
                       assignments=clusters.assignments, counts=clusters.counts)
        return DiscoveryResult(Theta, losses, grad_norms, clusters, top)

