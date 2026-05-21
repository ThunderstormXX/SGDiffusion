from __future__ import annotations

from .artifacts import ArtifactStore
from .basin_analysis import BasinAnalysis
from .config import parse_config
from .dataset import BinaryPointDataset
from .decision_plots import DecisionBoundaryPlots
from .discovery import MinimaDiscovery
from .gd_validation_stage import GDValidationStage
from .gd_validator import GDValidator
from .hessian import HessianAnalyzer
from .local_minima_plots import LocalMinimaPlots
from .loss import BCELoss
from .lr_sweep_stage import LRSweepStage
from .memory import MinimaMemory
from .minima import MinimaClusterer
from .model import SmoothTinyMLP
from .optim import LocalOptimizer
from .pair_landscape import PairLandscapeStage
from .plots import ExperimentPlots
from .report import TextReport
from .sampler import ParameterSampler
from .symmetry import HiddenPermutationSymmetry
from .walk_plots import TrajectoryPlots
from .walk_stage import SGDWalkStage
from .walker import SGDWalker


def confirmed_cluster_ids(grad_norms, best_indices):
    return (grad_norms[best_indices] < 1e-5).nonzero()[0]


def main() -> None:
    cfg = parse_config()
    store = ArtifactStore(cfg)
    store.json("config.json", cfg.to_dict())

    data = BinaryPointDataset(cfg).build()
    X, y = data.arrays()
    model = SmoothTinyMLP(cfg)
    loss = BCELoss(model, cfg)
    opt = LocalOptimizer(model, loss, cfg)
    sampler = ParameterSampler(cfg, model.dim)
    sym = HiddenPermutationSymmetry(model)
    clusterer = MinimaClusterer(sym, cfg.cluster_eps)
    plots = ExperimentPlots()
    plots.dataset(X, y, store.path("dataset.png"))

    discovery = MinimaDiscovery(cfg, loss, opt, sampler, clusterer, plots, store)
    found = discovery.run(X, y)
    confirmed_ids = confirmed_cluster_ids(found.grad_norms, found.clusters.best_indices)
    MinimaMemory(store).save(found.clusters, confirmed_ids)
    if cfg.make_mds:
        plots.clusters_mds(found.Theta, found.losses, found.clusters.reps, store.path("clusters_mds.png"))
        Z = LocalMinimaPlots().mds(
            found.clusters.reps, found.clusters.rep_losses, found.clusters.counts,
            confirmed_ids, store.path("local_minima_mds.png"),
            distances=sym.distance_matrix(found.clusters.reps),
        )
        store.npz("local_minima_mds.npz", Z=Z, confirmed_ids=confirmed_ids)
    DecisionBoundaryPlots(model).top_clusters(found.top, X, y, store.path("minima"), len(found.top.reps))

    analysis = BasinAnalysis(cfg, opt, sampler, clusterer, HessianAnalyzer(loss), plots, store)
    spectra, barriers = analysis.run(found.top, X, y)

    walker = SGDWalker(loss, sym, cfg)
    walk_stage = SGDWalkStage(cfg, walker, TrajectoryPlots(), store)
    runs, walk_summary = walk_stage.run(found.clusters, X, y)
    gd_check = GDValidationStage(GDValidator(cfg, loss, sym), store).run(found.clusters, runs, X, y, confirmed_ids)
    lr_sweep = LRSweepStage(cfg, walker, opt, store).run(found.clusters.reps, confirmed_ids, X, y)
    pair_landscape = PairLandscapeStage(cfg, loss, sampler, store).run(
        found.clusters.reps, runs, X, y, confirmed_ids
    )

    report = TextReport()
    store.text("REPORT.md", report.clusters(found.clusters, found.grad_norms))
    summary = report.summary(found.clusters, spectra, barriers)
    summary["sgd_walks"] = walk_summary
    summary["confirmed_cluster_ids"] = confirmed_ids.astype(int).tolist()
    summary["gd_validation"] = gd_check
    summary["sgd_lr_sweep"] = lr_sweep
    summary["pair_landscape"] = pair_landscape
    store.json("summary.json", summary)
    print(f"saved exp9 artifacts to {store.root}")


if __name__ == "__main__":
    main()
