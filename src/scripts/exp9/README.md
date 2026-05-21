# exp9: local minima in a 16D two-moons MLP

This experiment searches for local attractor points in a small nonconvex
network:

```text
p(y=1|x, theta) = sigmoid(W2 sigmoid(W1 x + b1))
```

With four hidden units and no output bias the model has 16 parameters.

The pipeline:

1. runs many random-start SGD trajectories;
2. polishes each endpoint with L-BFGS;
3. clusters local minima with hidden-unit permutation symmetry;
4. saves every cluster representative as a separate `minima/cluster_*.npz`;
5. visualizes the discovered minima in MDS space;
6. computes finite-difference Hessian spectra at top clusters;
7. scans loss barriers along closest pairs of cluster representatives;
8. probes local attraction by perturbing representatives and re-optimizing;
9. runs mini-batch SGD walks and assigns every checkpoint to the nearest found minimum.

Smoke run:

```bash
src/scripts/exp9/sh_scripts/run_smoke.sh
```

Full run:

```bash
python -m src.scripts.exp9.src.run --preset full
```

Results are written to `src/scripts/exp9/results/<result_name>/`.

Important outputs:

- `minima_manifest.json`: all found minima with 16D coordinates.
- `minima/cluster_*.npz`: one saved representative per local-minimum cluster.
- `clusters_mds.png`: visual map of found minima.
- `sgd_walk_*.npz`: SGD trajectory checkpoints and nearest-minimum ids.
- `sgd_walk_timeline.png`: nearest-minimum id, loss, and distance over SGD time.
- `sgd_walk_mds.png`: SGD paths projected together with saved minima.
- `cluster_transition_paths.png`: directed cluster-to-cluster switches along SGD walks.
- `cluster_transition_matrix.png`: transition-count matrix for observed switches.
- `pair_geodesic_motion.png`: point motion inside the selected two-cluster neighborhood.
- `pair_geodesic_motion_zoom.png`: full context plus a zoomed trajectory panel.
- `pair_landscape_3d.png`: 3D geodesic-MDS loss canvas for that neighborhood.
- `gd_validation_summary.json`: full-batch GD checks from midpoint/trajectory starts.
- `sgd_lr_sweep_mds.png`: geodesic-MDS LR paths with only visited minima shown.
- `sgd_lr_sweep_local_mds.png`: geodesic-MDS over actual SGD checkpoints only.
- `sgd_lr_sweep_summary.json`: nearest and polished destination for each LR.
