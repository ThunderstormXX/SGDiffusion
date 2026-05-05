# Paper-Ready Experimental Pack

This pack is designed for the NeurIPS paper "Why SGD is not Brownian Motion".
It reuses the existing `exp6` pipeline and keeps the artifact contract:

```text
run_experiment.py -> result_dir/
  config.yaml
  metrics.json
  runtime.json
  environment.json
  raw_outputs.npz
  figure_data.csv
  figure.png
  README_experiment.md
```

Additional aggregate files are written when needed, without replacing the
standard outputs.

## P0 Experiments

### EXP1B: SGD vs standard Langevin vs modified Langevin

Purpose: direct head-to-head falsification on MLP-386.

Smoke:

```bash
python -m src.scripts.exp6.src.run_experiment \
  src/scripts/exp6/configs/exp1b_mlp386/smoke.yaml \
  --make-figure
```

Full:

```bash
python -m src.scripts.exp6.src.run_experiment \
  src/scripts/exp6/configs/exp1b_mlp386/full.yaml \
  --make-figure
```

Main outputs:
- `figure_data.csv`: variance, mean displacement, mean path error to SGD for each method and learning rate.
- `metrics.json`: Wasserstein distance to SGD and modified/standard comparison per learning rate.
- `figure.png`: method comparison over the learning-rate grid.

Expected interpretation: small learning rates should make all methods closer;
finite/high learning rates should reveal systematic deviation of standard
Langevin from SGD, with modified Langevin closer in the finite-step regime.

### EXP2: quantitative Eq. 32 validation and Hessian-noise diagnostics

Purpose: predicted-vs-measured variance in the mean-Hessian eigenbasis, plus
assumption checks for Hessian noise structure.

Smoke:

```bash
python -m src.scripts.exp6.src.run_experiment \
  src/scripts/exp6/configs/exp2_eq32/smoke.yaml \
  --make-figure
```

Full:

```bash
python -m src.scripts.exp6.src.run_experiment \
  src/scripts/exp6/configs/exp2_eq32/full.yaml \
  --make-figure
```

Main outputs:
- `figure_data.csv`: predicted and measured variance by eigendirection.
- `sanity_checks.csv`: correlation, relative error, held-out scalar checks.
- `hessian_noise_diagnostics.csv`: rotated-Hessian off-diagonal mass and ratio statistics.
- `metrics.json`: gamma estimate, top/all/random direction scores, Hessian-noise summary.

Expected interpretation: top eigendirections should show meaningful agreement
between predicted and measured variance; diagnostics quantify how approximate
the diagonality/independence assumptions are.

### EXP10/EXP11: multi-seed alpha sweep

Purpose: test whether scaling the minibatch-noise component helps or hurts
optimization, with real error bars rather than a single trajectory.

EXP10 uses raw noise:

```bash
python -m src.scripts.exp6.src.run_experiment \
  src/scripts/exp6/configs/exp10_noise_alpha_sweep/full.yaml \
  --make-figure
```

EXP11 normalizes noise to the full-gradient norm:

```bash
python -m src.scripts.exp6.src.run_experiment \
  src/scripts/exp6/configs/exp11_normalized_noise_alpha_sweep/full.yaml \
  --make-figure
```

Smoke commands use the corresponding `smoke.yaml` files.

Main outputs:
- `loss_data.csv`: aggregated train/test curves with mean/std/95% CI.
- `loss_data_by_seed.csv`: per-seed train/test curves.
- `trajectory_data.csv`: aggregated displacement norm and ensemble parameter-variance trace over time.
- `trajectory_data_by_seed.csv`: per-seed displacement trajectories.
- `final_alpha_sweep.csv`: final metrics by alpha with mean/std/95% CI.
- `final_alpha_sweep_by_seed.csv`: per-seed final metrics.
- `figure.png`: loss curves, final alpha sweep, and ensemble displacement variance.

Expected interpretation: alpha controls the stochastic part only. `alpha=0`
is full-gradient descent from the same reference point and should have near-zero
ensemble variance. `alpha=1` recovers SGD in EXP10 and norm-matched
stochasticity in EXP11.

## P1 Experiments

### EXP5 learning-rate scaling

Purpose: show that flat-direction variance growth scales with learning rate,
while sharp-direction variance remains comparatively bounded.

Smoke:

```bash
python -m src.scripts.exp6.src.run_experiment \
  src/scripts/exp6/configs/exp5_lr_scaling/smoke.yaml \
  --make-figure
```

Full:

```bash
python -m src.scripts.exp6.src.run_experiment \
  src/scripts/exp6/configs/exp5_lr_scaling/full.yaml \
  --make-figure
```

Main outputs:
- `bucketed_statistics.csv`: flat/sharp bucket variance over time and learning rate.
- `metrics.json`: late-time slopes and plateau summaries per bucket/lr.
- `figure.png`: flat and sharp projected variance curves.

Expected interpretation: flat bucket late-time slope should increase with
learning rate; sharp bucket should remain bounded relative to flat directions.

## Recommended Full Run Order

1. EXP1B full: most important direct SGD/Langevin falsification.
2. EXP2 full: quantitative validation and assumption diagnostics.
3. EXP10 full: raw alpha sweep with multi-seed error bars.
4. EXP11 full: normalized-noise alpha sweep.
5. EXP5 lr-scaling full: learning-rate scaling of flat/sharp modes.
6. EXP3 full, optional: replacement vs no-replacement robustness.

## Main Paper/Rebuttal Figures

- `figures/exp1b_mlp386_full.png`: head-to-head SGD/Langevin comparison.
- `figures/exp2_eq32_full.png`: predicted vs measured variance.
- `figures/exp10_noise_alpha_sweep_full.png`: raw alpha sweep with error bars.
- `figures/exp11_normalized_noise_alpha_sweep_full.png`: normalized alpha sweep with error bars.
- `figures/exp5_lr_scaling_full.png`: learning-rate scaling in flat/sharp buckets.
