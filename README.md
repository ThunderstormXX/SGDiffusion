# Why SGD is not Brownian Motion: Reproducible Experiments

This repository contains the experimental code for the paper **"Why SGD is not
Brownian Motion"**.  The current reproducibility package is centered around the
`exp6` pipeline, which was designed to produce machine-readable artifacts,
figures, metrics, and environment metadata from a single command per experiment.

The experiments test finite-step SGD dynamics against Langevin-type
approximations, discrete Fokker-Planck predictions, corrected-generator
approximations, Hessian-basis diagnostics, and small real-network validations.

## Repository Layout

```text
src/scripts/exp6/
  configs/        # smoke/full YAML configs for each experiment
  src/            # experiment runners and shared code
  scripts/        # figure generation and helper launchers
  results/        # generated result directories
  figures/        # top-level generated figures
```

Each experiment run writes:

```text
config.yaml
metrics.json
raw_outputs.npz
figure_data.csv
figure.png
figure_clean.png        # for newer figures, without the setup box
README_experiment.md
runtime.json
environment.json
make_figure.py
```

Figures are generated only from saved intermediate artifacts.

## Installation

Python 3.10+ is recommended.

```bash
pip install uv
uv sync
source .venv/bin/activate
```

If using a plain environment instead of `uv`, install the project dependencies
from `pyproject.toml`.  The exp6 pipeline uses PyTorch, NumPy, SciPy,
Matplotlib, pandas, PyYAML, psutil, and torchvision for the MNIST experiments.

## Running One Experiment

Use the shared runner:

```bash
python -m src.scripts.exp6.src.run_experiment \
  src/scripts/exp6/configs/<experiment_name>/smoke.yaml \
  --make-figure
```

For a full run, replace `smoke.yaml` with `full.yaml`.

Example:

```bash
python -m src.scripts.exp6.src.run_experiment \
  src/scripts/exp6/configs/exp43_mlp_langevin_mean_covariance/full.yaml \
  --make-figure
```

## Complete Experiment Catalog

The `exp6` directory contains both the historical development experiments and
the cleaned paper-ready subset.  The table below lists every experiment that has
a reproducible config under `src/scripts/exp6/configs/`.

| ID | Config directory | Purpose |
|---|---|---|
| EXP1A | `exp1a_toy` | Toy falsification test comparing discrete multiplicative-noise SGD with standard and modified Langevin-style baselines. |
| EXP1B | `exp1b_mlp386` | MLP-386 finite-step comparison of SGD, without-replacement SGD, standard Langevin, and modified Langevin-style dynamics from a common reference point. |
| EXP2 | `exp2_eq32` | MLP-386 quantitative validation of the covariance formula using dense Hessian statistics and ensemble SGD variance. |
| EXP2-NanoGPT | `exp2_eq32_nanogpt` | NanoGPT/Shakespeare variant of EXP2, including a setup matched to the successful MLP protocol. |
| EXP3 | `exp3_sampling` | Sampling robustness: SGD with replacement vs epoch reshuffling without replacement. |
| EXP4 | `exp4_scaling` | Placeholder/lite config for larger-model scaling diagnostics; disabled by default unless compute budget is selected. |
| EXP5 | `exp5_nonstationary` | Non-stationary regime test from an early MLP-386 point with non-zero mean gradient. |
| EXP5-LR | `exp5_lr_scaling` | Learning-rate scaling of flat/sharp Hessian-bucket variance on MLP-386. |
| EXP6 | `exp6_generalization` | Diffusion-vs-generalization diagnostic correlating displacement proxies with validation loss. |
| EXP7 | `exp7_diffusion_control` | Hessian-guided suppression of stochastic update components in selected eigenspaces. |
| EXP8 | `exp8_diffusion_amplification` | Theory-guided amplification of minibatch noise in selected eigendirections. |
| EXP9 | `exp9_theory_denoised_sgd` | Theory-denoised SGD proof of concept: keep full-gradient drift and remove selected noise components. |
| EXP10 | `exp10_noise_alpha_sweep` | Alpha sweep interpolating between full-gradient descent and ordinary SGD by scaling minibatch noise. |
| EXP11 | `exp11_normalized_noise_alpha_sweep` | Alpha sweep with noise normalized to the full-gradient norm before scaling. |
| EXP12 | `exp12_langevin_verification` | Audit of EXP1B Langevin implementations: scaling, one-step moments, horizons, and calibration. |
| EXP13 | `exp13_langevin_falsification_modes` | Development falsification by flat/sharp Hessian mode variance for SGD vs Gaussian surrogates. |
| EXP14 | `exp14_eta_batch_scaling` | Joint learning-rate and batch-size scaling diagnostic for projected variance. |
| EXP15 | `exp15_gradient_noise_hessian_alignment` | Diagnostic for minibatch gradient-noise variance alignment with Hessian eigenvalues. |
| EXP16 | `exp16_langevin_evidence_toy` | Conditional likelihood sanity check: `gg^T` is not conditional covariance. |
| EXP17 | `exp17_covariance_fp_toy` | Ensemble covariance evolution: discrete FP vs standard FP. |
| EXP18 | `exp18_learning_rate_scaling` | Learning-rate scaling of centered covariance and raw second moment. |
| EXP20 | `exp20_noise_anisotropy` | Directional raw-moment anisotropy from `gg^T`. |
| EXP21 | `exp21_long_time_distribution_mismatch` | Long-time covariance mismatch between standard FP and discrete FP. |
| EXP23 | `exp23_batch_size_scaling` | Batch-size scaling: centered noise vanishes, raw drift floor remains. |
| EXP27 | `exp27_stability_boundary` | Finite-step stability boundary: discrete SGD vs standard Langevin. |
| EXP29 | `exp29_nonstationary_drift_spreading` | Non-stationary drift-induced spreading. |
| EXP33 | `exp33_real_hessian_eigenbasis` | Hessian eigenbasis decoupling on MLP/MNIST. |
| EXP35 | `exp35_real_sampling_replacement` | With-replacement vs without-replacement sampling on MLP/MNIST. |
| EXP36 | `exp36_real_learning_rate_scaling` | Learning-rate scaling on MLP/MNIST. |
| EXP37 | `exp37_corrected_langevin_mean_matching` | Corrected-generator mean-map matching. |
| EXP38 | `exp38_corrected_generator_distribution` | Corrected-generator distribution/moment matching. |
| EXP39 | `exp39_drift_correction_ablation` | Ablation: corrected drift vs wrong raw covariance. |
| EXP40 | `exp40_semigroup_log_matching` | Finite-state semigroup logarithm matching diagnostic. |
| EXP41 | `exp41_higher_order_cumulants` | Higher-order cumulant / Pawula limitation diagnostic. |
| EXP42 | `exp42_poisson_jump_surrogate` | Exploratory Poisson/jump-process surrogate. |
| EXP43 | `exp43_mlp_langevin_mean_covariance` | MLP/MNIST mean trajectory and covariance: SGD vs standard vs drift-corrected Langevin. |
| EXP44 | `exp44_rough_landscape_langevin` | Rough fluctuating 1D landscape: SGD vs standard vs drift-corrected Langevin. |

## Paper-Ready Subset

The current strongest subset for the NeurIPS paper/rebuttal is:

- **Local dynamics:** EXP16, EXP18, EXP20, EXP37, EXP39.
- **Distribution evolution:** EXP17, EXP21, EXP27, EXP38, EXP40.
- **Non-stationarity and scaling:** EXP23, EXP29, EXP36.
- **Real-network validation:** EXP2, EXP2-NanoGPT, EXP33, EXP35, EXP43.
- **Limitations/exploratory:** EXP41 and EXP42.

Earlier experiments EXP1A-EXP15 are retained because they document the
development path and provide useful checks, but they should not all be treated
as equally central paper figures. EXP7-EXP11 are intervention/alpha-sweep
diagnostics rather than core theory-validation results.

## Smoke Runs

Smoke runs are intended to verify that the pipeline, artifacts, and plotting
work end-to-end.  Most configs provide `smoke.yaml`; if a config only has
`full.yaml`, skip it in the smoke loop or run the full config directly.

To run every available smoke config:

```bash
for cfg in src/scripts/exp6/configs/*/smoke.yaml; do
  python -m src.scripts.exp6.src.run_experiment "$cfg" --make-figure
done
```

To run only the current paper-ready smoke subset:

```bash
for cfg in \
  src/scripts/exp6/configs/exp2_eq32/smoke.yaml \
  src/scripts/exp6/configs/exp2_eq32_nanogpt/smoke.yaml \
  src/scripts/exp6/configs/exp16_langevin_evidence_toy/smoke.yaml \
  src/scripts/exp6/configs/exp17_covariance_fp_toy/smoke.yaml \
  src/scripts/exp6/configs/exp18_learning_rate_scaling/smoke.yaml \
  src/scripts/exp6/configs/exp20_noise_anisotropy/smoke.yaml \
  src/scripts/exp6/configs/exp21_long_time_distribution_mismatch/smoke.yaml \
  src/scripts/exp6/configs/exp23_batch_size_scaling/smoke.yaml \
  src/scripts/exp6/configs/exp27_stability_boundary/smoke.yaml \
  src/scripts/exp6/configs/exp29_nonstationary_drift_spreading/smoke.yaml \
  src/scripts/exp6/configs/exp33_real_hessian_eigenbasis/smoke.yaml \
  src/scripts/exp6/configs/exp35_real_sampling_replacement/smoke.yaml \
  src/scripts/exp6/configs/exp36_real_learning_rate_scaling/smoke.yaml \
  src/scripts/exp6/configs/exp37_corrected_langevin_mean_matching/smoke.yaml \
  src/scripts/exp6/configs/exp38_corrected_generator_distribution/smoke.yaml \
  src/scripts/exp6/configs/exp39_drift_correction_ablation/smoke.yaml \
  src/scripts/exp6/configs/exp40_semigroup_log_matching/smoke.yaml \
  src/scripts/exp6/configs/exp41_higher_order_cumulants/smoke.yaml \
  src/scripts/exp6/configs/exp42_poisson_jump_surrogate/smoke.yaml \
  src/scripts/exp6/configs/exp43_mlp_langevin_mean_covariance/smoke.yaml \
  src/scripts/exp6/configs/exp44_rough_landscape_langevin/smoke.yaml
do
  python -m src.scripts.exp6.src.run_experiment "$cfg" --make-figure
done
```

## Full Runs

To run every available full config:

```bash
for cfg in src/scripts/exp6/configs/*/full.yaml; do
  python -m src.scripts.exp6.src.run_experiment "$cfg" --make-figure
done
```

Some experiments also contain specialized configs, for example
`exp2_eq32_nanogpt/mlpstyle_full.yaml`. Run those explicitly when needed:

```bash
python -m src.scripts.exp6.src.run_experiment \
  src/scripts/exp6/configs/exp2_eq32_nanogpt/mlpstyle_full.yaml \
  --make-figure
```

The following commands reproduce the full artifacts for the current
paper-ready subset.

```bash
for cfg in \
  src/scripts/exp6/configs/exp2_eq32/full.yaml \
  src/scripts/exp6/configs/exp2_eq32_nanogpt/full.yaml \
  src/scripts/exp6/configs/exp16_langevin_evidence_toy/full.yaml \
  src/scripts/exp6/configs/exp17_covariance_fp_toy/full.yaml \
  src/scripts/exp6/configs/exp18_learning_rate_scaling/full.yaml \
  src/scripts/exp6/configs/exp20_noise_anisotropy/full.yaml \
  src/scripts/exp6/configs/exp21_long_time_distribution_mismatch/full.yaml \
  src/scripts/exp6/configs/exp23_batch_size_scaling/full.yaml \
  src/scripts/exp6/configs/exp27_stability_boundary/full.yaml \
  src/scripts/exp6/configs/exp29_nonstationary_drift_spreading/full.yaml \
  src/scripts/exp6/configs/exp33_real_hessian_eigenbasis/full.yaml \
  src/scripts/exp6/configs/exp35_real_sampling_replacement/full.yaml \
  src/scripts/exp6/configs/exp36_real_learning_rate_scaling/full.yaml \
  src/scripts/exp6/configs/exp37_corrected_langevin_mean_matching/full.yaml \
  src/scripts/exp6/configs/exp38_corrected_generator_distribution/full.yaml \
  src/scripts/exp6/configs/exp39_drift_correction_ablation/full.yaml \
  src/scripts/exp6/configs/exp40_semigroup_log_matching/full.yaml \
  src/scripts/exp6/configs/exp41_higher_order_cumulants/full.yaml \
  src/scripts/exp6/configs/exp42_poisson_jump_surrogate/full.yaml \
  src/scripts/exp6/configs/exp43_mlp_langevin_mean_covariance/full.yaml \
  src/scripts/exp6/configs/exp44_rough_landscape_langevin/full.yaml
do
  python -m src.scripts.exp6.src.run_experiment "$cfg" --make-figure
done
```

The real-network MLP/MNIST experiments are EXP1B, EXP2, EXP3, EXP5, EXP33,
EXP35, EXP36, and EXP43. The NanoGPT/Shakespeare validation is
`exp2_eq32_nanogpt`. These are still small enough for CPU execution, but exact
Hessian computation can be slower than the synthetic experiments.

## Regenerating Figures from Existing Results

Each result directory contains the exact plotting script used for that run:

```bash
python src/scripts/exp6/results/<result_name>/make_figure.py \
  src/scripts/exp6/results/<result_name>
```

Example:

```bash
python src/scripts/exp6/results/exp44_rough_landscape_langevin_full/make_figure.py \
  src/scripts/exp6/results/exp44_rough_landscape_langevin_full
```

The top-level copies are written to:

```text
src/scripts/exp6/figures/
```

## Interpreting Key Experiments

- **EXP1A/EXP1B** are the original falsification checks. EXP1A is a controlled
  toy example; EXP1B moves the same question to MLP-386 and audits SGD vs
  Langevin-like surrogates at finite step size.
- **EXP2 and EXP2-NanoGPT** are the direct quantitative checks of the covariance
  prediction in the Hessian eigenbasis. EXP2 is the MLP-386 reference result;
  EXP2-NanoGPT repeats the same validation on the small Shakespeare/NanoGPT
  setup.
- **EXP3/EXP35** isolate the effect of the sampling assumption. The theory is
  derived for independent minibatches, so with-replacement sampling is the
  clean reference case; reshuffling without replacement is reported as a
  robustness/limitation check.
- **EXP4** is a disabled scaling placeholder. It documents the intended
  large-model extension but is not a central reproduced result.
- **EXP5/EXP5-LR** test non-stationary dynamics and learning-rate scaling on
  MLP-386. The important quantity is the contrast between flat-direction growth
  and sharp-direction confinement, not a perfect plateau.
- **EXP6** is a generalization diagnostic. It is useful for motivation, but the
  core paper claims do not rely on it.
- **EXP7/EXP8/EXP9** are theory-guided intervention prototypes. They explore
  suppressing, amplifying, or denoising selected stochastic update components;
  these are not claimed as production optimizers.
- **EXP10/EXP11** are causal alpha sweeps over the minibatch noise component.
  They test whether explicitly scaling the stochastic part changes loss,
  displacement, and variance in the expected direction.
- **EXP12** is an implementation audit for the Langevin baselines. It clarified
  that the older `D + gg^T` modification should not be interpreted as a
  conditional covariance correction.
- **EXP13/EXP14/EXP15** are development diagnostics for mode-wise falsification,
  eta/batch scaling, and gradient-noise/Hessian alignment. They are useful
  reviewer-facing checks but weaker than the cleaned EXP16+ suite.
- **EXP16** shows that `gg^T` does not belong to the conditional transition
  covariance. Standard conditional Gaussian likelihood is better there.
- **EXP17/EXP21** show that the discrete Fokker-Planck/raw-moment evolution
  matches ensemble covariance better than the standard FP approximation.
- **EXP37/EXP38/EXP39** test the corrected generator from the logarithmic
  matching note: the correction enters as a drift correction involving `H g`,
  not as `D + gg^T` conditional covariance.
- **EXP43** checks the same corrected-drift idea on the MLP/MNIST setup.
- **EXP44** gives a one-dimensional fluctuating-landscape visualization with
  exact SGD, standard Langevin, and drift-corrected Langevin.

## Notes for Anonymous Release

- The repository should be released without author-identifying paths or
  notebooks that are not needed for reproduction.
- `environment.json` records the local command, package versions, CPU/GPU
  metadata, and git commit/status for each run.
- The result directories are reproducible from their saved `config.yaml` files.
