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

## Current Experiment Pack

The following experiments form the current paper/rebuttal-ready package.

| ID | Config directory | Purpose |
|---|---|---|
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

## Smoke Runs

Smoke runs are intended to verify that the pipeline, artifacts, and plotting
work end-to-end.

```bash
for cfg in \
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

The following commands reproduce the full artifacts for the current
experimental package.

```bash
for cfg in \
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

The real-network MLP/MNIST experiments are EXP33, EXP35, EXP36, and EXP43.
They are still small enough for CPU execution, but exact Hessian computation can
be slower than the synthetic experiments.

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
