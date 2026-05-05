#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CONFIGS=(
  "src/scripts/exp6/configs/exp1a_toy/smoke.yaml"
  "src/scripts/exp6/configs/exp1b_mlp386/smoke.yaml"
  "src/scripts/exp6/configs/exp2_eq32/smoke.yaml"
  "src/scripts/exp6/configs/exp3_sampling/smoke.yaml"
  "src/scripts/exp6/configs/exp5_nonstationary/smoke.yaml"
  "src/scripts/exp6/configs/exp5_lr_scaling/smoke.yaml"
  "src/scripts/exp6/configs/exp6_generalization/smoke.yaml"
  "src/scripts/exp6/configs/exp7_diffusion_control/smoke.yaml"
  "src/scripts/exp6/configs/exp8_diffusion_amplification/smoke.yaml"
  "src/scripts/exp6/configs/exp9_theory_denoised_sgd/smoke.yaml"
  "src/scripts/exp6/configs/exp10_noise_alpha_sweep/smoke.yaml"
  "src/scripts/exp6/configs/exp11_normalized_noise_alpha_sweep/smoke.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "=== smoke: ${cfg} ==="
  python -m src.scripts.exp6.src.run_experiment "${cfg}" --make-figure
done

python -m src.scripts.exp6.src.summarize --results-root src/scripts/exp6/results --output src/scripts/exp6/RESULTS_SUMMARY.md
