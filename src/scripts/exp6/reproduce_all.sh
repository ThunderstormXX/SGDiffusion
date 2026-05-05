#!/usr/bin/env bash
set -euo pipefail

# By default this runs smoke-scale reproducible experiments.
# Use MODE=full for full configs. EXP4 is skipped unless RUN_LONG=1.
MODE="${MODE:-smoke}"
RUN_LONG="${RUN_LONG:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ "${MODE}" == "smoke" ]]; then
  bash src/scripts/exp6/scripts/run_smoke.sh
  exit 0
fi

CONFIGS=(
  "src/scripts/exp6/configs/exp1a_toy/full.yaml"
  "src/scripts/exp6/configs/exp1b_mlp386/full.yaml"
  "src/scripts/exp6/configs/exp2_eq32/full.yaml"
  "src/scripts/exp6/configs/exp3_sampling/full.yaml"
  "src/scripts/exp6/configs/exp5_nonstationary/full.yaml"
  "src/scripts/exp6/configs/exp5_lr_scaling/full.yaml"
  "src/scripts/exp6/configs/exp6_generalization/full.yaml"
  "src/scripts/exp6/configs/exp7_diffusion_control/full.yaml"
  "src/scripts/exp6/configs/exp8_diffusion_amplification/full.yaml"
  "src/scripts/exp6/configs/exp9_theory_denoised_sgd/full.yaml"
  "src/scripts/exp6/configs/exp10_noise_alpha_sweep/full.yaml"
  "src/scripts/exp6/configs/exp11_normalized_noise_alpha_sweep/full.yaml"
)

if [[ "${RUN_LONG}" == "1" ]]; then
  CONFIGS+=("src/scripts/exp6/configs/exp4_scaling/full.yaml")
else
  echo "[skip] EXP4 full scaling is long; set RUN_LONG=1 to include it."
fi

for cfg in "${CONFIGS[@]}"; do
  echo "=== full: ${cfg} ==="
  python -m src.scripts.exp6.src.run_experiment "${cfg}" --make-figure
done

python -m src.scripts.exp6.src.summarize --results-root src/scripts/exp6/results --output src/scripts/exp6/RESULTS_SUMMARY.md
