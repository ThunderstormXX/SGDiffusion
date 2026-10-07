#!/usr/bin/env bash
# The whole experiment, with automatic restarts (every stage resumes from its own checkpoint).
#
#   bash run_experiment.sh [OUT_DIR] [extra pipeline arguments]
#
# Inside tmux for a long run:
#   tmux new -d -s nanogpt 'bash run_experiment.sh runs/nanogpt6m 2>&1 | tee -a runs/nanogpt6m.log'
#
# Environment: GPUS (default: the single GPU with the most free memory; "0 1" splits the exact
# Hessian rounds over two GPUs). The reported run used one GPU and N_TRAJ=240 trajectories
# per learning rate, produced by a single trajectory worker (worker 0), which fixes the seeds.
set -uo pipefail
cd "$(dirname "$0")"
OUT=${1:-runs/nanogpt6m}; shift || true
EXTRA=("$@")
N_TRAJ=240
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -z "${GPUS:-}" ]; then
  GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
fi
read -r -a GPU <<< "$GPUS"
NG=${#GPU[@]}
echo "GPUs: ${GPU[*]}, out $OUT, $N_TRAJ trajectories per learning rate"

py() {   # py GPU args...
  local g=$1; shift
  CUDA_VISIBLE_DEVICES=$g python pipeline.py --out "$OUT" "${EXTRA[@]}" "$@"
}

run() {   # rerun a command until it succeeds
  local n=0
  until "$@"; do
    n=$((n + 1))
    if [ "$n" -ge "${MAX_RETRIES:-20}" ]; then
      echo "[supervisor] giving up after $n failures: $*"; return 1
    fi
    echo "[supervisor] $(date +%H:%M:%S) failed (attempt $n), restarting in ${RETRY_WAIT:-60}s: $*"
    sleep "${RETRY_WAIT:-60}"
  done
}

wait_all() { local rc=0 p; for p in "$@"; do wait "$p" || rc=1; done; return $rc; }

# 1. reference point w*: SGD, low-noise SGD, full-gradient descent; candidate basis by Lanczos
run py "${GPU[0]}" --stages data,train,anneal,refine,lanczos || exit 1

# 2. exact Hessian rounds (one worker per GPU), Rayleigh-Ritz after each round
while [ ! -f "$OUT/spectrum.pt" ]; do
  pids=()
  for ((p = 0; p < NG; p++)); do
    ( run py "${GPU[p]}" --stages exact --part "$p" --parts "$NG" ) >> "$OUT/exact_$p.log" 2>&1 &
    pids+=($!)
  done
  wait_all "${pids[@]}" || exit 1
  run py "${GPU[0]}" --stages ritz || exit 1
done

# 3. learning rates from the measured lambda_max; noise statistics alongside the trajectories
run py "${GPU[0]}" --stages plan || exit 1
( run py "${GPU[0]}" --stages stats,refine_ref ) >> "$OUT/stats.log" 2>&1 &
STATS_PID=$!
run py "${GPU[0]}" --stages traj --worker 0 --n_local "$N_TRAJ" || exit 1
wait "$STATS_PID" || exit 1

# 4. drift check, report, figure
run py "${GPU[0]}" --stages drift,analyze || exit 1
python make_figure.py "$OUT" "$OUT"
echo "[supervisor] $(date +%H:%M:%S) finished: $OUT/report.txt, $OUT/fig_nanogpt_finite_step.pdf"
