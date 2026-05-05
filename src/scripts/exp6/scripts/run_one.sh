#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?Usage: run_one.sh CONFIG.yaml [--make-figure]}"
shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python -m src.scripts.exp6.src.run_experiment "${CONFIG}" "$@"

