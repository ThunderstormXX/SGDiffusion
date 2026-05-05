#!/usr/bin/env python3
"""Run one exp6 experiment from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.exp6.src.common import (  # noqa: E402
    copy_make_figure,
    load_yaml,
    run_context,
    save_json,
    set_reproducible,
    write_experiment_readme,
)
from src.scripts.exp6.src.experiments import RUNNERS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one reproducible exp6 experiment")
    parser.add_argument("config", type=str, help="Path to config.yaml")
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--make-figure", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    exp_id = config["experiment_id"]
    if exp_id not in RUNNERS:
        raise ValueError(f"Unknown experiment_id={exp_id}; available={sorted(RUNNERS)}")

    results_root = Path(args.results_root) if args.results_root else Path(config.get("results_root", "src/scripts/exp6/results"))
    result_dir = results_root / config["result_name"]
    set_reproducible(int(config.get("seed", 42)), bool(config.get("deterministic", True)))

    with run_context(result_dir, config):
        metrics = RUNNERS[exp_id](config, result_dir)

    save_json(result_dir / "metrics.json", metrics)
    copy_make_figure(result_dir)
    write_experiment_readme(result_dir, config, metrics)

    if args.make_figure:
        import subprocess

        subprocess.check_call([sys.executable, str(result_dir / "make_figure.py"), str(result_dir)])

    print(f"[done] {exp_id} -> {result_dir}")
    print(f"[metrics] {result_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
