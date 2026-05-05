#!/usr/bin/env python3
"""Create exp6 summary table and RESULTS_SUMMARY.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.exp6.src.common import write_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, default="src/scripts/exp6/results")
    parser.add_argument("--output", type=str, default="src/scripts/exp6/RESULTS_SUMMARY.md")
    args = parser.parse_args()

    root = Path(args.results_root)
    rows = []
    lines = [
        "# EXP6 Results Summary",
        "",
        "This file is generated from saved `metrics.json` artifacts only.",
        "",
        "| Experiment | Result dir | Primary metrics | Pass |",
        "|---|---|---|---|",
    ]
    for metrics_path in sorted(root.glob("*/metrics.json")):
        result_dir = metrics_path.parent
        with open(metrics_path) as f:
            metrics = json.load(f)
        cfg_path = result_dir / "config.yaml"
        exp = result_dir.name
        primary = {k: v for k, v in metrics.items() if k != "pass"}
        compact = ", ".join(f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in list(primary.items())[:5])
        passed = metrics.get("pass")
        lines.append(f"| `{exp}` | `{result_dir}` | {compact} | `{passed}` |")
        row = {"experiment": exp, "result_dir": str(result_dir), "pass": passed}
        row.update(primary)
        rows.append(row)
        if cfg_path.exists():
            lines.append(f"<!-- config: {cfg_path} -->")
    lines.extend([
        "",
        "## Caveats",
        "",
        "- Smoke runs are deliberately small and validate the pipeline, not the final paper-scale claims.",
        "- EXP4 is implemented as a lightweight HVP diagnostic, not the requested 1M+ large-model experiment.",
        "- Metrics and figures are regenerated from raw saved artifacts; no manual figure editing is used.",
    ])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    if rows:
        all_keys = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        normalized = [{k: r.get(k, "") for k in all_keys} for r in rows]
        write_csv(root.parent / "tables" / "summary_table.csv", normalized)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
