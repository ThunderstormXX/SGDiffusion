#!/usr/bin/env python3
"""
Aggregate tensors from tunnel-based experiments.

Aggregates weights, hessians from all runs in each tunnel into single tensors.
Works with cartesian product (1×1×1000) and 1:1 modes.

Usage:
    python -m src.scripts.exp5.aggregate_tunnel \
        --exp_dir src/scripts/exp5/exp_results/mnist_manysgd_small_v2 \
        --output_dir src/scripts/exp5/exp_results/mnist_manysgd_small_v2/aggregated
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import contextlib

import torch


class TunnelAggregator:
    """
    Aggregates data from tunnel-based experiments.

    Supports:
    - Cartesian product mode (1×1×N)
    - 1:1 continuation mode (N×N×N)
    - Per-tunnel and combined aggregation
    """

    def __init__(self, exp_dir: Path, output_dir: Path):
        self.exp_dir = Path(exp_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Discover tunnels
        self.tunnels = self._discover_tunnels()

    def _discover_tunnels(self) -> list[dict[str, Any]]:
        """Discover all tunnels in the experiment directory."""
        tunnels = []

        for tunnel_dir in sorted(self.exp_dir.glob("tunnel_*")):
            if not tunnel_dir.is_dir():
                continue

            # Load tunnel metadata
            metadata_path = tunnel_dir / "tunnel_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            # Count runs
            runs = list(tunnel_dir.glob("run_*"))

            tunnels.append({
                "name": tunnel_dir.name,
                "path": tunnel_dir,
                "n_runs": len(runs),
                "metadata": metadata,
            })

        return tunnels

    def aggregate_all(self) -> None:
        """Aggregate all tunnels."""
        print("=" * 70)
        print("AGGREGATING TUNNEL DATA")
        print("=" * 70)
        print(f"Experiment: {self.exp_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Tunnels found: {len(self.tunnels)}")
        print("=" * 70)

        for tunnel_info in self.tunnels:
            print(f"\n▶ {tunnel_info['name']} ({tunnel_info['n_runs']} runs)")
            self._aggregate_tunnel(tunnel_info)

        # Save experiment summary
        self._save_experiment_summary()

        print("\n" + "=" * 70)
        print("✅ AGGREGATION COMPLETE")
        print("=" * 70)

    def _aggregate_tunnel(self, tunnel_info: dict[str, Any]) -> None:
        """Aggregate a single tunnel."""
        tunnel_path = tunnel_info["path"]
        tunnel_name = tunnel_info["name"]
        n_runs = tunnel_info["n_runs"]

        if n_runs == 0:
            print("  No runs found, skipping")
            return

        # Collect weights (try tunnel-specific naming first)
        tunnel_name = tunnel_info["name"]
        weights_data = self._collect_tensor(tunnel_path, f"weights_{tunnel_name}.pt", n_runs)
        if weights_data is None:
            weights_data = self._collect_tensor(tunnel_path, "weights.pt", n_runs)
        if weights_data is not None:
            self._save_aggregated_tensor(
                tensor=weights_data["tensor"],
                name=f"{tunnel_name}_weights",
                info={
                    "tunnel": tunnel_name,
                    "type": "weights",
                    "shape": list(weights_data["tensor"].shape),
                    "n_runs": weights_data["n_runs"],
                    "n_steps": weights_data["n_steps"],
                    "n_params": weights_data["n_params"],
                    "steps": weights_data.get("steps", []),
                },
            )

        # Collect hessians (try tunnel-specific naming first)
        hessians_data = self._collect_tensor(tunnel_path, f"hessians_{tunnel_name}.pt", n_runs)
        if hessians_data is None:
            hessians_data = self._collect_tensor(tunnel_path, "hessians.pt", n_runs)
        if hessians_data is not None:
            # Also aggregate eigenvalues
            self._save_aggregated_tensor(
                tensor=hessians_data["tensor"],
                name=f"{tunnel_name}_hessians",
                info={
                    "tunnel": tunnel_name,
                    "type": "hessians",
                    "shape": list(hessians_data["tensor"].shape),
                    "n_runs": hessians_data["n_runs"],
                    "n_steps": hessians_data["n_steps"],
                    "steps": hessians_data.get("steps", []),
                },
            )

            # Aggregate eigenvalues if available
            if "eigenvalues" in hessians_data:
                self._save_aggregated_tensor(
                    tensor=hessians_data["eigenvalues"],
                    name=f"{tunnel_name}_eigenvalues",
                    info={
                        "tunnel": tunnel_name,
                        "type": "eigenvalues",
                        "shape": list(hessians_data["eigenvalues"].shape),
                        "n_runs": hessians_data["n_runs"],
                        "n_steps": hessians_data["n_steps"],
                    },
                )

    def _collect_tensor(
        self,
        tunnel_path: Path,
        filename: str,
        n_runs: int,
    ) -> dict[str, Any] | None:
        """
        Collect tensors from all runs in a tunnel.

        Returns dict with:
        - tensor: aggregated tensor [n_runs, n_steps, ...]
        - n_runs, n_steps, n_params
        - steps: step indices
        """
        all_tensors = []
        all_steps = []
        all_eigenvalues = []

        for run_idx in range(n_runs):
            run_dir = tunnel_path / f"run_{run_idx}"
            tensor_path = run_dir / filename

            if not tensor_path.exists():
                continue

            data = torch.load(tensor_path, map_location="cpu")

            # Handle different data formats
            if isinstance(data, dict):
                if "weights" in data:
                    tensor = data["weights"]
                    steps = data.get("steps", list(range(tensor.shape[0])))
                elif "hessians" in data:
                    # Hessians are stored as list of tensors
                    hessian_list = data["hessians"]
                    if isinstance(hessian_list, list):
                        tensor = torch.stack(hessian_list)
                    else:
                        tensor = hessian_list
                    steps = data.get("steps", list(range(len(tensor))))

                    # Collect eigenvalues if available
                    if "eigenvalues" in data:
                        eig_list = data["eigenvalues"]
                        if isinstance(eig_list, list):
                            all_eigenvalues.append(torch.stack(eig_list))
                        else:
                            all_eigenvalues.append(eig_list)
                else:
                    continue
            else:
                tensor = data
                steps = list(range(tensor.shape[0]))

            all_tensors.append(tensor)
            all_steps.append(steps)

        if not all_tensors:
            return None

        # Stack all runs: [n_runs, n_steps, ...]
        try:
            stacked = torch.stack(all_tensors)
        except RuntimeError:
            # Different shapes, try padding
            max_steps = max(t.shape[0] for t in all_tensors)
            padded = []
            for t in all_tensors:
                if t.shape[0] < max_steps:
                    padding = torch.zeros(max_steps - t.shape[0], *t.shape[1:])
                    t = torch.cat([t, padding], dim=0)
                padded.append(t)
            stacked = torch.stack(padded)

        result = {
            "tensor": stacked,
            "n_runs": stacked.shape[0],
            "n_steps": stacked.shape[1],
            "n_params": stacked.shape[2] if stacked.ndim >= 3 else 0,
            "steps": all_steps[0] if all_steps else [],
        }

        # Add eigenvalues if collected
        if all_eigenvalues:
            with contextlib.suppress(RuntimeError):
                result["eigenvalues"] = torch.stack(all_eigenvalues)

        return result

    def _save_aggregated_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        info: dict[str, Any],
    ) -> None:
        """Save aggregated tensor and info file."""
        # Save tensor
        tensor_path = self.output_dir / f"{name}.pt"
        torch.save(tensor, str(tensor_path))

        # Save info
        info["created_at"] = datetime.now().isoformat()
        info["file"] = f"{name}.pt"
        info["file_size_mb"] = tensor_path.stat().st_size / (1024 * 1024)

        info_path = self.output_dir / f"{name}_info.txt"
        with open(info_path, "w") as f:
            f.write(f"Aggregated Tensor: {name}\n")
            f.write("=" * 50 + "\n")
            for key, value in info.items():
                if key == "steps" and len(str(value)) > 100:
                    f.write(f"{key}: [{value[0]}, ..., {value[-1]}] ({len(value)} steps)\n")
                else:
                    f.write(f"{key}: {value}\n")

        print(f"  ✓ Saved {name}.pt ({info['file_size_mb']:.2f} MB)")

    def _save_experiment_summary(self) -> None:
        """Save experiment summary."""
        summary = {
            "experiment_dir": str(self.exp_dir),
            "output_dir": str(self.output_dir),
            "n_tunnels": len(self.tunnels),
            "tunnels": [
                {
                    "name": t["name"],
                    "n_runs": t["n_runs"],
                    "description": t["metadata"].get("description", ""),
                }
                for t in self.tunnels
            ],
            "created_at": datetime.now().isoformat(),
        }

        with open(self.output_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Aggregate tunnel-based experiment data")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for aggregated data")

    args = parser.parse_args()

    aggregator = TunnelAggregator(
        exp_dir=Path(args.exp_dir),
        output_dir=Path(args.output_dir),
    )
    aggregator.aggregate_all()


if __name__ == "__main__":
    main()
