#!/usr/bin/env python3
"""
Aggregate tensors from multiple runs into single tensors.

Creates:
    - weights_{stage}.pt: (n_runs, n_steps, n_params)
    - hessians_{stage}.pt: (n_runs, n_steps, n_params, n_params)
    - grads_{stage}.pt: (n_runs, n_steps, n_params)
    - {name}_tensor_info.txt: Shape and metadata info

Usage:
    python -m src.scripts.exp5.aggregate_tensors \
        --input_dir src/scripts/exp5/exp_results/shakespeare_small \
        --n_runs 5 \
        --base_seed 42
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from tqdm import tqdm


def aggregate_weights(run_dirs: list, stage_name: str, output_dir: Path) -> dict:
    """Aggregate weight trajectories from multiple runs."""
    all_weights = []
    all_steps = []
    valid_runs = 0
    
    for run_dir in run_dirs:
        weights_path = run_dir / f"weights_{stage_name}.pt"
        if not weights_path.exists():
            continue
        
        data = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        weights = data.get("weights")
        steps = data.get("steps", [])
        
        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights = weights.numpy()
            all_weights.append(weights)
            all_steps.append(steps)
            valid_runs += 1
    
    if not all_weights:
        return None
    
    # Find minimum steps to align
    min_steps = min(w.shape[0] for w in all_weights)
    
    # Stack into (n_runs, n_steps, n_params)
    stacked = np.stack([w[:min_steps] for w in all_weights], axis=0)
    tensor = torch.from_numpy(stacked)
    
    # Save tensor
    output_path = output_dir / f"weights_{stage_name}.pt"
    torch.save(tensor, str(output_path))
    
    # Save info
    info = {
        "name": f"weights_{stage_name}",
        "shape": list(tensor.shape),
        "shape_description": "(n_runs, n_steps, n_params)",
        "n_runs": tensor.shape[0],
        "n_steps": tensor.shape[1],
        "n_params": tensor.shape[2],
        "dtype": str(tensor.dtype),
        "stage": stage_name,
        "valid_runs": valid_runs,
        "steps": all_steps[0][:min_steps] if all_steps else [],
        "created_at": datetime.now().isoformat(),
    }
    
    info_path = output_dir / f"weights_{stage_name}_tensor_info.txt"
    with open(info_path, "w") as f:
        f.write(f"Tensor: weights_{stage_name}.pt\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Shape: {tuple(tensor.shape)}\n")
        f.write(f"Description: (n_runs, n_steps, n_params)\n")
        f.write(f"  n_runs:   {tensor.shape[0]}\n")
        f.write(f"  n_steps:  {tensor.shape[1]}\n")
        f.write(f"  n_params: {tensor.shape[2]}\n")
        f.write(f"Dtype: {tensor.dtype}\n")
        f.write(f"Stage: {stage_name}\n")
        f.write(f"Valid runs: {valid_runs}\n")
        f.write(f"Created: {info['created_at']}\n")
    
    print(f"  [saved] {output_path} - shape {tuple(tensor.shape)}")
    print(f"  [saved] {info_path}")
    
    return info


def aggregate_hessians(run_dirs: list, stage_name: str, output_dir: Path) -> dict:
    """Aggregate Hessian tensors from multiple runs."""
    all_hessians = []
    all_steps = []
    valid_runs = 0
    
    for run_dir in run_dirs:
        hessian_path = run_dir / f"hessian_{stage_name}.pt"
        if not hessian_path.exists():
            continue
        
        data = torch.load(str(hessian_path), map_location="cpu", weights_only=False)
        hessians = data.get("hessians")
        steps = data.get("steps", [])
        
        if hessians is not None:
            if isinstance(hessians, list):
                # Convert list of hessians to tensor
                hessians = torch.stack([h if isinstance(h, torch.Tensor) else torch.from_numpy(h) for h in hessians])
            elif isinstance(hessians, np.ndarray):
                hessians = torch.from_numpy(hessians)
            
            all_hessians.append(hessians)
            all_steps.append(steps)
            valid_runs += 1
    
    if not all_hessians:
        return None
    
    # Find minimum steps
    min_steps = min(h.shape[0] for h in all_hessians)
    
    # Stack into (n_runs, n_steps, n_params, n_params)
    stacked = torch.stack([h[:min_steps] for h in all_hessians], dim=0)
    
    # Save tensor
    output_path = output_dir / f"hessians_{stage_name}.pt"
    torch.save(stacked, str(output_path))
    
    # Save info
    info_path = output_dir / f"hessians_{stage_name}_tensor_info.txt"
    with open(info_path, "w") as f:
        f.write(f"Tensor: hessians_{stage_name}.pt\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Shape: {tuple(stacked.shape)}\n")
        f.write(f"Description: (n_runs, n_steps, n_params, n_params)\n")
        f.write(f"  n_runs:   {stacked.shape[0]}\n")
        f.write(f"  n_steps:  {stacked.shape[1]}\n")
        f.write(f"  n_params: {stacked.shape[2]} x {stacked.shape[3]}\n")
        f.write(f"Dtype: {stacked.dtype}\n")
        f.write(f"Stage: {stage_name}\n")
        f.write(f"Valid runs: {valid_runs}\n")
        f.write(f"Created: {datetime.now().isoformat()}\n")
    
    print(f"  [saved] {output_path} - shape {tuple(stacked.shape)}")
    print(f"  [saved] {info_path}")
    
    return {"shape": list(stacked.shape), "valid_runs": valid_runs}


def aggregate_grads(run_dirs: list, stage_name: str, output_dir: Path) -> dict:
    """Aggregate gradient tensors from multiple runs."""
    all_grads = []
    all_steps = []
    valid_runs = 0
    
    for run_dir in run_dirs:
        grads_path = run_dir / f"grads_{stage_name}.pt"
        if not grads_path.exists():
            continue
        
        data = torch.load(str(grads_path), map_location="cpu", weights_only=False)
        grads = data.get("grads")
        steps = data.get("steps", [])
        
        if grads is not None:
            if isinstance(grads, torch.Tensor):
                grads = grads.numpy()
            all_grads.append(grads)
            all_steps.append(steps)
            valid_runs += 1
    
    if not all_grads:
        return None
    
    # Find minimum steps
    min_steps = min(g.shape[0] for g in all_grads)
    
    # Stack into (n_runs, n_steps, n_params)
    stacked = np.stack([g[:min_steps] for g in all_grads], axis=0)
    tensor = torch.from_numpy(stacked)
    
    # Save tensor
    output_path = output_dir / f"grads_{stage_name}.pt"
    torch.save(tensor, str(output_path))
    
    # Save info
    info_path = output_dir / f"grads_{stage_name}_tensor_info.txt"
    with open(info_path, "w") as f:
        f.write(f"Tensor: grads_{stage_name}.pt\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Shape: {tuple(tensor.shape)}\n")
        f.write(f"Description: (n_runs, n_steps, n_params)\n")
        f.write(f"  n_runs:   {tensor.shape[0]}\n")
        f.write(f"  n_steps:  {tensor.shape[1]}\n")
        f.write(f"  n_params: {tensor.shape[2]}\n")
        f.write(f"Dtype: {tensor.dtype}\n")
        f.write(f"Stage: {stage_name}\n")
        f.write(f"Valid runs: {valid_runs}\n")
        f.write(f"Created: {datetime.now().isoformat()}\n")
    
    print(f"  [saved] {output_path} - shape {tuple(tensor.shape)}")
    print(f"  [saved] {info_path}")
    
    return {"shape": list(tensor.shape), "valid_runs": valid_runs}


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate tensors from multiple runs",
    )
    
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Base directory containing run_seed* folders")
    parser.add_argument("--n_runs", type=int, required=True,
                        help="Number of runs")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: input_dir/aggregated)")
    parser.add_argument("--stages", type=str, nargs="+", 
                        default=["stage1_sgd", "stage2_gd", "stage3_sgd"],
                        help="Stage names to aggregate")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get run directories
    seeds = [args.base_seed + i for i in range(args.n_runs)]
    run_dirs = [input_dir / f"run_seed{seed}" for seed in seeds]
    
    # Check how many exist
    existing = [d for d in run_dirs if d.exists()]
    print("=" * 60)
    print("AGGREGATING TENSORS")
    print("=" * 60)
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Runs:   {len(existing)}/{len(run_dirs)} exist")
    print(f"  Stages: {args.stages}")
    print("=" * 60)
    
    if not existing:
        print("ERROR: No run directories found!")
        return
    
    # Aggregate for each stage
    for stage_name in args.stages:
        print(f"\n--- {stage_name} ---")
        
        # Weights
        weights_info = aggregate_weights(existing, stage_name, output_dir)
        
        # Hessians (usually only stage3)
        hessians_info = aggregate_hessians(existing, stage_name, output_dir)
        
        # Gradients
        grads_info = aggregate_grads(existing, stage_name, output_dir)
        
        if not weights_info and not hessians_info and not grads_info:
            print(f"  [skip] No data found for {stage_name}")
    
    print()
    print("=" * 60)
    print("AGGREGATION COMPLETE")
    print("=" * 60)
    print(f"Output: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
