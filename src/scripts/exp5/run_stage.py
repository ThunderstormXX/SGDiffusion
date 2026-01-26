#!/usr/bin/env python3
"""
Run a single stage of the tunnel experiment for multiple seeds.

This script runs ONE stage (SGD1, GD, or SGD2) for ALL seeds,
loading checkpoints from the previous stage if needed.

Usage:
    # Run stage 1 (SGD) for seeds 42-46
    python -m src.scripts.exp5.run_stage \
        --preset shakespeare_small \
        --stage 1 \
        --n_runs 5 \
        --base_seed 42

    # Run stage 2 (GD) using stage 1 checkpoints
    python -m src.scripts.exp5.run_stage \
        --preset shakespeare_small \
        --stage 2 \
        --n_runs 5 \
        --base_seed 42
"""

import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from src.datamodelopt.core.config import ExperimentConfig
from src.datamodelopt.core.seed import set_seed
from src.datamodelopt.core.registry import registry
from src.datamodelopt.core.checkpointing import load_model, save_model
from src.datamodelopt.experiments.pipeline import StageRunner
from src.datamodelopt.training.tasks import LanguageModelingTask, ClassificationTask


def main():
    parser = argparse.ArgumentParser(
        description="Run a single stage for multiple seeds",
    )
    
    parser.add_argument("--preset", type=str, required=True,
                        help="Preset name (e.g., shakespeare_small)")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3],
                        help="Stage number: 1=SGD1, 2=GD, 3=SGD2")
    parser.add_argument("--n_runs", type=int, required=True,
                        help="Number of runs")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base output directory")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, cuda, mps")
    parser.add_argument("--dtype", type=str, default="float32",
                        help="Data type")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bars")
    
    args = parser.parse_args()
    
    # Load preset config
    from src.scripts.exp5.presets import get_preset
    base_config = get_preset(args.preset)
    
    # Override device/dtype
    base_config.device = args.device
    base_config.dtype = args.dtype
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(base_config.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stage_idx = args.stage - 1  # 0-indexed
    stage_config = base_config.stages[stage_idx]
    stage_name = stage_config.name
    
    print("=" * 60)
    print(f"STAGE {args.stage}: {stage_name}")
    print("=" * 60)
    print(f"  Preset:  {args.preset}")
    print(f"  N_runs:  {args.n_runs}")
    print(f"  Seeds:   {args.base_seed} - {args.base_seed + args.n_runs - 1}")
    print(f"  Device:  {args.device}")
    print(f"  Output:  {output_dir}")
    print("=" * 60)
    
    # Get device and dtype
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    
    # Build data module once (shared across runs)
    data_factory = registry.get_data(base_config.data.name)
    data_module = data_factory(**base_config.data.kwargs)
    data_module.setup()
    
    # Build model factory class
    model_factory_class = registry.get_model(base_config.model.name)
    
    # Determine task type
    if base_config.model.name == "nanogpt":
        task = LanguageModelingTask()
    else:
        task = ClassificationTask()
    
    # Run stage for each seed
    seeds = [args.base_seed + i for i in range(args.n_runs)]
    
    iterator = tqdm(seeds, desc=f"Stage {args.stage}") if not args.no_progress else seeds
    
    for seed in iterator:
        run_dir = output_dir / f"run_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed
        set_seed(seed)
        
        # Build fresh model
        model_factory = model_factory_class(**base_config.model.kwargs)
        model = model_factory.build(device=device, dtype=dtype)
        
        # Load checkpoint from previous stage if needed
        if args.stage > 1:
            prev_stage_name = base_config.stages[stage_idx - 1].name
            checkpoint_path = run_dir / f"{prev_stage_name}.pt"
            
            if not checkpoint_path.exists():
                if not args.no_progress:
                    tqdm.write(f"  [SKIP] No checkpoint for seed {seed}: {checkpoint_path}")
                continue
            
            load_model(str(checkpoint_path), model, map_location=str(device))
        
        # Modify stage_config to set checkpoint path for saving
        stage_config.save_checkpoint = f"{stage_name}.pt"
        
        # Create stage runner
        stage_runner = StageRunner(
            model=model,
            data_module=data_module,
            task=task,
            device=device,
            dtype=dtype,
            run_dir=str(run_dir),
        )
        
        # Run stage (no inner progress bar, no verbose output)
        stage_runner.run(
            stage_config=stage_config,
            stage_idx=stage_idx,
            global_step=0,
            progress=False,
            verbose=False,
        )
        
        # Save config for this run (on first stage only)
        if args.stage == 1:
            config_dict = base_config.to_dict()
            config_dict['seed'] = seed
            with open(run_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)
    
    print()
    print("=" * 60)
    print(f"STAGE {args.stage} COMPLETE")
    print("=" * 60)
    print(f"Results: {output_dir}/run_seed*/")
    print("=" * 60)


if __name__ == "__main__":
    main()
