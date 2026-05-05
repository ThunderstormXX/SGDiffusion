"""
Tunnel-based experiment runner.

A tunnel is a stage in the pipeline with multiple parallel runs.
Each tunnel can continue from a previous tunnel using either:
- 1to1 mode: run_i continues from source_tunnel/run_i
- cartesian mode: cartesian product from previous tunnel runs
"""

import gc
from pathlib import Path

import torch
from tqdm import tqdm

from ..core.checkpointing import load_model
from ..core.config import ExperimentConfig, TunnelConfig
from ..core.registry import registry
from ..core.seed import set_seed
from .pipeline import StageRunner


class TunnelRunner:
    """
    Runs a single tunnel with multiple parallel runs.

    Encapsulates:
    - Loading checkpoints from source tunnel (if any)
    - Running multiple seeds in parallel
    - Saving results to tunnel-specific directories

    Principles:
    - Encapsulation: All tunnel logic is self-contained
    - Composition: Uses StageRunner internally
    - Abstraction: Hides complexity of checkpoint loading modes
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        tunnel_config: TunnelConfig,
        experiment_dir: Path,
    ):
        """
        Initialize tunnel runner.

        Args:
            experiment_config: Full experiment configuration
            tunnel_config: Configuration for this specific tunnel
            experiment_dir: Base directory for experiment (e.g., mnist_manysgd_small/)
        """
        self.experiment_config = experiment_config
        self.tunnel_config = tunnel_config
        self.experiment_dir = Path(experiment_dir)
        self.tunnel_dir = self.experiment_dir / tunnel_config.name

        # Create tunnel directory
        self.tunnel_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        n_runs: int,
        seeds: list[int],
        device: torch.device,
        dtype: torch.dtype,
        verbose: bool = True,
        debug: bool = False,
        n_initial_weights: int = 1,
    ) -> None:
        """
        Run this tunnel for multiple seeds.

        For the first tunnel (no source), n_initial_weights controls how many
        different initial weights are sampled. Each initial weight spawns n_runs
        trajectories. Total runs = n_initial_weights * n_runs.

        For subsequent tunnels, n_initial_weights is ignored (grouping is determined
        by source runs).

        Args:
            n_runs: Number of runs per initial weight (for first tunnel) or per source point
            seeds: List of random seeds (length must match total_runs)
            device: Torch device
            dtype: Torch dtype
            verbose: Print progress info
            debug: Show detailed tqdm for each training step
            n_initial_weights: Number of different initial weights (only for first tunnel)
        """
        # For first tunnel: total = n_initial_weights * n_runs
        # For cartesian: total = n_source_runs * n_runs (per source)
        # For 1to1: total = n_runs
        is_first_tunnel = self.tunnel_config.source_mode is None

        if is_first_tunnel:
            total_runs = n_initial_weights * n_runs
            effective_n_initial_weights = n_initial_weights
        elif self.tunnel_config.source_mode == "cartesian":
            n_source_runs = self._count_source_runs()
            if n_source_runs == 0:
                raise ValueError("No runs found in previous tunnel for cartesian mode")
            total_runs = n_source_runs * n_runs
            effective_n_initial_weights = 1
        else:
            total_runs = n_runs
            effective_n_initial_weights = 1  # Not used for subsequent tunnels
        
        if len(seeds) != total_runs:
            raise ValueError(f"Number of seeds ({len(seeds)}) must match total_runs ({total_runs})")

        if verbose:
            print("=" * 70)
            print(f"TUNNEL {self.tunnel_config.tunnel_index}: {self.tunnel_config.name}")
            print("=" * 70)
            print(f"  Description: {self.tunnel_config.description}")
            print(f"  Mode:        {self.tunnel_config.mode}")
            if self.tunnel_config.mode == "steps":
                print(f"  Steps:       {self.tunnel_config.steps}")
            else:
                print(f"  Epochs:      {self.tunnel_config.epochs}")
            print(f"  Optimizer:   {self.tunnel_config.optimizer.name}")
            print(f"  LR:          {self.tunnel_config.get_learning_rate()}")

            # Detailed DataLoader info
            data_kwargs = self.experiment_config.data.kwargs
            sample_size = data_kwargs.get("sample_size", "full")
            # Use tunnel batch_size if set, otherwise from data config
            batch_size = self.tunnel_config.batch_size or data_kwargs.get("batch_size", 64)
            replacement = data_kwargs.get("replacement", False)
            replacement_str = "with replacement" if replacement else "without replacement"

            if self.tunnel_config.dataloader_mode == "fullbatch":
                print("  Dataloader:  fullbatch (GD mode)")
                print(f"    - samples:     {sample_size}")
                print(f"    - batch_size:  {batch_size} (full dataset)")
            else:
                print(f"  Dataloader:  minibatch ({replacement_str})")
                print(f"    - samples:     {sample_size}")
                print(f"    - batch_size:  {batch_size}")

            if is_first_tunnel and effective_n_initial_weights > 1:
                print(f"  N_initial_weights: {effective_n_initial_weights}")
                print(f"  N_runs_per_init:   {n_runs}")
                print(f"  Total_runs:        {total_runs}")
            else:
                print(f"  N_runs:      {total_runs}")
            print(f"  Seeds:       {seeds[0]} - {seeds[-1]}")
            if self.tunnel_config.source_mode:
                print(f"  Source mode: {self.tunnel_config.source_mode}")
            print("=" * 70)

        # Build data module once (shared across runs)
        data_factory = registry.get_data(self.experiment_config.data.name)
        data_module = data_factory(**self.experiment_config.data.kwargs)
        data_module.setup()

        # Build model factory class
        model_factory_class = registry.get_model(self.experiment_config.model.name)

        # Determine task type
        if self.experiment_config.model.name == "nanogpt":
            from ..training.tasks import LanguageModelingTask
            task = LanguageModelingTask()
        else:
            from ..training.tasks import ClassificationTask
            task = ClassificationTask()

        # Convert TunnelConfig to StageConfig for StageRunner compatibility
        from ..core.config import StageConfig
        stage_config = StageConfig(
            name=self.tunnel_config.name,
            mode=self.tunnel_config.mode,
            steps=self.tunnel_config.steps,
            epochs=self.tunnel_config.epochs,
            optimizer=self.tunnel_config.optimizer,
            dataloader_mode=self.tunnel_config.dataloader_mode,
            batch_size=self.tunnel_config.batch_size,
            trackers=self.tunnel_config.trackers,
            eval_every=self.tunnel_config.eval_every,
            save_checkpoint="checkpoint.pt" if self.tunnel_config.save_checkpoint else None,
        )

        # Run for each seed
        iterator = tqdm(range(total_runs), desc=f"Tunnel {self.tunnel_config.tunnel_index}") if verbose else range(total_runs)

        if is_first_tunnel and effective_n_initial_weights > 1:
            # First tunnel with multiple initial weights
            # Group runs by initial_weight_index
            for run_index in iterator:
                initial_weight_idx = run_index // n_runs  # Which initial weight this run belongs to
                run_within_group = run_index % n_runs  # Index within the group
                seed = seeds[run_index]
                
                run_dir = self.tunnel_dir / f"run_{run_index}"
                run_dir.mkdir(parents=True, exist_ok=True)

                # Use initial_weight_idx as the model initialization seed
                # This ensures all runs in the same group start from the same initial weights
                init_seed = seeds[initial_weight_idx * n_runs]
                set_seed(init_seed)

                # Build fresh model with deterministic initialization
                model_factory = model_factory_class(**self.experiment_config.model.kwargs)
                model = model_factory.build(device=device, dtype=dtype)

                # Create stage runner
                stage_runner = StageRunner(
                    model=model,
                    data_module=data_module,
                    task=task,
                    device=device,
                    dtype=dtype,
                    run_dir=str(run_dir),
                )

                # Run training with this run's seed for data sampling
                stage_runner.run(
                    stage_config=stage_config,
                    stage_idx=self.tunnel_config.tunnel_index,
                    global_step=0,
                    progress=debug,
                    verbose=debug,
                    seed=seed,  # Each run uses different seed for data sampling
                )

                # Save run metadata with grouping info
                self._save_run_metadata(
                    run_dir, run_index, seed, None,
                    initial_weight_index=initial_weight_idx,
                    run_within_group=run_within_group,
                    n_runs_per_initial_weight=n_runs,
                    n_initial_weights=effective_n_initial_weights,
                )
        else:
            # Standard execution (no initial weight grouping or subsequent tunnel)
            # For cartesian mode: n_runs is per source point, total = n_source_runs × n_runs
            n_runs_per_source = n_runs  # How many runs we spawn from each source
            
            for run_index in iterator:
                seed = seeds[run_index]
                run_dir = self.tunnel_dir / f"run_{run_index}"
                run_dir.mkdir(parents=True, exist_ok=True)

                # Set seed
                set_seed(seed)

                # Build fresh model
                model_factory = model_factory_class(**self.experiment_config.model.kwargs)
                model = model_factory.build(device=device, dtype=dtype)

                # Load checkpoint from source tunnel if needed
                source_run_index = None
                run_within_source_group = None
                if self.tunnel_config.source_mode == "1to1":
                    source_run_index = run_index
                elif self.tunnel_config.source_mode == "cartesian":
                    source_run_index = run_index // n_runs_per_source
                    run_within_source_group = run_index % n_runs_per_source

                checkpoint_path = self._get_source_checkpoint(source_run_index)
                
                if checkpoint_path is not None:
                    if checkpoint_path.exists():
                        load_model(str(checkpoint_path), model, map_location=str(device))
                    else:
                        if verbose:
                            tqdm.write(f"  [SKIP] Missing checkpoint for run {run_index}: {checkpoint_path}")
                        continue

                # Create stage runner
                stage_runner = StageRunner(
                    model=model,
                    data_module=data_module,
                    task=task,
                    device=device,
                    dtype=dtype,
                    run_dir=str(run_dir),
                )

                # Run training
                stage_runner.run(
                    stage_config=stage_config,
                    stage_idx=self.tunnel_config.tunnel_index,
                    global_step=0,
                    progress=debug,
                    verbose=debug,
                    seed=seed,  # Each run uses different seed for data sampling
                )

                # Save run metadata with source info
                self._save_run_metadata(
                    run_dir, run_index, seed, checkpoint_path,
                    source_run_index=source_run_index,
                    run_within_group=run_within_source_group,
                    n_runs_per_initial_weight=n_runs_per_source if self.tunnel_config.source_mode == "cartesian" else None,
                )

        # Save tunnel-level metadata
        self._save_tunnel_metadata()

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            print()
            print("=" * 70)
            print(f"TUNNEL {self.tunnel_config.tunnel_index} COMPLETE")
            print("=" * 70)
            print(f"Results: {self.tunnel_dir}/run_*/")
            print("=" * 70)

    def _get_source_checkpoint(self, source_run_index: int | None) -> Path | None:
        """
        Get checkpoint path from source tunnel based on source_mode.

        Args:
            source_run_index: Index of source run

        Returns:
            Path to checkpoint or None if no source
        """
        if self.tunnel_config.source_mode is None or source_run_index is None:
            return None

        # Get source tunnel directory
        source_tunnel_index = self.tunnel_config.tunnel_index - 1
        if source_tunnel_index < 0:
            raise ValueError(f"Cannot load from source: tunnel {self.tunnel_config.tunnel_index} has no previous tunnel")

        source_tunnel_dir = self.experiment_dir / f"tunnel_{source_tunnel_index}"

        source_run_dir = source_tunnel_dir / f"run_{source_run_index}"

        checkpoint_path = source_run_dir / "checkpoint.pt"
        return checkpoint_path

    def _count_source_runs(self) -> int:
        """Count number of runs in the source tunnel."""
        source_tunnel_index = self.tunnel_config.tunnel_index - 1
        if source_tunnel_index < 0:
            return 0
        source_tunnel_dir = self.experiment_dir / f"tunnel_{source_tunnel_index}"
        if not source_tunnel_dir.exists():
            return 0
        return len(list(source_tunnel_dir.glob("run_*")))

    def _extract_source_run_index(self, checkpoint_path: Path) -> int | None:
        """Extract source run index from checkpoint path."""
        # Path is like .../tunnel_X/run_Y/checkpoint.pt
        try:
            run_dir_name = checkpoint_path.parent.name  # "run_Y"
            if run_dir_name.startswith("run_"):
                return int(run_dir_name.split("_")[1])
        except (ValueError, IndexError):
            pass
        return None

    def _save_run_metadata(
        self,
        run_dir: Path,
        run_index: int,
        seed: int,
        source_checkpoint: Path | None,
        initial_weight_index: int | None = None,
        run_within_group: int | None = None,
        n_runs_per_initial_weight: int | None = None,
        n_initial_weights: int | None = None,
        source_run_index: int | None = None,
    ) -> None:
        """Save metadata for a single run including grouping information."""
        import json
        from datetime import datetime

        metadata = {
            "tunnel_index": self.tunnel_config.tunnel_index,
            "tunnel_name": self.tunnel_config.name,
            "run_index": run_index,
            "seed": seed,
            "description": self.tunnel_config.description,
            "mode": self.tunnel_config.mode,
            "steps": self.tunnel_config.steps,
            "epochs": self.tunnel_config.epochs,
            "optimizer": self.tunnel_config.optimizer.to_dict(),
            "learning_rate": self.tunnel_config.get_learning_rate(),
            "dataloader_mode": self.tunnel_config.dataloader_mode,
            "batch_size": self.tunnel_config.batch_size,
            "source_mode": self.tunnel_config.source_mode,
            "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
            "created_at": datetime.now().isoformat(),
        }
        
        # Add grouping info for first tunnel with n_initial_weights
        if initial_weight_index is not None:
            metadata["grouping"] = {
                "initial_weight_index": initial_weight_index,
                "run_within_group": run_within_group,
                "n_runs_per_initial_weight": n_runs_per_initial_weight,
                "n_initial_weights": n_initial_weights,
            }
        
        # Add source run info for subsequent tunnels
        if source_run_index is not None:
            metadata["source_run_index"] = source_run_index

        with open(run_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_tunnel_metadata(self) -> None:
        """Save metadata for the entire tunnel."""
        import json
        from datetime import datetime

        metadata = {
            "tunnel_index": self.tunnel_config.tunnel_index,
            "tunnel_name": self.tunnel_config.name,
            "description": self.tunnel_config.description,
            "configuration": self.tunnel_config.to_dict(),
            "experiment_config": {
                "data": self.experiment_config.data.to_dict(),
                "model": self.experiment_config.model.to_dict(),
            },
            "created_at": datetime.now().isoformat(),
        }

        with open(self.tunnel_dir / "tunnel_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
