"""
Experiment pipeline and runner implementations.
"""

import gc
import os
from pathlib import Path

import torch
import torch.nn as nn

from ..core.checkpointing import load_model, save_model
from ..core.config import (
    ExperimentConfig,
    StageConfig,
    get_device,
    get_torch_dtype,
)
from ..core.registry import registry
from ..core.seed import set_seed
from ..data.base import DataModule
from ..models.factories import build_model_from_config
from ..optim.factories import build_optimizer_from_config
from ..tracking.base import Tracker
from ..training.tasks import ClassificationTask, LanguageModelingTask, Task
from ..training.trainer import Trainer, TrainingContext


class StageRunner:
    """
    Runs a single training stage.

    Handles:
        - Loading checkpoint if provided
        - Building optimizer for the stage
        - Creating data loaders (minibatch or fullbatch)
        - Training for steps or epochs
        - Calling tracker hooks
        - Saving checkpoint at end
    """

    def __init__(
        self,
        model: nn.Module,
        data_module: DataModule,
        task: Task,
        device: torch.device,
        dtype: torch.dtype,
        run_dir: str,
    ):
        """
        Initialize the stage runner.

        Args:
            model: The model to train.
            data_module: Data module for loading data.
            task: The training task.
            device: Training device.
            dtype: Training dtype.
            run_dir: Directory for saving outputs.
        """
        self.model = model
        self.data_module = data_module
        self.task = task
        self.device = device
        self.dtype = dtype
        self.run_dir = run_dir

    def run(
        self,
        stage_config: StageConfig,
        stage_idx: int,
        global_step: int = 0,
        progress: bool = True,
        verbose: bool = True,
        seed: int | None = None,
    ) -> int:
        """
        Run a training stage.

        Args:
            stage_config: Configuration for this stage.
            stage_idx: Index of this stage (0-based).
            global_step: Starting global step.
            progress: Show progress bars.
            verbose: Print detailed status messages.
            seed: Random seed for data sampling (each run should have different seed).

        Returns:
            Updated global step after this stage.
        """
        if verbose:
            print(f"\n=== Stage {stage_idx}: {stage_config.name} ===")

        # Load checkpoint if specified
        if stage_config.load_checkpoint:
            if verbose:
                print(f"Loading checkpoint from: {stage_config.load_checkpoint}")
            load_model(
                stage_config.load_checkpoint,
                self.model,
                map_location=str(self.device),
            )

        # Build optimizer
        optimizer = build_optimizer_from_config(
            stage_config.optimizer,
            self.model.parameters(),
        )

        # Create data loaders
        fullbatch = stage_config.dataloader_mode == "fullbatch"
        batch_size = stage_config.batch_size or self.data_module.batch_size

        train_loader = self.data_module.train_loader(
            batch_size=batch_size,
            fullbatch=fullbatch,
            seed=seed,  # Different seed for each run!
        )
        val_loader = self.data_module.val_loader()

        if verbose:
            if fullbatch:
                print(f"Using fullbatch mode (GD): batch_size={len(self.data_module.train_dataset)}")
            else:
                print(f"Using minibatch mode: batch_size={batch_size}")

        # Create trainer
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            task=self.task,
            device=self.device,
            dtype=self.dtype,
        )

        # Create training context
        ctx = TrainingContext(
            model=self.model,
            optimizer=optimizer,
            device=self.device,
            dtype=self.dtype,
            run_dir=self.run_dir,
        )
        ctx.stage_name = stage_config.name
        ctx.stage_idx = stage_idx
        ctx.global_step = global_step
        ctx.train_loader = train_loader
        ctx.val_loader = val_loader

        # Build trackers
        trackers = self._build_trackers(stage_config)

        # Call tracker on_stage_start
        for tracker in trackers:
            tracker.on_stage_start(ctx)

        # Evaluation function
        def eval_fn():
            return trainer.evaluate(train_loader, val_loader)

        # Train
        if stage_config.mode == "steps":
            if verbose:
                print(f"Training for {stage_config.steps} steps...")
            trainer.train_steps(
                train_loader=train_loader,
                n_steps=stage_config.steps,
                ctx=ctx,
                trackers=trackers,
                eval_fn=eval_fn,
                eval_every=stage_config.eval_every,
                progress=progress,
            )
        else:  # epochs
            if verbose:
                print(f"Training for {stage_config.epochs} epochs...")
            trainer.train_epochs(
                train_loader=train_loader,
                n_epochs=stage_config.epochs,
                ctx=ctx,
                trackers=trackers,
                eval_fn=eval_fn,
                eval_every=stage_config.eval_every,
                progress=progress,
            )

        # Final evaluation
        train_metrics, val_metrics = eval_fn()
        ctx.train_eval = train_metrics
        ctx.val_eval = val_metrics
        ctx.eval_step = ctx.step_in_stage
        ctx.eval_global_step = ctx.global_step
        ctx.eval_epoch = ctx.epoch if ctx.epoch else None
        ctx.eval_counter += 1
        if verbose:
            print(f"Stage {stage_config.name} complete:")
            print(f"  Train loss: {train_metrics['loss']:.4f}, acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val loss: {val_metrics['loss']:.4f}, acc: {val_metrics['accuracy']:.4f}")

        # Call tracker on_stage_end
        for tracker in trackers:
            tracker.on_stage_end(ctx)

        # Flush trackers and free memory
        for tracker in trackers:
            tracker.flush(self.run_dir)

        # Force garbage collection to free memory from flushed tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save checkpoint if specified
        if stage_config.save_checkpoint:
            checkpoint_path = os.path.join(self.run_dir, stage_config.save_checkpoint)
            save_model(
                checkpoint_path,
                self.model,
                extra={
                    "stage_name": stage_config.name,
                    "stage_idx": stage_idx,
                    "global_step": ctx.global_step,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
            )

        return ctx.global_step

    def _build_trackers(self, stage_config: StageConfig) -> list[Tracker]:
        """Build trackers from stage config."""
        trackers = []
        for tracker_config in stage_config.trackers:
            tracker_cls = registry.get_tracker(tracker_config.name)
            tracker = tracker_cls(**tracker_config.kwargs)
            trackers.append(tracker)
        return trackers


class ExperimentRunner:
    """
    Runs a full experiment with multiple stages.

    Handles:
        - Setting up run directory
        - Seeding
        - Building data module and model once
        - Running stages sequentially
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize the experiment runner.

        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.model: nn.Module | None = None
        self.data_module: DataModule | None = None
        self.task: Task | None = None

    def setup(self) -> None:
        """Set up the experiment (directories, seed, data, model)."""
        # Create run directory
        Path(self.config.run_dir).mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = os.path.join(self.config.run_dir, "config.json")
        self.config.save_json(config_path)
        print(f"Saved config to: {config_path}")

        # Set seed
        set_seed(self.config.seed)
        print(f"Set seed: {self.config.seed}")

        # Get device and dtype
        self.device = get_device(self.config.device)
        self.dtype = get_torch_dtype(self.config.dtype)
        print(f"Device: {self.device}, dtype: {self.dtype}")

        # Build data module
        data_cls = registry.get_data(self.config.data.name)
        self.data_module = data_cls(
            seed=self.config.seed,
            **self.config.data.kwargs,
        )
        self.data_module.setup()
        print(f"Data module: {self.config.data.name}, train_size={len(self.data_module.train_dataset)}")

        # Build model
        self.model = build_model_from_config(
            self.config.model,
            device=self.device,
            dtype=self.dtype,
        )
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {self.config.model.name}, params={n_params}")

        # Determine task type
        # Use LanguageModelingTask for NanoGPT-like models, ClassificationTask otherwise
        model_name = self.config.model.name.lower()
        if "gpt" in model_name or "lm" in model_name or "nanogpt" in model_name:
            self.task = LanguageModelingTask()
            print("Task: LanguageModelingTask")
        else:
            self.task = ClassificationTask()
            print("Task: ClassificationTask")

    def run(self, progress: bool = True) -> None:
        """
        Run the full experiment.

        Args:
            progress: Show progress bars.
        """
        if self.model is None:
            self.setup()

        print(f"\n{'='*60}")
        print(f"Starting experiment: {self.config.run_dir}")
        print(f"{'='*60}")

        # Create stage runner
        stage_runner = StageRunner(
            model=self.model,
            data_module=self.data_module,
            task=self.task,
            device=self.device,
            dtype=self.dtype,
            run_dir=self.config.run_dir,
        )

        # Run stages
        global_step = 0
        for idx, stage_config in enumerate(self.config.stages):
            global_step = stage_runner.run(
                stage_config=stage_config,
                stage_idx=idx,
                global_step=global_step,
                progress=progress,
            )

        print(f"\n{'='*60}")
        print("Experiment complete!")
        print(f"Results saved to: {self.config.run_dir}")
        print(f"Total steps: {global_step}")
        print(f"{'='*60}")

    @classmethod
    def from_json(cls, path: str) -> "ExperimentRunner":
        """
        Create an ExperimentRunner from a JSON config file.

        Args:
            path: Path to the JSON config file.

        Returns:
            An ExperimentRunner instance.
        """
        config = ExperimentConfig.load_json(path)
        return cls(config)
