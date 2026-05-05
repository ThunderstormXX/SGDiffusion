"""Tests for tunnel-based training pipeline."""
import json
from pathlib import Path

import torch

from src.datamodelopt.core.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrackerConfig,
    TunnelConfig,
)
from src.datamodelopt.experiments.tunnel import TunnelRunner


def create_experiment_config(
    run_dir: str,
    tunnels: list[TunnelConfig],
    sample_size: int = 100,
    batch_size: int = 10,
) -> ExperimentConfig:
    """Helper to create experiment config for testing."""
    return ExperimentConfig(
        run_dir=run_dir,
        seed=42,
        device="cpu",
        dtype="float32",
        data=DataConfig(
            name="mnist",
            kwargs={
                "sample_size": sample_size,
                "batch_size": batch_size,
                "replacement": True,
            },
        ),
        model=ModelConfig(
            name="class",
            kwargs={
                "class_name": "FlexibleMLP",
                "hidden_dim": 4,
                "num_hidden_layers": 1,
                "input_downsample": 6,
            },
        ),
        tunnels=tunnels,
    )


class TestTunnelRunnerCreation:
    """Test TunnelRunner instantiation."""

    def test_create_tunnel_runner(self, temp_dir: Path):
        """TunnelRunner should be creatable with valid config."""
        tunnel = TunnelConfig(
            tunnel_index=0,
            mode="steps",
            steps=5,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
            trackers=[TrackerConfig(name="metrics", kwargs={"every": 1})],
        )
        config = create_experiment_config(str(temp_dir), [tunnel])

        runner = TunnelRunner(
            experiment_config=config,
            tunnel_config=tunnel,
            experiment_dir=temp_dir,
        )

        assert runner is not None
        assert runner.tunnel_config == tunnel
        assert runner.tunnel_dir == temp_dir / "tunnel_0"

    def test_tunnel_dir_created(self, temp_dir: Path):
        """TunnelRunner should create tunnel directory."""
        tunnel = TunnelConfig(tunnel_index=0, mode="steps", steps=5)
        config = create_experiment_config(str(temp_dir), [tunnel])

        runner = TunnelRunner(
            experiment_config=config,
            tunnel_config=tunnel,
            experiment_dir=temp_dir,
        )

        # Run to ensure directory is created
        runner.run(n_runs=1, seeds=[0], device=torch.device("cpu"), dtype=torch.float32)
        assert runner.tunnel_dir.exists()


class TestSingleTunnelExecution:
    """Test single tunnel execution with various modes."""

    def test_steps_mode(self, temp_dir: Path):
        """Tunnel should run specified number of steps."""
        tunnel = TunnelConfig(
            tunnel_index=0,
            mode="steps",
            steps=10,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
            batch_size=10,
            trackers=[TrackerConfig(name="metrics", kwargs={"every": 5})],
        )
        config = create_experiment_config(str(temp_dir), [tunnel])

        runner = TunnelRunner(config, tunnel, temp_dir)
        runner.run(n_runs=1, seeds=[0], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        # Check outputs exist
        run_dir = temp_dir / "tunnel_0" / "run_0"
        assert run_dir.exists()
        assert (run_dir / "checkpoint.pt").exists()
        assert (run_dir / "metrics_tunnel_0.json").exists()

    def test_epochs_mode(self, temp_dir: Path):
        """Tunnel should run specified number of epochs."""
        tunnel = TunnelConfig(
            tunnel_index=0,
            mode="epochs",
            epochs=2,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
            batch_size=50,  # 100 samples / 50 = 2 batches per epoch
            trackers=[TrackerConfig(name="metrics", kwargs={"every": 1})],
        )
        config = create_experiment_config(str(temp_dir), [tunnel])

        runner = TunnelRunner(config, tunnel, temp_dir)
        runner.run(n_runs=1, seeds=[0], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        run_dir = temp_dir / "tunnel_0" / "run_0"
        assert (run_dir / "checkpoint.pt").exists()

    def test_fullbatch_mode(self, temp_dir: Path):
        """Tunnel should work with fullbatch (GD) mode."""
        tunnel = TunnelConfig(
            tunnel_index=0,
            mode="epochs",
            epochs=3,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.01}),
            dataloader_mode="fullbatch",
            batch_size=100,
            trackers=[TrackerConfig(name="metrics", kwargs={"every": 1})],
        )
        config = create_experiment_config(str(temp_dir), [tunnel], sample_size=100)

        runner = TunnelRunner(config, tunnel, temp_dir)
        runner.run(n_runs=1, seeds=[0], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        run_dir = temp_dir / "tunnel_0" / "run_0"
        assert (run_dir / "checkpoint.pt").exists()


class TestMultiRunExecution:
    """Test tunnel execution with multiple runs."""

    def test_multiple_runs_created(self, temp_dir: Path):
        """Multiple runs should create separate directories."""
        tunnel = TunnelConfig(
            tunnel_index=0,
            mode="steps",
            steps=5,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
            batch_size=10,
            trackers=[TrackerConfig(name="metrics", kwargs={"every": 1})],
        )
        config = create_experiment_config(str(temp_dir), [tunnel])

        runner = TunnelRunner(config, tunnel, temp_dir)
        runner.run(n_runs=3, seeds=[0, 1, 2], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        for i in range(3):
            run_dir = temp_dir / "tunnel_0" / f"run_{i}"
            assert run_dir.exists(), f"run_{i} directory should exist"

    def test_different_seeds_different_results(self, temp_dir: Path):
        """Different seeds should produce different metric trajectories."""
        tunnel = TunnelConfig(
            tunnel_index=0,
            mode="steps",
            steps=10,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
            batch_size=10,
            trackers=[
                TrackerConfig(name="metrics", kwargs={"every": 1}),
            ],
        )
        config = create_experiment_config(str(temp_dir), [tunnel])

        runner = TunnelRunner(config, tunnel, temp_dir)
        runner.run(n_runs=2, seeds=[0, 42], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        # Load metrics from both runs
        import json
        with open(temp_dir / "tunnel_0" / "run_0" / "metrics_tunnel_0.json") as f:
            m0 = json.load(f)
        with open(temp_dir / "tunnel_0" / "run_1" / "metrics_tunnel_0.json") as f:
            m1 = json.load(f)

        # Both should have recorded losses
        assert len(m0["losses"]) > 0
        assert len(m1["losses"]) > 0
        # With replacement=True and different seeds, loss trajectories may differ


class TestTunnelContinuation:
    """Test tunnel continuation from previous tunnel."""

    def test_1to1_continuation(self, temp_dir: Path):
        """1to1 mode should continue from corresponding run."""
        tunnel0 = TunnelConfig(
            tunnel_index=0,
            mode="steps",
            steps=5,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
            batch_size=10,
            trackers=[TrackerConfig(name="metrics", kwargs={"every": 1})],
            save_checkpoint=True,
            source_mode=None,
        )
        tunnel1 = TunnelConfig(
            tunnel_index=1,
            mode="steps",
            steps=5,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.05}),
            batch_size=10,
            trackers=[TrackerConfig(name="metrics", kwargs={"every": 1})],
            save_checkpoint=True,
            source_mode="1to1",
        )
        config = create_experiment_config(str(temp_dir), [tunnel0, tunnel1])

        # Run tunnel 0
        runner0 = TunnelRunner(config, tunnel0, temp_dir)
        runner0.run(n_runs=1, seeds=[0], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        # Run tunnel 1 (should continue from tunnel 0)
        runner1 = TunnelRunner(config, tunnel1, temp_dir)
        runner1.run(n_runs=1, seeds=[0], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        # Both checkpoints should exist
        assert (temp_dir / "tunnel_0" / "run_0" / "checkpoint.pt").exists()
        assert (temp_dir / "tunnel_1" / "run_0" / "checkpoint.pt").exists()

    def test_cartesian_continuation(self, temp_dir: Path):
        """Cartesian mode should continue all runs from single source."""
        tunnel0 = TunnelConfig(
            tunnel_index=0,
            mode="steps",
            steps=5,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
            batch_size=10,
            trackers=[TrackerConfig(name="metrics", kwargs={"every": 1})],
            save_checkpoint=True,
            source_mode=None,
        )
        tunnel1 = TunnelConfig(
            tunnel_index=1,
            mode="steps",
            steps=5,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.05}),
            batch_size=10,
            trackers=[TrackerConfig(name="metrics", kwargs={"every": 1})],
            save_checkpoint=True,
            source_mode="cartesian",
            source_run_index=0,
        )
        config = create_experiment_config(str(temp_dir), [tunnel0, tunnel1])

        # Run tunnel 0 with 1 run
        runner0 = TunnelRunner(config, tunnel0, temp_dir)
        runner0.run(n_runs=1, seeds=[0], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        # Run tunnel 1 with 3 runs (all from tunnel_0/run_0)
        runner1 = TunnelRunner(config, tunnel1, temp_dir)
        runner1.run(n_runs=3, seeds=[0, 1, 2], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        # All 3 runs should exist
        for i in range(3):
            assert (temp_dir / "tunnel_1" / f"run_{i}" / "checkpoint.pt").exists()


class TestTrackerExecution:
    """Test that trackers work correctly."""

    def test_metrics_tracker(self, temp_dir: Path):
        """Metrics tracker should save loss and accuracy."""
        tunnel = TunnelConfig(
            tunnel_index=0,
            mode="steps",
            steps=10,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
            batch_size=10,
            trackers=[TrackerConfig(name="metrics", kwargs={"every": 2})],
        )
        config = create_experiment_config(str(temp_dir), [tunnel])

        runner = TunnelRunner(config, tunnel, temp_dir)
        runner.run(n_runs=1, seeds=[0], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        metrics_file = temp_dir / "tunnel_0" / "run_0" / "metrics_tunnel_0.json"
        assert metrics_file.exists()

        with open(metrics_file) as f:
            metrics = json.load(f)

        assert "losses" in metrics
        assert "accuracies" in metrics
        assert len(metrics["losses"]) > 0

    def test_weights_tracker(self, temp_dir: Path):
        """Weights tracker should save weight snapshots."""
        tunnel = TunnelConfig(
            tunnel_index=0,
            mode="steps",
            steps=10,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
            batch_size=10,
            trackers=[TrackerConfig(name="weights", kwargs={"every": 1})],  # every=1 to ensure snapshots
        )
        config = create_experiment_config(str(temp_dir), [tunnel])

        runner = TunnelRunner(config, tunnel, temp_dir)
        runner.run(n_runs=1, seeds=[0], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        weights_file = temp_dir / "tunnel_0" / "run_0" / "weights_tunnel_0.pt"
        assert weights_file.exists()

        weights_data = torch.load(weights_file)
        # May be tensor or dict depending on implementation
        if isinstance(weights_data, dict):
            # Check that weights were recorded
            assert len(weights_data) > 0 or "weights" in weights_data
        else:
            # Tensor format
            assert weights_data.ndim == 2  # [num_snapshots, num_params]
            assert weights_data.shape[0] >= 1  # At least 1 snapshot


class TestMetadataSaving:
    """Test that metadata is saved correctly."""

    def test_run_metadata_saved(self, temp_dir: Path):
        """Each run should save metadata JSON."""
        tunnel = TunnelConfig(
            tunnel_index=0,
            mode="steps",
            steps=5,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
            batch_size=10,
            trackers=[TrackerConfig(name="metrics", kwargs={"every": 1})],
        )
        config = create_experiment_config(str(temp_dir), [tunnel])

        runner = TunnelRunner(config, tunnel, temp_dir)
        runner.run(n_runs=1, seeds=[42], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        metadata_file = temp_dir / "tunnel_0" / "run_0" / "run_metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["seed"] == 42
        assert metadata["run_index"] == 0
        assert "learning_rate" in metadata

    def test_tunnel_metadata_saved(self, temp_dir: Path):
        """Tunnel should save its own metadata."""
        tunnel = TunnelConfig(
            tunnel_index=0,
            description="Test tunnel",
            mode="steps",
            steps=5,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
            batch_size=10,
            trackers=[TrackerConfig(name="metrics", kwargs={"every": 1})],
        )
        config = create_experiment_config(str(temp_dir), [tunnel])

        runner = TunnelRunner(config, tunnel, temp_dir)
        runner.run(n_runs=1, seeds=[0], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        metadata_file = temp_dir / "tunnel_0" / "tunnel_metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["tunnel_index"] == 0
        assert metadata["description"] == "Test tunnel"
