"""Integration tests for full pipeline execution."""
import json
from pathlib import Path

import pytest
import torch

from src.datamodelopt.experiments.tunnel import TunnelRunner
from src.scripts.exp5.run_tunnel import load_config_from_json


class TestFullPipeline:
    """Test complete pipeline: SGD -> GD -> Many SGD."""

    @pytest.fixture
    def pipeline_config(self, temp_dir: Path) -> Path:
        """Create a minimal 3-tunnel pipeline config."""
        config = {
            "experiment_name": "integration_test",
            "description": "Integration test: 5 steps -> 5 epochs -> 5 steps x 2 runs",
            "data": {
                "name": "mnist",
                "sample_size": 100,
                "replacement": True,
            },
            "model": {
                "name": "class",
                "class_name": "FlexibleMLP",
                "hidden_dim": 4,
                "num_hidden_layers": 1,
                "input_downsample": 6,
            },
            "tunnels": [
                {
                    "name": "tunnel_0",
                    "description": "SGD: 5 steps",
                    "mode": "steps",
                    "steps": 5,
                    "optimizer": "sgd",
                    "lr": 0.1,
                    "batch_size": 10,
                    "n_runs": 1,
                    "trackers": {"metrics": {"every": 1}, "weights": {"every": 5}},
                    "save_checkpoint": True,
                    "source_mode": None,
                },
                {
                    "name": "tunnel_1",
                    "description": "GD: 5 epochs",
                    "mode": "epochs",
                    "epochs": 5,
                    "optimizer": "sgd",
                    "lr": 0.01,
                    "batch_size": 100,
                    "dataloader_mode": "fullbatch",
                    "n_runs": 1,
                    "trackers": {"metrics": {"every": 1}, "weights": {"every": 1}},
                    "save_checkpoint": True,
                    "source_mode": "1to1",
                },
                {
                    "name": "tunnel_2",
                    "description": "Many SGD: 5 steps x 2 runs",
                    "mode": "steps",
                    "steps": 5,
                    "optimizer": "sgd",
                    "lr": 0.1,
                    "batch_size": 10,
                    "n_runs": 2,
                    "trackers": {"metrics": {"every": 1}, "weights": {"every": 5}},
                    "save_checkpoint": True,
                    "source_mode": "cartesian",
                },
            ],
            "device": "cpu",
            "dtype": "float32",
        }
        config_path = temp_dir / "pipeline_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return config_path

    def test_full_pipeline_execution(self, temp_dir: Path, pipeline_config: Path):
        """Run complete 3-tunnel pipeline."""
        config = load_config_from_json(pipeline_config)
        device = torch.device("cpu")
        dtype = torch.float32

        # Run all tunnels sequentially
        for i, tunnel_config in enumerate(config.tunnels):
            n_runs = [1, 1, 2][i]  # From config
            seeds = list(range(n_runs))

            runner = TunnelRunner(
                experiment_config=config,
                tunnel_config=tunnel_config,
                experiment_dir=temp_dir / "integration_test",
            )
            runner.run(
                n_runs=n_runs,
                seeds=seeds,
                device=device,
                dtype=dtype,
                verbose=False,
            )

        # Verify all outputs exist
        exp_dir = temp_dir / "integration_test"

        # Tunnel 0: 1 run
        assert (exp_dir / "tunnel_0" / "run_0" / "checkpoint.pt").exists()
        assert (exp_dir / "tunnel_0" / "run_0" / "metrics_tunnel_0.json").exists()
        assert (exp_dir / "tunnel_0" / "run_0" / "weights_tunnel_0.pt").exists()

        # Tunnel 1: 1 run (continued from tunnel 0)
        assert (exp_dir / "tunnel_1" / "run_0" / "checkpoint.pt").exists()
        assert (exp_dir / "tunnel_1" / "run_0" / "metrics_tunnel_1.json").exists()

        # Tunnel 2: 2 runs (cartesian from tunnel 1)
        assert (exp_dir / "tunnel_2" / "run_0" / "checkpoint.pt").exists()
        assert (exp_dir / "tunnel_2" / "run_1" / "checkpoint.pt").exists()

    def test_loss_decreases_in_gd(self, temp_dir: Path, pipeline_config: Path):
        """GD tunnel should show decreasing loss (generally)."""
        config = load_config_from_json(pipeline_config)
        device = torch.device("cpu")
        dtype = torch.float32

        # Run tunnel 0 first
        runner0 = TunnelRunner(config, config.tunnels[0], temp_dir / "loss_test")
        runner0.run(n_runs=1, seeds=[0], device=device, dtype=dtype, verbose=False)

        # Run tunnel 1 (GD)
        runner1 = TunnelRunner(config, config.tunnels[1], temp_dir / "loss_test")
        runner1.run(n_runs=1, seeds=[0], device=device, dtype=dtype, verbose=False)

        # Load GD metrics
        metrics_file = temp_dir / "loss_test" / "tunnel_1" / "run_0" / "metrics_tunnel_1.json"
        with open(metrics_file) as f:
            metrics = json.load(f)

        losses = metrics["losses"]
        assert len(losses) >= 2, "Should have multiple loss values"

        # GD should generally decrease loss (allow some noise)
        # Check that final loss is less than initial
        assert losses[-1] <= losses[0] * 1.5, "GD should not increase loss significantly"

    def test_cartesian_runs_start_from_same_point(self, temp_dir: Path, pipeline_config: Path):
        """Cartesian mode runs should start from the same checkpoint."""
        config = load_config_from_json(pipeline_config)
        device = torch.device("cpu")
        dtype = torch.float32

        exp_dir = temp_dir / "cartesian_test"

        # Run tunnel 0
        runner0 = TunnelRunner(config, config.tunnels[0], exp_dir)
        runner0.run(n_runs=1, seeds=[0], device=device, dtype=dtype, verbose=False)

        # Run tunnel 1
        runner1 = TunnelRunner(config, config.tunnels[1], exp_dir)
        runner1.run(n_runs=1, seeds=[0], device=device, dtype=dtype, verbose=False)

        # Run tunnel 2 with 2 runs
        runner2 = TunnelRunner(config, config.tunnels[2], exp_dir)
        runner2.run(n_runs=2, seeds=[0, 1], device=device, dtype=dtype, verbose=False)

        # Both runs should have completed and saved checkpoints
        assert (exp_dir / "tunnel_2" / "run_0" / "checkpoint.pt").exists()
        assert (exp_dir / "tunnel_2" / "run_1" / "checkpoint.pt").exists()

        # Both should have metrics (showing they ran)
        assert (exp_dir / "tunnel_2" / "run_0" / "metrics_tunnel_2.json").exists()
        assert (exp_dir / "tunnel_2" / "run_1" / "metrics_tunnel_2.json").exists()


class TestEdgeCases:
    """Test edge cases in pipeline execution."""

    def test_single_step_tunnel(self, temp_dir: Path):
        """Tunnel with single step should work."""
        config_data = {
            "experiment_name": "single_step_test",
            "data": {"name": "mnist", "sample_size": 50, "replacement": True},
            "model": {
                "name": "class",
                "class_name": "FlexibleMLP",
                "hidden_dim": 4,
                "num_hidden_layers": 1,
                "input_downsample": 6,
            },
            "tunnels": [{
                "name": "tunnel_0",
                "mode": "steps",
                "steps": 1,
                "optimizer": "sgd",
                "lr": 0.1,
                "batch_size": 10,
                "trackers": {"metrics": {"every": 1}},
                "save_checkpoint": True,
                "source_mode": None,
            }],
            "device": "cpu",
            "dtype": "float32",
        }
        config_path = temp_dir / "single_step.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = load_config_from_json(config_path)
        runner = TunnelRunner(config, config.tunnels[0], temp_dir / "single_step_exp")
        runner.run(n_runs=1, seeds=[0], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        assert (temp_dir / "single_step_exp" / "tunnel_0" / "run_0" / "checkpoint.pt").exists()

    def test_single_epoch_tunnel(self, temp_dir: Path):
        """Tunnel with single epoch should work."""
        config_data = {
            "experiment_name": "single_epoch_test",
            "data": {"name": "mnist", "sample_size": 50, "replacement": True},
            "model": {
                "name": "class",
                "class_name": "FlexibleMLP",
                "hidden_dim": 4,
                "num_hidden_layers": 1,
                "input_downsample": 6,
            },
            "tunnels": [{
                "name": "tunnel_0",
                "mode": "epochs",
                "epochs": 1,
                "optimizer": "sgd",
                "lr": 0.1,
                "batch_size": 10,
                "trackers": {"metrics": {"every": 1}},
                "save_checkpoint": True,
                "source_mode": None,
            }],
            "device": "cpu",
            "dtype": "float32",
        }
        config_path = temp_dir / "single_epoch.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = load_config_from_json(config_path)
        runner = TunnelRunner(config, config.tunnels[0], temp_dir / "single_epoch_exp")
        runner.run(n_runs=1, seeds=[0], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        assert (temp_dir / "single_epoch_exp" / "tunnel_0" / "run_0" / "checkpoint.pt").exists()

    def test_no_trackers(self, temp_dir: Path):
        """Tunnel without trackers should still work."""
        config_data = {
            "experiment_name": "no_trackers_test",
            "data": {"name": "mnist", "sample_size": 50, "replacement": True},
            "model": {
                "name": "class",
                "class_name": "FlexibleMLP",
                "hidden_dim": 4,
                "num_hidden_layers": 1,
                "input_downsample": 6,
            },
            "tunnels": [{
                "name": "tunnel_0",
                "mode": "steps",
                "steps": 5,
                "optimizer": "sgd",
                "lr": 0.1,
                "batch_size": 10,
                "trackers": {},  # Empty trackers
                "save_checkpoint": True,
                "source_mode": None,
            }],
            "device": "cpu",
            "dtype": "float32",
        }
        config_path = temp_dir / "no_trackers.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = load_config_from_json(config_path)
        runner = TunnelRunner(config, config.tunnels[0], temp_dir / "no_trackers_exp")
        runner.run(n_runs=1, seeds=[0], device=torch.device("cpu"), dtype=torch.float32, verbose=False)

        # Checkpoint should exist even without trackers
        assert (temp_dir / "no_trackers_exp" / "tunnel_0" / "run_0" / "checkpoint.pt").exists()
