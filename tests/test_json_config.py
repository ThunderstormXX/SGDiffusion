"""Tests for JSON config loading."""
import json
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.exp5.run_tunnel import load_config_from_json


class TestConfigLoading:
    """Test JSON config loading functionality."""

    def test_load_minimal_config(self, config_file: Path):
        """Should load a minimal valid config."""
        config = load_config_from_json(config_file)

        assert config is not None
        assert config.run_dir.endswith("test_experiment")
        assert config.device == "cpu"
        assert config.dtype == "float32"

    def test_load_data_config(self, config_file: Path):
        """Should correctly load data configuration."""
        config = load_config_from_json(config_file)

        assert config.data.name == "mnist"
        assert config.data.kwargs["sample_size"] == 100
        assert config.data.kwargs["replacement"] is True

    def test_load_model_config(self, config_file: Path):
        """Should correctly load model configuration."""
        config = load_config_from_json(config_file)

        assert config.model.name == "class"
        assert config.model.kwargs["class_name"] == "FlexibleMLP"
        assert config.model.kwargs["hidden_dim"] == 4

    def test_load_tunnels(self, config_file: Path):
        """Should correctly load all tunnels."""
        config = load_config_from_json(config_file)

        assert len(config.tunnels) == 3

        # Tunnel 0
        t0 = config.tunnels[0]
        assert t0.tunnel_index == 0
        assert t0.mode == "steps"
        assert t0.steps == 10
        assert t0.optimizer.name == "sgd"
        assert t0.optimizer.kwargs["lr"] == 0.1
        assert t0.source_mode is None

        # Tunnel 1 (GD)
        t1 = config.tunnels[1]
        assert t1.mode == "epochs"
        assert t1.epochs == 10
        assert t1.dataloader_mode == "fullbatch"
        assert t1.source_mode == "1to1"

        # Tunnel 2 (Many SGD)
        t2 = config.tunnels[2]
        assert t2.mode == "steps"
        assert t2.source_mode == "cartesian"

    def test_load_trackers(self, config_file: Path):
        """Should correctly load tracker configurations."""
        config = load_config_from_json(config_file)

        # Tunnel 0 has metrics tracker
        t0_trackers = {t.name for t in config.tunnels[0].trackers}
        assert "metrics" in t0_trackers

        # Tunnel 2 has metrics and weights trackers
        t2_trackers = {t.name for t in config.tunnels[2].trackers}
        assert "metrics" in t2_trackers
        assert "weights" in t2_trackers


class TestConfigValidation:
    """Test config validation and error handling."""

    def test_missing_experiment_name(self, temp_dir: Path):
        """Should handle missing experiment_name gracefully."""
        config_data = {
            "data": {"name": "mnist"},
            "model": {"name": "class", "class_name": "FlexibleMLP"},
            "tunnels": [],
        }
        config_path = temp_dir / "bad_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(KeyError):
            load_config_from_json(config_path)

    def test_empty_tunnels_list(self, temp_dir: Path):
        """Should handle empty tunnels list."""
        config_data = {
            "experiment_name": "empty_test",
            "data": {"name": "mnist"},
            "model": {"name": "class", "class_name": "FlexibleMLP"},
            "tunnels": [],
        }
        config_path = temp_dir / "empty_tunnels.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = load_config_from_json(config_path)
        assert len(config.tunnels) == 0

    def test_config_file_not_found(self, temp_dir: Path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config_from_json(temp_dir / "nonexistent.json")


class TestConfigDefaults:
    """Test that missing fields get sensible defaults."""

    def test_default_device(self, temp_dir: Path):
        """Missing device should default to 'cpu'."""
        config_data = {
            "experiment_name": "test",
            "data": {"name": "mnist"},
            "model": {"name": "class", "class_name": "FlexibleMLP"},
            "tunnels": [],
        }
        config_path = temp_dir / "no_device.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = load_config_from_json(config_path)
        assert config.device == "cpu"

    def test_default_dtype(self, temp_dir: Path):
        """Missing dtype should default to 'float32'."""
        config_data = {
            "experiment_name": "test",
            "data": {"name": "mnist"},
            "model": {"name": "class", "class_name": "FlexibleMLP"},
            "tunnels": [],
        }
        config_path = temp_dir / "no_dtype.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = load_config_from_json(config_path)
        assert config.dtype == "float32"

    def test_default_tunnel_mode(self, temp_dir: Path):
        """Missing tunnel mode should default to 'steps'."""
        config_data = {
            "experiment_name": "test",
            "data": {"name": "mnist"},
            "model": {"name": "class", "class_name": "FlexibleMLP"},
            "tunnels": [
                {"name": "tunnel_0", "steps": 10, "optimizer": "sgd", "lr": 0.1}
            ],
        }
        config_path = temp_dir / "no_mode.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = load_config_from_json(config_path)
        assert config.tunnels[0].mode == "steps"
