"""Tests for configuration dataclasses."""
from dataclasses import fields, is_dataclass

import pytest

from src.datamodelopt.core.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrackerConfig,
    TunnelConfig,
)


class TestDataclassStructure:
    """Test that config classes are properly structured dataclasses."""

    @pytest.mark.parametrize("cls", [
        DataConfig,
        ModelConfig,
        OptimizerConfig,
        TrackerConfig,
        TunnelConfig,
        ExperimentConfig,
    ])
    def test_is_dataclass(self, cls):
        """All config classes should be dataclasses."""
        assert is_dataclass(cls), f"{cls.__name__} should be a dataclass"

    def test_data_config_required_fields(self):
        """DataConfig should have 'name' as required field."""
        field_names = [f.name for f in fields(DataConfig)]
        assert "name" in field_names
        assert "kwargs" in field_names

    def test_model_config_required_fields(self):
        """ModelConfig should have 'name' as required field."""
        field_names = [f.name for f in fields(ModelConfig)]
        assert "name" in field_names
        assert "kwargs" in field_names

    def test_tunnel_config_required_fields(self):
        """TunnelConfig should have essential fields."""
        field_names = [f.name for f in fields(TunnelConfig)]
        required = ["tunnel_index", "mode", "optimizer", "trackers", "source_mode"]
        for field in required:
            assert field in field_names, f"TunnelConfig missing '{field}'"

    def test_experiment_config_required_fields(self):
        """ExperimentConfig should have essential fields."""
        field_names = [f.name for f in fields(ExperimentConfig)]
        required = ["run_dir", "data", "model"]
        for field in required:
            assert field in field_names, f"ExperimentConfig missing '{field}'"


class TestConfigDefaults:
    """Test that configs have sensible defaults."""

    def test_data_config_defaults(self):
        """DataConfig should have empty kwargs by default."""
        config = DataConfig(name="test")
        assert config.kwargs == {}

    def test_model_config_defaults(self):
        """ModelConfig should have empty kwargs by default."""
        config = ModelConfig(name="test")
        assert config.kwargs == {}

    def test_optimizer_config_defaults(self):
        """OptimizerConfig should have SGD as default."""
        config = OptimizerConfig(name="sgd")
        assert config.name == "sgd"
        assert config.kwargs == {}

    def test_tracker_config_defaults(self):
        """TrackerConfig should have metrics as valid option."""
        config = TrackerConfig(name="metrics", kwargs={"every": 1})
        assert config.name == "metrics"
        assert config.kwargs["every"] == 1

    def test_tunnel_config_defaults(self):
        """TunnelConfig should have sensible defaults."""
        config = TunnelConfig(tunnel_index=0, steps=10)  # steps required for mode='steps'
        assert config.mode == "steps"
        assert config.dataloader_mode == "minibatch"
        assert config.save_checkpoint is True
        assert config.source_mode is None


class TestTunnelConfigMethods:
    """Test TunnelConfig methods."""

    def test_get_learning_rate(self):
        """get_learning_rate should extract lr from optimizer kwargs."""
        optimizer = OptimizerConfig(name="sgd", kwargs={"lr": 0.01})
        config = TunnelConfig(tunnel_index=0, steps=10, optimizer=optimizer)
        assert config.get_learning_rate() == 0.01

    def test_get_learning_rate_default(self):
        """get_learning_rate should return default (0.0) if not set."""
        optimizer = OptimizerConfig(name="sgd", kwargs={})
        config = TunnelConfig(tunnel_index=0, steps=10, optimizer=optimizer)
        # Returns 0.0 as default when lr not specified
        assert config.get_learning_rate() == 0.0

    def test_tunnel_index_accessible(self):
        """Tunnel index should be directly accessible."""
        config = TunnelConfig(tunnel_index=2, steps=10)
        assert config.tunnel_index == 2


class TestExperimentConfigMethods:
    """Test ExperimentConfig methods."""

    def test_is_tunnel_based_with_tunnels(self):
        """is_tunnel_based should return True when tunnels exist."""
        tunnel = TunnelConfig(tunnel_index=0, steps=10)
        config = ExperimentConfig(
            run_dir="test",
            data=DataConfig(name="test"),
            model=ModelConfig(name="test"),
            tunnels=[tunnel],
        )
        assert config.is_tunnel_based is True

    def test_is_tunnel_based_without_tunnels(self):
        """is_tunnel_based should return False when no tunnels."""
        config = ExperimentConfig(
            run_dir="test",
            data=DataConfig(name="test"),
            model=ModelConfig(name="test"),
        )
        assert config.is_tunnel_based is False

    def test_num_tunnels(self):
        """num_tunnels should return correct count."""
        tunnels = [TunnelConfig(tunnel_index=i, steps=10) for i in range(3)]
        config = ExperimentConfig(
            run_dir="test",
            data=DataConfig(name="test"),
            model=ModelConfig(name="test"),
            tunnels=tunnels,
        )
        assert config.num_tunnels == 3


class TestConfigImmutability:
    """Test that configs can be used safely."""

    def test_optimizer_kwargs_not_shared(self):
        """Two OptimizerConfigs should not share kwargs dict."""
        c1 = OptimizerConfig(name="sgd")
        c2 = OptimizerConfig(name="sgd")
        c1.kwargs["lr"] = 0.1
        assert "lr" not in c2.kwargs

    def test_data_config_kwargs_not_shared(self):
        """Two DataConfigs should not share kwargs dict."""
        c1 = DataConfig(name="test")
        c2 = DataConfig(name="test")
        c1.kwargs["batch_size"] = 64
        assert "batch_size" not in c2.kwargs
