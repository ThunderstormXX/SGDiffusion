"""Tests for OOP quality of the codebase."""
import inspect
from dataclasses import is_dataclass
from typing import get_type_hints

from src.datamodelopt.core.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrackerConfig,
    TunnelConfig,
)
from src.datamodelopt.data.base import DataModule
from src.datamodelopt.models.factories import ClassFactory, ModelFactory
from src.datamodelopt.tracking.base import Tracker


class TestEncapsulation:
    """Test that classes properly encapsulate their data."""

    def test_config_classes_use_dataclass(self):
        """Config classes should use dataclass for clean structure."""
        configs = [
            DataConfig, ModelConfig, OptimizerConfig,
            TrackerConfig, TunnelConfig, ExperimentConfig
        ]
        for cls in configs:
            assert is_dataclass(cls), f"{cls.__name__} should be a dataclass"

    def test_tunnel_config_has_getter_method(self):
        """TunnelConfig should expose lr through getter, not direct access."""
        config = TunnelConfig(
            tunnel_index=0,
            steps=10,
            optimizer=OptimizerConfig(name="sgd", kwargs={"lr": 0.1}),
        )
        # Should use method, not direct nested access
        assert hasattr(config, "get_learning_rate")
        assert config.get_learning_rate() == 0.1

    def test_tunnel_config_has_index_field(self):
        """TunnelConfig should have tunnel_index field."""
        config = TunnelConfig(tunnel_index=2, steps=10)
        assert hasattr(config, "tunnel_index")
        assert config.tunnel_index == 2


class TestAbstraction:
    """Test that classes use proper abstraction."""

    def test_data_module_is_abstract_base(self):
        """DataModule should be an abstract base class."""
        assert inspect.isclass(DataModule)
        # Should have abstract methods
        assert hasattr(DataModule, "setup")
        assert hasattr(DataModule, "train_loader")

    def test_model_factory_is_abstract(self):
        """ModelFactory should be an abstract factory."""
        assert inspect.isclass(ModelFactory)
        assert hasattr(ModelFactory, "build")

    def test_tracker_is_base_class(self):
        """Tracker should be a base class with interface."""
        assert inspect.isclass(Tracker)
        # Should define interface methods
        assert hasattr(Tracker, "on_step_end")
        assert hasattr(Tracker, "flush")

    def test_class_factory_extends_model_factory(self):
        """ClassFactory should extend ModelFactory."""
        assert issubclass(ClassFactory, ModelFactory)


class TestSingleResponsibility:
    """Test that classes have single responsibilities."""

    def test_config_classes_only_hold_data(self):
        """Config dataclasses should primarily hold data, not logic."""
        # Check that they don't have many complex methods
        for cls in [DataConfig, ModelConfig, OptimizerConfig, TrackerConfig]:
            methods = [m for m in dir(cls) if not m.startswith("_") and callable(getattr(cls, m))]
            # Dataclass methods + a few helpers at most
            assert len(methods) <= 5, f"{cls.__name__} has too many methods: {methods}"

    def test_tunnel_config_helper_methods_limited(self):
        """TunnelConfig may have helper methods but should be limited."""
        methods = [
            m for m in dir(TunnelConfig)
            if not m.startswith("_") and callable(getattr(TunnelConfig, m))
        ]
        # Should have limited public methods
        assert len(methods) <= 10, f"TunnelConfig has too many methods: {methods}"


class TestTypeAnnotations:
    """Test that classes use proper type annotations."""

    def test_config_classes_are_typed(self):
        """Config classes should have type annotations."""
        for cls in [DataConfig, ModelConfig, TunnelConfig, ExperimentConfig]:
            hints = get_type_hints(cls)
            assert len(hints) > 0, f"{cls.__name__} should have type hints"

    def test_tunnel_config_typed_fields(self):
        """TunnelConfig should have typed essential fields."""
        hints = get_type_hints(TunnelConfig)
        assert "tunnel_index" in hints
        assert "mode" in hints
        assert "optimizer" in hints
        assert "trackers" in hints

    def test_experiment_config_typed_fields(self):
        """ExperimentConfig should have typed essential fields."""
        hints = get_type_hints(ExperimentConfig)
        assert "run_dir" in hints
        assert "data" in hints
        assert "model" in hints


class TestComposition:
    """Test that classes use composition properly."""

    def test_experiment_config_composes_configs(self):
        """ExperimentConfig should compose other configs."""
        hints = get_type_hints(ExperimentConfig)
        assert hints.get("data") == DataConfig
        assert hints.get("model") == ModelConfig

    def test_tunnel_config_composes_optimizer(self):
        """TunnelConfig should compose OptimizerConfig."""
        hints = get_type_hints(TunnelConfig)
        assert hints.get("optimizer") == OptimizerConfig


class TestInterfaceConsistency:
    """Test that similar classes have consistent interfaces."""

    def test_all_configs_have_name_or_index(self):
        """Data/Model configs should have 'name', Tunnel should have 'tunnel_index'."""
        assert "name" in get_type_hints(DataConfig)
        assert "name" in get_type_hints(ModelConfig)
        assert "tunnel_index" in get_type_hints(TunnelConfig)

    def test_all_configs_serializable(self):
        """All config classes should be JSON-serializable conceptually."""
        # Dataclasses can be converted to dict easily
        data = DataConfig(name="test", kwargs={"a": 1})
        model = ModelConfig(name="test", kwargs={"b": 2})

        # Should be able to access as dict-like
        from dataclasses import asdict
        data_dict = asdict(data)
        model_dict = asdict(model)

        assert "name" in data_dict
        assert "name" in model_dict


class TestMethodSignatures:
    """Test that method signatures are clean and consistent."""

    def test_tracker_on_step_end_signature(self):
        """Tracker.on_step_end should have consistent signature."""
        sig = inspect.signature(Tracker.on_step_end)
        params = list(sig.parameters.keys())
        # Should accept self and context
        assert "self" in params
        assert "ctx" in params or len(params) >= 2

    def test_tracker_flush_signature(self):
        """Tracker.flush should accept output path."""
        sig = inspect.signature(Tracker.flush)
        params = list(sig.parameters.keys())
        assert len(params) >= 2  # self + save_dir


class TestNamingConventions:
    """Test that naming conventions are followed."""

    def test_class_names_are_pascal_case(self):
        """Class names should be PascalCase."""
        classes = [
            DataConfig, ModelConfig, OptimizerConfig,
            TrackerConfig, TunnelConfig, ExperimentConfig,
            DataModule, ModelFactory, ClassFactory, Tracker,
        ]
        for cls in classes:
            name = cls.__name__
            # Should start with uppercase
            assert name[0].isupper(), f"{name} should start with uppercase"
            # Should not have underscores
            assert "_" not in name, f"{name} should be PascalCase, not snake_case"

    def test_method_names_are_snake_case(self):
        """Public method names should be snake_case."""
        classes = [TunnelConfig, ExperimentConfig, DataModule, Tracker]
        for cls in classes:
            methods = [m for m in dir(cls) if not m.startswith("_") and callable(getattr(cls, m, None))]
            for method in methods:
                # Check it's snake_case or single word
                assert method.islower() or "_" in method, \
                    f"{cls.__name__}.{method} should be snake_case"
