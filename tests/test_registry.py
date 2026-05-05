"""Tests for the component registry."""
import pytest

from src.datamodelopt.core.registry import Registry, registry


class TestRegistrySingleton:
    """Test that registry is a proper singleton-like global instance."""

    def test_registry_exists(self):
        """Global registry should exist."""
        assert registry is not None
        assert isinstance(registry, Registry)

    def test_registry_has_methods(self):
        """Registry should have registration and retrieval methods."""
        assert hasattr(registry, "register_model")
        assert hasattr(registry, "register_data")
        assert hasattr(registry, "register_tracker")
        assert hasattr(registry, "get_model")
        assert hasattr(registry, "get_data")
        assert hasattr(registry, "get_tracker")


class TestModelRegistry:
    """Test model registration and retrieval."""

    def test_class_model_registered(self):
        """ClassFactory should be registered as 'class'."""
        factory = registry.get_model("class")
        assert factory is not None
        assert callable(factory)

    def test_nanogpt_model_registered(self):
        """NanoGPTFactory should be registered as 'nanogpt'."""
        factory = registry.get_model("nanogpt")
        assert factory is not None
        assert callable(factory)

    def test_model_not_found_raises(self):
        """Getting non-existent model should raise KeyError."""
        with pytest.raises(KeyError) as exc_info:
            registry.get_model("nonexistent_model")
        assert "nonexistent_model" in str(exc_info.value)

    def test_model_case_insensitive(self):
        """Model names should be case-insensitive."""
        factory1 = registry.get_model("class")
        factory2 = registry.get_model("CLASS")
        assert factory1 == factory2


class TestDataRegistry:
    """Test data module registration and retrieval."""

    def test_mnist_data_registered(self):
        """MNISTDataModule should be registered as 'mnist'."""
        factory = registry.get_data("mnist")
        assert factory is not None
        assert callable(factory)

    def test_data_not_found_raises(self):
        """Getting non-existent data should raise KeyError."""
        with pytest.raises(KeyError) as exc_info:
            registry.get_data("nonexistent_data")
        assert "nonexistent_data" in str(exc_info.value)


class TestTrackerRegistry:
    """Test tracker registration and retrieval."""

    def test_metrics_tracker_registered(self):
        """MetricsTracker should be registered as 'metrics'."""
        tracker_cls = registry.get_tracker("metrics")
        assert tracker_cls is not None

    def test_weights_tracker_registered(self):
        """WeightsTracker should be registered as 'weights'."""
        tracker_cls = registry.get_tracker("weights")
        assert tracker_cls is not None

    def test_hessian_tracker_registered(self):
        """HessianTracker should be registered as 'hessian'."""
        tracker_cls = registry.get_tracker("hessian")
        assert tracker_cls is not None

    def test_tracker_not_found_raises(self):
        """Getting non-existent tracker should raise KeyError."""
        with pytest.raises(KeyError) as exc_info:
            registry.get_tracker("nonexistent_tracker")
        assert "nonexistent_tracker" in str(exc_info.value)


class TestCustomRegistration:
    """Test that custom components can be registered."""

    def test_register_custom_model(self):
        """Should be able to register a custom model factory."""
        def dummy_factory(**kwargs):
            return None

        # Use decorator style
        registry.register_model("test_dummy_model", dummy_factory)
        retrieved = registry.get_model("test_dummy_model")
        assert retrieved == dummy_factory

    def test_register_custom_tracker(self):
        """Should be able to register a custom tracker."""
        class DummyTracker:
            pass

        registry.register_tracker("test_dummy_tracker", DummyTracker)
        retrieved = registry.get_tracker("test_dummy_tracker")
        assert retrieved == DummyTracker
