"""
Registry for components (data, models, optimizers, trackers).
"""

from collections.abc import Callable


class Registry:
    """
    A simple registry for mapping names to factory functions or classes.

    Usage:
        registry = Registry()
        registry.register_data("shakespeare", shakespeare_factory)
        data_module = registry.get_data("shakespeare")(**kwargs)
    """

    def __init__(self):
        self._data: dict[str, Callable] = {}
        self._models: dict[str, Callable] = {}
        self._optimizers: dict[str, Callable] = {}
        self._trackers: dict[str, type] = {}

    # --- Data ---
    def register_data(self, name: str, factory: Callable) -> None:
        """Register a data factory by name."""
        self._data[name.lower()] = factory

    def get_data(self, name: str) -> Callable:
        """Get a data factory by name."""
        name = name.lower()
        if name not in self._data:
            raise KeyError(f"Data '{name}' not found. Available: {list(self._data.keys())}")
        return self._data[name]

    def data_decorator(self, name: str):
        """Decorator to register a data factory."""
        def decorator(factory: Callable) -> Callable:
            self.register_data(name, factory)
            return factory
        return decorator

    # --- Models ---
    def register_model(self, name: str, factory: Callable) -> None:
        """Register a model factory by name."""
        self._models[name.lower()] = factory

    def get_model(self, name: str) -> Callable:
        """Get a model factory by name."""
        name = name.lower()
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        return self._models[name]

    def model_decorator(self, name: str):
        """Decorator to register a model factory."""
        def decorator(factory: Callable) -> Callable:
            self.register_model(name, factory)
            return factory
        return decorator

    # --- Optimizers ---
    def register_optimizer(self, name: str, factory: Callable) -> None:
        """Register an optimizer factory by name."""
        self._optimizers[name.lower()] = factory

    def get_optimizer(self, name: str) -> Callable:
        """Get an optimizer factory by name."""
        name = name.lower()
        if name not in self._optimizers:
            raise KeyError(f"Optimizer '{name}' not found. Available: {list(self._optimizers.keys())}")
        return self._optimizers[name]

    def optimizer_decorator(self, name: str):
        """Decorator to register an optimizer factory."""
        def decorator(factory: Callable) -> Callable:
            self.register_optimizer(name, factory)
            return factory
        return decorator

    # --- Trackers ---
    def register_tracker(self, name: str, tracker_cls: type) -> None:
        """Register a tracker class by name."""
        self._trackers[name.lower()] = tracker_cls

    def get_tracker(self, name: str) -> type:
        """Get a tracker class by name."""
        name = name.lower()
        if name not in self._trackers:
            raise KeyError(f"Tracker '{name}' not found. Available: {list(self._trackers.keys())}")
        return self._trackers[name]

    def tracker_decorator(self, name: str):
        """Decorator to register a tracker class."""
        def decorator(tracker_cls: type) -> type:
            self.register_tracker(name, tracker_cls)
            return tracker_cls
        return decorator

    # --- Utility ---
    def list_all(self) -> dict[str, list]:
        """List all registered components."""
        return {
            "data": list(self._data.keys()),
            "models": list(self._models.keys()),
            "optimizers": list(self._optimizers.keys()),
            "trackers": list(self._trackers.keys()),
        }


# Global registry instance
registry = Registry()
