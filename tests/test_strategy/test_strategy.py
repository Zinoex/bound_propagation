"""
Tests for bounding strategy framework.

DEPRECATED: Old strategy architecture replaced with method-based propagators.
See tests/test_linear_relaxation.py and tests/test_relaxation_strategies.py for new tests.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Old strategy architecture deprecated")

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.ir import OperationType
from bound_propagation.regions import HyperRectangle

# from bound_propagation.strategy import (
#     BoundingStrategy,
#     StrategyConfig,
#     StrategyRegistry,
#     get_global_registry,
#     get_strategy,
#     register_fallback,
#     register_strategy,
# )

# Stub classes to prevent import errors (file is skipped anyway)
class BoundingStrategy:
    pass

class StrategyConfig:
    pass

class StrategyRegistry:
    pass


class MockBoundingStrategy(BoundingStrategy):
    """Mock strategy for testing."""

    def __init__(self, method: str):
        self._method = method

    def compute_bounds(self, node, input_bounds, config):
        """Mock compute bounds - just returns input bounds."""
        if not input_bounds:
            # No inputs - return dummy bounds
            region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
            return IntervalBounds(region, torch.tensor([0.0]), torch.tensor([1.0]))
        return input_bounds[0]

    @property
    def method_name(self) -> str:
        return self._method


class TestStrategyConfig:
    """Tests for StrategyConfig."""

    def test_create_config(self):
        """Test creating a config."""
        config = StrategyConfig()
        assert config.same_slope is True
        assert config.custom_params == {}

    def test_config_with_params(self):
        """Test creating config with parameters."""
        config = StrategyConfig(
            same_slope=False,
            custom_params={"param1": 123, "param2": "value"},
        )

        assert config.same_slope is False
        assert config.get("param1") == 123
        assert config.get("param2") == "value"

    def test_get_and_set(self):
        """Test getting and setting custom parameters."""
        config = StrategyConfig()

        # Get with default
        assert config.get("missing", "default") == "default"

        # Set and get
        config.set("key", "value")
        assert config.get("key") == "value"


class TestBoundingStrategy:
    """Tests for BoundingStrategy interface."""

    def test_strategy_interface(self):
        """Test that strategy implements required interface."""
        strategy = MockBoundingStrategy("test")

        assert strategy.method_name == "test"
        assert hasattr(strategy, "compute_bounds")

    def test_strategy_repr(self):
        """Test strategy string representation."""
        strategy = MockBoundingStrategy("ibp")
        repr_str = repr(strategy)

        assert "MockBoundingStrategy" in repr_str
        assert "ibp" in repr_str

    def test_compute_bounds_signature(self):
        """Test compute_bounds is callable with correct signature."""
        strategy = MockBoundingStrategy("test")
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        bounds = IntervalBounds(region, torch.tensor([0.0]), torch.tensor([1.0]))
        config = StrategyConfig()

        # Mock node (we don't need a real one for this test)
        node = None

        result = strategy.compute_bounds(node, [bounds], config)
        assert isinstance(result, IntervalBounds)


class TestStrategyRegistry:
    """Tests for StrategyRegistry."""

    def test_create_registry(self):
        """Test creating a registry."""
        registry = StrategyRegistry()
        assert len(registry) == 0

    def test_register_strategy(self):
        """Test registering a strategy."""
        registry = StrategyRegistry()
        strategy = MockBoundingStrategy("ibp")

        registry.register(OperationType.ADD, "ibp", strategy)

        assert len(registry) == 1
        assert registry.has_strategy(OperationType.ADD, "ibp")

    def test_register_duplicate_raises(self):
        """Test that registering duplicate raises error."""
        registry = StrategyRegistry()
        strategy1 = MockBoundingStrategy("ibp")
        strategy2 = MockBoundingStrategy("ibp")

        registry.register(OperationType.ADD, "ibp", strategy1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(OperationType.ADD, "ibp", strategy2)

    def test_get_strategy(self):
        """Test getting a registered strategy."""
        registry = StrategyRegistry()
        strategy = MockBoundingStrategy("ibp")

        registry.register(OperationType.ADD, "ibp", strategy)

        retrieved = registry.get(OperationType.ADD, "ibp")
        assert retrieved is strategy

    def test_get_missing_strategy(self):
        """Test getting a missing strategy returns None."""
        registry = StrategyRegistry()

        result = registry.get(OperationType.ADD, "ibp")
        assert result is None

    def test_register_fallback(self):
        """Test registering a fallback strategy."""
        registry = StrategyRegistry()
        fallback = MockBoundingStrategy("ibp")

        registry.register_fallback("ibp", fallback)

        # Should work for any operation
        assert registry.get(OperationType.ADD, "ibp") is fallback
        assert registry.get(OperationType.MUL, "ibp") is fallback
        assert registry.has_strategy(OperationType.RELU, "ibp")

    def test_specific_overrides_fallback(self):
        """Test that specific strategy overrides fallback."""
        registry = StrategyRegistry()
        fallback = MockBoundingStrategy("ibp_fallback")
        specific = MockBoundingStrategy("ibp_specific")

        registry.register_fallback("ibp", fallback)
        registry.register(OperationType.ADD, "ibp", specific)

        # ADD should get specific strategy
        assert registry.get(OperationType.ADD, "ibp") is specific

        # Other ops should get fallback
        assert registry.get(OperationType.MUL, "ibp") is fallback

    def test_get_supported_methods(self):
        """Test getting supported methods for an operation."""
        registry = StrategyRegistry()

        registry.register(OperationType.RELU, "ibp", MockBoundingStrategy("ibp"))
        registry.register(OperationType.RELU, "forward", MockBoundingStrategy("forward"))
        registry.register_fallback("backward", MockBoundingStrategy("backward"))

        methods = registry.get_supported_methods(OperationType.RELU)

        assert "ibp" in methods
        assert "forward" in methods
        assert "backward" in methods  # From fallback

    def test_get_registered_operations(self):
        """Test getting registered operations for a method."""
        registry = StrategyRegistry()

        registry.register(OperationType.ADD, "ibp", MockBoundingStrategy("ibp"))
        registry.register(OperationType.MUL, "ibp", MockBoundingStrategy("ibp"))
        registry.register(OperationType.RELU, "forward", MockBoundingStrategy("forward"))

        ops = registry.get_registered_operations("ibp")

        assert OperationType.ADD in ops
        assert OperationType.MUL in ops
        assert OperationType.RELU not in ops

    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = StrategyRegistry()

        registry.register(OperationType.ADD, "ibp", MockBoundingStrategy("ibp"))
        registry.register_fallback("forward", MockBoundingStrategy("forward"))

        assert len(registry) == 2

        registry.clear()

        assert len(registry) == 0
        assert not registry.has_strategy(OperationType.ADD, "ibp")

    def test_registry_repr(self):
        """Test registry string representation."""
        registry = StrategyRegistry()

        registry.register(OperationType.ADD, "ibp", MockBoundingStrategy("ibp"))
        registry.register_fallback("forward", MockBoundingStrategy("forward"))

        repr_str = repr(registry)
        assert "strategies=1" in repr_str
        assert "fallbacks=1" in repr_str


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_global_registry(self):
        """Test getting global registry."""
        registry = get_global_registry()
        assert isinstance(registry, StrategyRegistry)

        # Should return the same instance
        registry2 = get_global_registry()
        assert registry is registry2

    def test_register_strategy_global(self):
        """Test registering through global function."""
        # Clear first
        get_global_registry().clear()

        strategy = MockBoundingStrategy("ibp")
        register_strategy(OperationType.ADD, "ibp", strategy)

        retrieved = get_strategy(OperationType.ADD, "ibp")
        assert retrieved is strategy

    def test_register_fallback_global(self):
        """Test registering fallback through global function."""
        # Clear first
        get_global_registry().clear()

        fallback = MockBoundingStrategy("ibp")
        register_fallback("ibp", fallback)

        retrieved = get_strategy(OperationType.MUL, "ibp")
        assert retrieved is fallback
