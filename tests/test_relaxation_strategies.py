"""
Tests for RelaxationStrategy and RelaxationRegistry.
"""

import pytest
import torch

from bound_propagation.bounds.interval_bounds import IntervalBounds
from bound_propagation.ir.metadata import TensorMetadata
from bound_propagation.ir.node import Node, NodeType
from bound_propagation.ir.operations import OperationType
from bound_propagation.relaxations import (
    LinearRelaxation,
    RelaxationRegistry,
    RelaxationStrategy,
    register_relaxation_strategy,
)


@pytest.fixture(autouse=True)
def preserve_registry():
    """Save and restore the registry state around each test."""
    original_registry = RelaxationRegistry._registry.copy()
    yield
    RelaxationRegistry._registry = original_registry


class MockRelaxationStrategy(RelaxationStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, op_type: OperationType):
        self._op_type = op_type
    
    @property
    def supported_op_type(self) -> OperationType:
        return self._op_type
    
    def relax(self, node: Node, interval_inputs: list[IntervalBounds]) -> LinearRelaxation:
        # Return a simple identity relaxation for testing
        shape = interval_inputs[0].lower.shape
        device = interval_inputs[0].lower.device
        dtype = interval_inputs[0].lower.dtype
        return LinearRelaxation.create_identity(shape, device, dtype)


class TestRelaxationRegistry:
    """Test the RelaxationRegistry."""
    
    def setup_method(self):
        """Clear registry before each test."""
        RelaxationRegistry.clear()
    
    def teardown_method(self):
        """Clear registry after each test."""
        RelaxationRegistry.clear()
    
    def test_register_and_get_strategy(self):
        """Test registering and retrieving a strategy."""
        strategy = MockRelaxationStrategy(OperationType.RELU)
        RelaxationRegistry.register(OperationType.RELU, strategy)
        
        retrieved = RelaxationRegistry.get(OperationType.RELU)
        assert retrieved is strategy
    
    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate strategies raises an error."""
        strategy1 = MockRelaxationStrategy(OperationType.RELU)
        strategy2 = MockRelaxationStrategy(OperationType.RELU)
        
        RelaxationRegistry.register(OperationType.RELU, strategy1)
        
        with pytest.raises(ValueError, match="already registered"):
            RelaxationRegistry.register(OperationType.RELU, strategy2)
    
    def test_get_unregistered_returns_none(self):
        """Test that getting an unregistered operation returns None."""
        result = RelaxationRegistry.get(OperationType.SIGMOID)
        assert result is None
    
    def test_has_strategy(self):
        """Test checking if a strategy is registered."""
        strategy = MockRelaxationStrategy(OperationType.RELU)
        RelaxationRegistry.register(OperationType.RELU, strategy)
        
        assert RelaxationRegistry.has_strategy(OperationType.RELU)
        assert not RelaxationRegistry.has_strategy(OperationType.SIGMOID)
    
    def test_list_registered_ops(self):
        """Test listing all registered operation types."""
        strategy1 = MockRelaxationStrategy(OperationType.RELU)
        strategy2 = MockRelaxationStrategy(OperationType.SIGMOID)
        
        RelaxationRegistry.register(OperationType.RELU, strategy1)
        RelaxationRegistry.register(OperationType.SIGMOID, strategy2)
        
        registered = RelaxationRegistry.list_registered_ops()
        assert OperationType.RELU in registered
        assert OperationType.SIGMOID in registered
        assert len(registered) == 2
    
    def test_clear_registry(self):
        """Test clearing all registered strategies."""
        strategy = MockRelaxationStrategy(OperationType.RELU)
        RelaxationRegistry.register(OperationType.RELU, strategy)
        
        assert RelaxationRegistry.has_strategy(OperationType.RELU)
        
        RelaxationRegistry.clear()
        
        assert not RelaxationRegistry.has_strategy(OperationType.RELU)
        assert len(RelaxationRegistry.list_registered_ops()) == 0


class TestRegisterDecorator:
    """Test the register_relaxation_strategy decorator."""
    
    def test_decorator_registers_strategy(self):
        """Test that the decorator automatically registers a strategy."""
        
        @register_relaxation_strategy
        class TestStrategy(RelaxationStrategy):
            @property
            def supported_op_type(self) -> OperationType:
                return OperationType.ADD  # Use an unregistered operation
            
            def relax(self, node: Node, interval_inputs: list[IntervalBounds]) -> LinearRelaxation:
                shape = interval_inputs[0].lower.shape
                device = interval_inputs[0].lower.device
                dtype = interval_inputs[0].lower.dtype
                return LinearRelaxation.create_identity(shape, device, dtype)
        
        # Should be automatically registered
        assert RelaxationRegistry.has_strategy(OperationType.ADD)
        
        strategy = RelaxationRegistry.get(OperationType.ADD)
        assert isinstance(strategy, TestStrategy)
    
    def test_decorator_returns_class(self):
        """Test that the decorator returns the class for chaining."""
        
        @register_relaxation_strategy
        class TestStrategy(RelaxationStrategy):
            @property
            def supported_op_type(self) -> OperationType:
                return OperationType.SUB  # Use a different unregistered operation
            
            def relax(self, node: Node, interval_inputs: list[IntervalBounds]) -> LinearRelaxation:
                shape = interval_inputs[0].lower.shape
                device = interval_inputs[0].lower.device
                dtype = interval_inputs[0].lower.dtype
                return LinearRelaxation.create_identity(shape, device, dtype)
        
        # Should be able to instantiate the class
        instance = TestStrategy()
        assert isinstance(instance, RelaxationStrategy)


class TestRelaxationStrategyInterface:
    """Test the RelaxationStrategy interface."""
    
    def setup_method(self):
        """Clear registry before each test."""
        RelaxationRegistry.clear()
    
    def teardown_method(self):
        """Clear registry after each test."""
        RelaxationRegistry.clear()
    
    def test_mock_strategy_basic_usage(self):
        """Test basic usage of a mock strategy."""
        strategy = MockRelaxationStrategy(OperationType.RELU)
        
        # Create a simple node and interval bounds
        metadata = TensorMetadata(
            shape=(2, 3),
            dtype="float32",
        )
        node = Node(
            id=1,
            op_type=OperationType.RELU,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        
        interval_bounds = IntervalBounds(
            lower=torch.tensor([[0.0, -1.0, 2.0], [0.5, -0.5, 1.5]]),
            upper=torch.tensor([[1.0, 0.0, 3.0], [1.5, 0.5, 2.5]]),
        )
        
        # Compute relaxation
        relaxation = strategy.relax(node, [interval_bounds])
        
        # Should return a valid LinearRelaxation
        assert isinstance(relaxation, LinearRelaxation)
        assert relaxation.num_inputs == 1
    
    def test_cannot_instantiate_abstract_strategy(self):
        """Test that abstract RelaxationStrategy cannot be instantiated."""
        with pytest.raises(TypeError):
            RelaxationStrategy()
