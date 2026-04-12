"""
Base classes and registry for relaxation strategies.

RelaxationStrategy implementations compute linear approximations of operations
given concrete interval bounds on their inputs. These relaxations are shared
between forward and backward LBP propagation.
"""

from abc import ABC, abstractmethod

from ...bounds import IntervalBounds
from ...ir import Node, OperationType
from .linear_relaxation import LinearRelaxation


class RelaxationStrategy(ABC):
    """
    Abstract base class for computing linear relaxations of operations.

    A RelaxationStrategy computes a linear approximation of an operation's
    output given concrete interval bounds on its inputs. The relaxation
    is sound (contains all possible outputs) but may be conservative.

    Subclasses implement the relax() method for specific operation types.
    """

    @abstractmethod
    def relax(
        self,
        node: Node,
        interval_inputs: list[IntervalBounds],
    ) -> LinearRelaxation:
        """
        Compute a linear relaxation for this operation.

        Args:
            node: The operation node containing op_type and attributes.
            interval_inputs: Concrete interval bounds for each input.
                             Length must match the number of inputs to the operation.

        Returns:
            LinearRelaxation: A linear approximation of the operation.
                             For operation z = f(x1, x2, ...), returns coefficients
                             and biases such that:
                             z_lower >= sum(W_i_lower @ x_i) + b_lower
                             z_upper <= sum(W_i_upper @ x_i) + b_upper

        Raises:
            ValueError: If the number of interval inputs doesn't match expectations
                       or if the operation is not supported.
        """
        pass

    @property
    @abstractmethod
    def supported_op_type(self) -> OperationType:
        """Return the operation type this strategy handles."""
        pass


class RelaxationRegistry:
    """
    Registry for looking up relaxation strategies by operation type.

    This allows decoupling relaxation computation from propagation logic.
    Strategies are registered at module import time and retrieved when needed.
    """

    _registry: dict[OperationType, RelaxationStrategy] = {}

    @classmethod
    def register(
        cls,
        op_type: OperationType,
        strategy: RelaxationStrategy,
    ) -> None:
        """
        Register a relaxation strategy for an operation type.

        Args:
            op_type: The operation type to register for.
            strategy: The strategy instance to use for this operation.

        Raises:
            ValueError: If a strategy is already registered for this operation.
        """
        if op_type in cls._registry:
            raise ValueError(
                f"Relaxation strategy already registered for {op_type}. Existing: {cls._registry[op_type].__class__.__name__}, New: {strategy.__class__.__name__}"
            )
        cls._registry[op_type] = strategy

    @classmethod
    def get(cls, op_type: OperationType) -> RelaxationStrategy | None:
        """
        Get the relaxation strategy for an operation type.

        Args:
            op_type: The operation type to look up.

        Returns:
            The registered RelaxationStrategy, or None if not found.
        """
        return cls._registry.get(op_type)

    @classmethod
    def has_strategy(cls, op_type: OperationType) -> bool:
        """
        Check if a strategy is registered for an operation type.

        Args:
            op_type: The operation type to check.

        Returns:
            True if a strategy is registered, False otherwise.
        """
        return op_type in cls._registry

    @classmethod
    def clear(cls) -> None:
        """Clear all registered strategies. Mainly for testing."""
        cls._registry.clear()

    @classmethod
    def list_registered_ops(cls) -> list[OperationType]:
        """Return a list of all operation types with registered strategies."""
        return list(cls._registry.keys())


def register_relaxation_strategy(strategy: RelaxationStrategy) -> RelaxationStrategy:
    """
    Decorator to automatically register a relaxation strategy.

    Usage:
        @register_relaxation_strategy
        class ReluRelaxationStrategy(RelaxationStrategy):
            @property
            def supported_op_type(self) -> OperationType:
                return OperationType.RELU
            ...

    Args:
        strategy: The strategy class (uninstantiated).

    Returns:
        The same strategy class (for chaining).
    """
    # Instantiate the strategy
    instance = strategy()
    # Register it
    RelaxationRegistry.register(instance.supported_op_type, instance)
    return strategy
