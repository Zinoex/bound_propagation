"""
Registry for bounding strategies.

Maps (operation_type, method) to strategy instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir import OperationType
    from .strategy import BoundingStrategy


class StrategyRegistry:
    """
    Registry mapping (operation_type, method) to bounding strategies.

    This allows registering different strategies for different operations
    and different propagation methods. For example:
    - (ADD, "ibp") -> IBP_AddStrategy()
    - (RELU, "ibp") -> IBP_ReLUStrategy()
    - (LINEAR, "backward") -> Backward_LinearStrategy()

    The registry supports:
    - Registering strategies for specific (op, method) pairs
    - Registering fallback strategies for all operations of a method
    - Querying available strategies
    """

    def __init__(self):
        """Initialize empty registry."""
        # Map: (operation_type, method) -> strategy
        self._strategies: dict[tuple[OperationType, str], BoundingStrategy] = {}

        # Fallback strategies for a method (apply to all ops if no specific strategy)
        self._fallback_strategies: dict[str, BoundingStrategy] = {}

    def register(
        self,
        operation_type: OperationType,
        method: str,
        strategy: BoundingStrategy,
    ) -> None:
        """
        Register a strategy for a specific (operation, method) pair.

        Args:
            operation_type: The operation type this strategy handles
            method: The bounding method name (e.g., "ibp", "forward", "backward")
            strategy: The strategy instance

        Example:
            registry.register(OperationType.ADD, "ibp", IBP_AddStrategy())
        """
        key = (operation_type, method)
        if key in self._strategies:
            raise ValueError(
                f"Strategy already registered for {operation_type} with method {method}"
            )
        self._strategies[key] = strategy

    def register_fallback(self, method: str, strategy: BoundingStrategy) -> None:
        """
        Register a fallback strategy for a method.

        This strategy will be used for any operation that doesn't have
        a specific strategy registered for this method.

        Args:
            method: The bounding method name
            strategy: The fallback strategy instance

        Example:
            # Use IBP for all ops by default
            registry.register_fallback("ibp", IBP_GenericStrategy())
        """
        if method in self._fallback_strategies:
            raise ValueError(f"Fallback strategy already registered for method {method}")
        self._fallback_strategies[method] = strategy

    def get(
        self,
        operation_type: OperationType,
        method: str,
    ) -> BoundingStrategy | None:
        """
        Get the strategy for an (operation, method) pair.

        First looks for a specific strategy, then falls back to the
        method's fallback strategy if available.

        Args:
            operation_type: The operation type
            method: The bounding method name

        Returns:
            Strategy instance, or None if not found

        Example:
            strategy = registry.get(OperationType.ADD, "ibp")
            if strategy:
                bounds = strategy.compute_bounds(node, input_bounds, config)
        """
        # Try specific strategy first
        key = (operation_type, method)
        if key in self._strategies:
            return self._strategies[key]

        # Fall back to method fallback
        return self._fallback_strategies.get(method)

    def has_strategy(self, operation_type: OperationType, method: str) -> bool:
        """
        Check if a strategy is available for an (operation, method) pair.

        Args:
            operation_type: The operation type
            method: The bounding method name

        Returns:
            True if a strategy is available (specific or fallback)
        """
        return self.get(operation_type, method) is not None

    def get_supported_methods(self, operation_type: OperationType) -> set[str]:
        """
        Get all methods that have strategies for an operation.

        Args:
            operation_type: The operation type

        Returns:
            Set of method names that support this operation

        Example:
            methods = registry.get_supported_methods(OperationType.RELU)
            # methods might be: {"ibp", "forward", "backward"}
        """
        methods = set()

        # Specific strategies
        for (op_type, method), _ in self._strategies.items():
            if op_type == operation_type:
                methods.add(method)

        # Fallback strategies
        methods.update(self._fallback_strategies.keys())

        return methods

    def get_registered_operations(self, method: str) -> set[OperationType]:
        """
        Get all operations that have specific strategies for a method.

        Note: This doesn't include operations covered by fallback strategies.

        Args:
            method: The bounding method name

        Returns:
            Set of operation types with specific strategies for this method

        Example:
            ops = registry.get_registered_operations("ibp")
            # ops might be: {OperationType.ADD, OperationType.MUL, OperationType.RELU}
        """
        operations = set()
        for (op_type, m), _ in self._strategies.items():
            if m == method:
                operations.add(op_type)
        return operations

    def clear(self) -> None:
        """Clear all registered strategies."""
        self._strategies.clear()
        self._fallback_strategies.clear()

    def __len__(self) -> int:
        """Get the number of registered strategies."""
        return len(self._strategies) + len(self._fallback_strategies)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StrategyRegistry("
            f"strategies={len(self._strategies)}, "
            f"fallbacks={len(self._fallback_strategies)})"
        )


# Global registry instance
_global_registry: StrategyRegistry | None = None


def get_global_registry() -> StrategyRegistry:
    """
    Get the global strategy registry.

    Returns:
        The global registry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = StrategyRegistry()
    return _global_registry


def register_strategy(
    operation_type: OperationType,
    method: str,
    strategy: BoundingStrategy,
) -> None:
    """
    Register a strategy in the global registry.

    Args:
        operation_type: The operation type
        method: The bounding method name
        strategy: The strategy instance
    """
    get_global_registry().register(operation_type, method, strategy)


def register_fallback(method: str, strategy: BoundingStrategy) -> None:
    """
    Register a fallback strategy in the global registry.

    Args:
        method: The bounding method name
        strategy: The fallback strategy instance
    """
    get_global_registry().register_fallback(method, strategy)


def get_strategy(
    operation_type: OperationType,
    method: str,
) -> BoundingStrategy | None:
    """
    Get a strategy from the global registry.

    Args:
        operation_type: The operation type
        method: The bounding method name

    Returns:
        Strategy instance, or None if not found
    """
    return get_global_registry().get(operation_type, method)
