from __future__ import annotations

from typing import TYPE_CHECKING

from ...ir import OperationType

if TYPE_CHECKING:
    from .base import IntervalBoundingStrategy

class IBPStrategyRegistry:
    """Registry for IBP strategies."""

    def __init__(self):
        self._registry: dict[OperationType, IntervalBoundingStrategy] = {}

    def register(self, operation_type: OperationType, strategy: IntervalBoundingStrategy):
        if operation_type in self._registry:
            raise ValueError(f"Strategy for operation type '{operation_type}' is already registered.")
        self._registry[operation_type] = strategy

    def get_strategy(self, operation_type: OperationType) -> IntervalBoundingStrategy:
        if operation_type not in self._registry:
            raise ValueError(f"No strategy registered for operation type '{operation_type}'.")
        return self._registry[operation_type]


    @classmethod
    def default_registry(cls) -> IBPStrategyRegistry:
        """
        Singleton registry instance with default strategies registered.
        """
        if not hasattr(cls, "_default_instance"):
            cls._default_instance = cls()
        return cls._default_instance

    @classmethod
    def register_default(cls, operation_type: OperationType, strategy: IntervalBoundingStrategy):
        """Helper to register a default strategy for an operation type."""
        registry = cls.default_registry()
        registry.register(operation_type, strategy)
