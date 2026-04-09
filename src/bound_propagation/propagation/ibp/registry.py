from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...ir import OperationType
    from .base import ForwardIBPStrategy

class ForwardIBPStrategyRegistry:
    """Registry for IBP strategies."""

    def __init__(self):
        self._registry: dict[OperationType, ForwardIBPStrategy] = {}

    def register(self, operation_type: OperationType, strategy: ForwardIBPStrategy):
        if operation_type in self._registry:
            raise ValueError(f"Strategy for operation type '{operation_type}' is already registered.")
        self._registry[operation_type] = strategy

    def get_strategy(self, operation_type: OperationType) -> ForwardIBPStrategy:
        if operation_type not in self._registry:
            raise ValueError(f"No strategy registered for operation type '{operation_type}'.")
        return self._registry[operation_type]


    @classmethod
    def default_registry(cls) -> ForwardIBPStrategyRegistry:
        """
        Singleton registry instance with default strategies registered.
        """
        if not hasattr(cls, "_default_instance"):
            cls._default_instance = cls()
        return cls._default_instance

    @classmethod
    def register_default(cls, operation_type: OperationType, strategy: ForwardIBPStrategy):
        """Helper to register a default strategy for an operation type."""
        registry = cls.default_registry()
        registry.register(operation_type, strategy)
