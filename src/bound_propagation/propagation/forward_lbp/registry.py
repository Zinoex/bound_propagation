from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...ir import OperationType
    from .base import ForwardLBPStrategy


class ForwardLBPStrategyRegistry:
    """
    Registry for forward linear bounding strategies.
    """

    def __init__(self):
        self._registry: dict[OperationType, ForwardLBPStrategy] = {}

    def register(self, op_type: OperationType, strategy: ForwardLBPStrategy):
        self._registry[op_type] = strategy

    def get_strategy(self, op_type: OperationType):
        if op_type not in self._registry:
            raise ValueError(f"No strategy registered for op type {op_type}")
        return self._registry[op_type]

    @classmethod
    def default_registry(cls) -> ForwardLBPStrategyRegistry:
        """
        Singleton registry instance with default strategies registered.
        """
        if not hasattr(cls, "_default_instance"):
            cls._default_instance = cls()
        return cls._default_instance

    @classmethod
    def register_default(cls, op_type: OperationType, strategy: ForwardLBPStrategy):
        """Helper to register a default strategy for an operation type."""
        registry = cls.default_registry()
        registry.register(op_type, strategy)
