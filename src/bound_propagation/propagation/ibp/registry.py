from __future__ import annotations

from typing import TYPE_CHECKING

from ...ir import AbstractValueType

if TYPE_CHECKING:
    from ...ir import OperationType
    from .base import ForwardIBPStrategy


class ForwardIBPStrategyRegistry:
    """Registry for IBP strategies."""

    def __init__(self):
        self._registry: dict[tuple[OperationType, tuple[AbstractValueType, ...]], ForwardIBPStrategy] = {}

    def register(
        self,
        operation_type: OperationType,
        strategy: ForwardIBPStrategy,
        abstract_signature: tuple[AbstractValueType, ...],
    ):
        key = (operation_type, abstract_signature)
        if key in self._registry:
            raise ValueError(
                f"Strategy for operation type '{operation_type}' and signature={abstract_signature} "
                "is already registered."
            )
        self._registry[key] = strategy

    def get_strategy(
        self,
        operation_type: OperationType,
        signature: tuple[AbstractValueType, ...],
    ) -> ForwardIBPStrategy:
        key = (operation_type, signature)
        strategy = self._registry.get(key)
        if strategy is not None:
            return strategy

        # Fallback: For variable-arity operations (e.g., concat, stack),
        # if registered with (ABSTRACT,), match any tuple of all ABSTRACT values

        if all(s == AbstractValueType.ABSTRACT for s in signature):
            fallback_key = (operation_type, (AbstractValueType.ABSTRACT,))
            strategy = self._registry.get(fallback_key)
            if strategy is not None:
                return strategy

        raise ValueError(f"No strategy registered for operation type '{operation_type}' with signature={signature}.")

    @classmethod
    def default_registry(cls) -> ForwardIBPStrategyRegistry:
        """
        Singleton registry instance with default strategies registered.
        """
        if not hasattr(cls, "_default_instance"):
            cls._default_instance = cls()
        return cls._default_instance

    @classmethod
    def register_default(
        cls,
        operation_type: OperationType,
        strategy: ForwardIBPStrategy,
        signature: tuple[AbstractValueType, ...],
    ):
        """Helper to register a default strategy for an operation type."""
        registry = cls.default_registry()
        registry.register(operation_type, strategy, signature)
