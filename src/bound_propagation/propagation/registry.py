"""Target-based strategy registry for bound propagation.

Maps torch.fx node targets (callables and module types) to bounding
strategy instances.  Replaces the old ``OperationType`` enum and the
per-method registries.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import torch
import torch.fx as fx

from .constants import CONSTANT_PRODUCING_TARGETS
from .strategy import BoundingStrategy

# ---------------------------------------------------------------------------
# Method-name → canonical callable mapping (for call_method nodes)
# ---------------------------------------------------------------------------

_METHOD_MAP: dict[str, Callable[..., Any]] = {
    # Arithmetic
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "neg": operator.neg,
    "pow": operator.pow,
    # Activations / element-wise
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "abs": torch.abs,
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "clamp": torch.clamp,
    "reciprocal": torch.reciprocal,
    # Reductions
    "sum": torch.sum,
    "mean": torch.mean,
    "amax": torch.amax,
    "amin": torch.amin,
    # Structural
    "reshape": torch.Tensor.reshape,
    "view": torch.Tensor.view,
    "transpose": torch.Tensor.transpose,
    "permute": torch.Tensor.permute,
    "flatten": torch.Tensor.flatten,
    "unsqueeze": torch.Tensor.unsqueeze,
    "squeeze": torch.Tensor.squeeze,
    "select": torch.Tensor.select,
}


def normalize_target(node: fx.Node, graph_module: fx.GraphModule) -> Callable[..., Any] | type:
    """Derive an extensible dispatch key from an fx node.

    Returns:
        For ``call_function``: the target callable (e.g. ``torch.relu``).
        For ``call_method``: the canonical callable via ``_METHOD_MAP``.
        For ``call_module``: the *type* of the sub-module (e.g. ``nn.Linear``).

    Raises:
        ValueError: If the target cannot be normalized.
    """
    if node.op == "call_function":
        return node.target  # ty:ignore[invalid-return-type]

    if node.op == "call_method":
        # TODO: the method to call depends on the type of the first
        # argument (e.g. "relu" could be Tensor.relu for a torch.Tensor
        # input, or a .relu method from some custom class). For now we
        # assume the common case of Tensor methods.
        name: str = node.target  # ty:ignore[invalid-assignment]
        canonical = _METHOD_MAP.get(name)
        if canonical is None:
            raise ValueError(f"Unsupported call_method target: {name!r}")
        return canonical

    if node.op == "call_module":
        target_str: str = node.target  # ty:ignore[invalid-assignment]
        module = graph_module.get_submodule(target_str)
        return type(module)

    raise ValueError(f"Cannot normalize target for node op={node.op!r}")


T = TypeVar("T", bound=BoundingStrategy)


class TargetRegistry(Generic[T]):
    """Maps normalized fx targets to bounding-strategy instances.

    Each propagation method (IBP, Forward LBP, Backward LBP) maintains its
    own ``TargetRegistry`` so that strategy types are kept separate.

    Usage::

        registry = TargetRegistry[MyReluStrategy]()
        registry.register(torch.relu, MyReluStrategy())
        registry.register(torch.nn.ReLU, MyReluStrategy())

        strategy = registry.get_strategy(some_fx_node, graph_module)
    """

    def __init__(self) -> None:
        self._strategies: dict[Callable[..., Any] | type, T] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        target: Callable[..., Any] | type,
        strategy: T,
    ) -> None:
        """Register a strategy for a single target.

        Raises:
            ValueError: If the target is already registered.
        """
        if target in self._strategies:
            raise ValueError(f"Strategy already registered for target {target!r}")
        self._strategies[target] = strategy

    def register_many(
        self,
        targets: list[Callable[..., Any] | type],
        strategy: T,
    ) -> None:
        """Register the same strategy for multiple targets."""
        for t in targets:
            self.register(t, strategy)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_strategy(self, node: fx.Node, graph_module: fx.GraphModule) -> T:
        """Look up the strategy for *node*.

        Raises:
            ValueError: If no strategy is registered for the normalized target.
        """
        target = normalize_target(node, graph_module)
        strategy = self._strategies.get(target)
        if strategy is None:
            raise ValueError(f"No strategy registered for target {target!r} (node {node.name!r})")
        return strategy

    def is_supported(self, node: fx.Node, graph_module: fx.GraphModule) -> bool:
        """Return ``True`` if *node* can be handled.

        Includes nodes with a registered strategy as well as
        :data:`~bound_propagation.propagation.constants.CONSTANT_PRODUCING_TARGETS`,
        which are evaluated concretely without a strategy.
        """
        try:
            target = normalize_target(node, graph_module)
        except ValueError:
            return False
        if target in self._strategies:
            return True
        return target in CONSTANT_PRODUCING_TARGETS

    def supports_target(self, target: Callable[..., Any] | type) -> bool:
        """Return ``True`` if *target* has a registered strategy."""
        return target in self._strategies

    def targets(self) -> list[Callable[..., Any] | type]:
        """Return all targets that have a registered strategy."""
        return list(self._strategies.keys())
