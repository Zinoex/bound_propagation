"""Target-based strategy registry for bound propagation.

Maps torch.fx node targets (callables and module types) to bounding
strategy instances.  Replaces the old ``OperationType`` enum and the
per-method registries.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx

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
        return node.target  # type: ignore[return-value]

    if node.op == "call_method":
        # TODO: the method to call depends on the type of the first
        # argument (e.g. "relu" could be Tensor.relu for a torch.Tensor
        # input, or a .relu method from some custom class). For now we
        # assume the common case of Tensor methods.
        name: str = node.target  # type: ignore[assignment]
        canonical = _METHOD_MAP.get(name)
        if canonical is None:
            raise ValueError(f"Unsupported call_method target: {name!r}")
        return canonical

    if node.op == "call_module":
        target_str: str = node.target  # type: ignore[assignment]
        module = graph_module.get_submodule(target_str)
        return type(module)

    raise ValueError(f"Cannot normalize target for node op={node.op!r}")


class TargetRegistry:
    """Maps normalized fx targets to bounding-strategy instances.

    Each propagation method (IBP, Forward LBP, Backward LBP) maintains its
    own ``TargetRegistry`` so that strategy types are kept separate.

    Usage::

        registry = TargetRegistry()
        registry.register(torch.relu, MyReluStrategy())
        registry.register(torch.nn.ReLU, MyReluStrategy())

        strategy = registry.get_strategy(some_fx_node, graph_module)
    """

    def __init__(self) -> None:
        self._strategies: dict[Callable[..., Any] | type, BoundingStrategy] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        target: Callable[..., Any] | type,
        strategy: BoundingStrategy,
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
        strategy: BoundingStrategy,
    ) -> None:
        """Register the same strategy for multiple targets."""
        for t in targets:
            self.register(t, strategy)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_strategy(self, node: fx.Node, graph_module: fx.GraphModule) -> BoundingStrategy:
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
        """Return ``True`` if a strategy exists for *node*."""
        try:
            target = normalize_target(node, graph_module)
        except ValueError:
            return False
        return target in self._strategies

    def supports_target(self, target: Callable[..., Any] | type) -> bool:
        """Return ``True`` if *target* has a registered strategy."""
        return target in self._strategies
