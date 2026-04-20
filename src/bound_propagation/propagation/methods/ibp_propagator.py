"""Interval Bound Propagation (IBP) propagator.

Walks a :class:`torch.fx.GraphModule` in forward topological order,
propagating :class:`IntervalBounds` through each operation node.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from ...regions import SimpleRegion
from ..constants import CONSTANT_PRODUCING_TARGETS, evaluate_constant_producer
from ..context import PropagationContext
from ..ibp import ForwardIBPStrategy, create_default_ibp_registry
from ..registry import TargetRegistry
from .base import BoundPropagator


class IBPPropagator(BoundPropagator):
    """Forward interval bound propagation.

    Uses interval arithmetic rules for each operation. Fast but less
    precise than LBP because it does not track linear dependencies.

    Args:
        graph_module: The traced ``fx.GraphModule`` with metadata
            annotations (from :class:`MetadataPass`).
        registry: Strategy registry. If ``None``, uses the built-in
            default IBP registry.
    """

    def __init__(
        self,
        graph_module: fx.GraphModule,
        registry: TargetRegistry[ForwardIBPStrategy] | None = None,
    ) -> None:
        super().__init__(graph_module)
        self._registry = registry or create_default_ibp_registry()

    @property
    def method_name(self) -> str:
        return "ibp"

    def propagate(
        self,
        input_regions: Sequence[SimpleRegion],
        batch_ndim: int = 0,
    ) -> IntervalBounds:
        del batch_ndim  # IBP is shape-transparent; accepted for API uniformity.
        placeholders = self._placeholder_nodes()
        if len(input_regions) != len(placeholders):
            raise ValueError(f"Expected {len(placeholders)} input regions, got {len(input_regions)}")

        ctx = self._new_context()

        # Seed placeholder bounds
        for ph, region in zip(placeholders, input_regions, strict=True):
            lower, upper = region.aabb()
            ctx.store(ph, IntervalBounds(lower, upper))

        # Forward walk (graph.nodes is already in topological order)
        for node in self._graph_module.graph.nodes:
            if node.op == "placeholder":
                continue
            elif node.op == "get_attr":
                ctx.store(node, ctx.fetch_attr(node.target))
            elif node.op in ("call_function", "call_method", "call_module"):
                self._propagate_operation(node, ctx)
            elif node.op == "output":
                args, _ = ctx.resolve_args(node)
                value = args[0]
                if not isinstance(value, IntervalBounds):
                    raise TypeError(f"Expected output to be IntervalBounds, got {type(value)}")
                return value

            ctx.release(node)

        raise RuntimeError("Graph has no output node")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def registry(self) -> TargetRegistry[ForwardIBPStrategy]:
        """The strategy registry used for dispatch."""
        return self._registry

    def _new_context(self) -> PropagationContext[IntervalBounds]:
        """Create a fresh :class:`PropagationContext`."""
        return PropagationContext[IntervalBounds](self._graph_module)

    def _propagate_operation(self, node: fx.Node, ctx: PropagationContext[IntervalBounds]) -> None:
        """Dispatch *node* to its IBP strategy or evaluate concretely."""
        is_abstract = node.meta.get("is_abstract", True)

        if not is_abstract:
            # Constant sub-expression — evaluate concretely
            ctx.store(node, self._evaluate_concrete(node, ctx))
            return

        strategy = self.registry.get_strategy(node, self._graph_module)
        result = strategy.propagate_forward(node, ctx)
        ctx.store(node, result)

    @staticmethod
    def _evaluate_concrete(node: fx.Node, ctx: PropagationContext[IntervalBounds]) -> torch.Tensor:
        """Concretely evaluate a non-abstract node."""
        target = node.target
        if node.op == "call_function" and target in CONSTANT_PRODUCING_TARGETS:
            return evaluate_constant_producer(node)

        args, kwargs = ctx.resolve_args(node)
        if node.op == "call_function":
            return target(*args, **kwargs)  # ty:ignore[call-non-callable]
        if node.op == "call_method":
            return getattr(args[0], target)(*args[1:], **kwargs)  # ty:ignore[invalid-argument-type]
        if node.op == "call_module":
            module = ctx.get_module(target)  # ty:ignore[invalid-argument-type]
            return module(*args, **kwargs)
        raise ValueError(f"Cannot evaluate node op={node.op!r}")
