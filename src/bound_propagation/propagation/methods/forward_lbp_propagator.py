"""Forward Linear Bound Propagation (LBP) propagator.

Walks a :class:`torch.fx.GraphModule` in forward topological order,
propagating :class:`LinearBounds` through each operation node.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from ...regions import SimpleRegion
from ..context import PropagationContext
from ..forward_lbp import ForwardLBPStrategy, create_default_forward_lbp_registry
from ..forward_lbp.utils import create_identity_bounds
from ..registry import TargetRegistry
from .base import BoundPropagator


class ForwardLBPPropagator(BoundPropagator):
    """Forward linear bound propagation.

    Tracks affine (linear) dependencies between input perturbations and
    intermediate values.  More precise than IBP at a higher
    computational cost.

    Args:
        graph_module: The traced ``fx.GraphModule`` with metadata
            annotations (from :class:`MetadataPass`).
        registry: Strategy registry. If ``None``, uses the built-in
            default Forward LBP registry.
    """

    def __init__(
        self,
        graph_module: fx.GraphModule,
        registry: TargetRegistry[ForwardLBPStrategy] | None = None,
    ) -> None:
        super().__init__(graph_module)

        self._registry = registry or create_default_forward_lbp_registry()

    @property
    def method_name(self) -> str:
        return "forward_lbp"

    def propagate(
        self,
        input_regions: Sequence[SimpleRegion],
    ) -> Sequence[LinearBounds]:
        placeholders = self._placeholder_nodes()
        if len(input_regions) != len(placeholders):
            raise ValueError(f"Expected {len(placeholders)} input regions, got {len(input_regions)}")

        ctx = self._new_context()

        # Seed placeholder bounds
        for id, (ph, region) in enumerate(zip(placeholders, input_regions, strict=True)):
            shape = region.lower.shape
            ctx.store(ph, create_identity_bounds(id, region, shape))

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
                outputs: list[LinearBounds] = []
                output_values = args[0] if isinstance(args[0], (tuple, list)) else args
                for val in output_values:
                    if not isinstance(val, LinearBounds):
                        raise TypeError(f"Expected output to be LinearBounds, got {type(val)}")
                    outputs.append(val)
                return outputs

            ctx.release(node)

        raise RuntimeError("Graph has no output node")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def registry(self) -> TargetRegistry[ForwardLBPStrategy]:
        """The strategy registry used for dispatch."""
        return self._registry

    def _new_context(self) -> PropagationContext[LinearBounds]:
        """Create a fresh :class:`PropagationContext`."""
        return PropagationContext[LinearBounds](self._graph_module)

    def _propagate_operation(self, node: fx.Node, ctx: PropagationContext[LinearBounds]) -> None:
        """Dispatch *node* to its Forward LBP strategy or evaluate concretely."""
        is_abstract = node.meta.get("is_abstract", True)

        if not is_abstract:
            ctx.store(node, self._evaluate_concrete(node, ctx))
            return

        strategy = self.registry.get_strategy(node, self._graph_module)
        result = strategy.propagate_forward(node, ctx)
        ctx.store(node, result)

    @staticmethod
    def _evaluate_concrete(node: fx.Node, ctx: PropagationContext[LinearBounds]) -> torch.Tensor:
        """Concretely evaluate a non-abstract node."""
        args, kwargs = ctx.resolve_args(node)
        target = node.target
        if node.op == "call_function":
            return target(*args, **kwargs)  # ty:ignore[call-non-callable]
        if node.op == "call_method":
            return getattr(args[0], target)(*args[1:], **kwargs)  # ty:ignore[invalid-argument-type]
        if node.op == "call_module":
            module = ctx.get_module(target)  # ty:ignore[invalid-argument-type]
            return module(*args, **kwargs)
        raise ValueError(f"Cannot evaluate node op={node.op!r}")
