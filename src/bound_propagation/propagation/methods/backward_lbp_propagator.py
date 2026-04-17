"""Backward Linear Bound Propagation (CROWN) propagator.

Walks a :class:`torch.fx.GraphModule` in forward topological order,
building a symbolic relaxation tree.  At the output, backward-concretizes
the tree to produce :class:`LinearBounds`.

Intermediate bounds at nonlinear nodes are computed via recursive CROWN
(backward through the symbolic subtree built so far) rather than IBP,
yielding tighter bounds.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from ...regions import SimpleRegion
from ..backward_lbp import create_default_backward_lbp_registry
from ..backward_lbp.base import BackwardLBPStrategy
from ..context import PropagationContext
from ..linear_relaxations.base import (
    InputIdentityRelaxation,
    OutputLinearRelaxation,
    SymbolicLinearRelaxation,
)
from ..registry import TargetRegistry
from .base import BoundPropagator


class BackwardLBPPropagator(BoundPropagator):
    """Backward-mode linear bound propagation (CROWN).

    Builds a symbolic relaxation tree during a forward graph walk,
    then backward-concretizes at the output to obtain ``LinearBounds``.

    Args:
        graph_module: The traced ``fx.GraphModule`` with metadata
            annotations (from :class:`MetadataPass`).
        registry: Strategy registry.  If ``None``, uses the built-in
            default backward LBP registry.
    """

    def __init__(
        self,
        graph_module: fx.GraphModule,
        registry: TargetRegistry | None = None,
    ) -> None:
        super().__init__(graph_module, registry or create_default_backward_lbp_registry())

    @property
    def method_name(self) -> str:
        return "backward_lbp"

    def propagate(
        self,
        input_regions: Sequence[SimpleRegion],
    ) -> Sequence[LinearBounds]:
        placeholders = self._placeholder_nodes()
        if len(input_regions) != len(placeholders):
            raise ValueError(f"Expected {len(placeholders)} input regions, got {len(input_regions)}")

        ctx = self._new_context()

        # Seed placeholders with InputIdentityRelaxation
        for ph, region in zip(placeholders, input_regions, strict=True):
            ctx.store(ph, InputIdentityRelaxation(input_region=region))

        # Forward walk: build symbolic relaxation tree
        for node in self._graph_module.graph.nodes:
            if node.op == "placeholder":
                continue
            elif node.op == "get_attr":
                ctx.store(node, ctx.fetch_attr(node.target))
            elif node.op in ("call_function", "call_method", "call_module"):
                self._propagate_operation(node, ctx)
            elif node.op == "output":
                return self._handle_output(node, ctx)

            ctx.release(node)

        raise RuntimeError("Graph has no output node")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _new_context(self) -> PropagationContext[SymbolicLinearRelaxation]:
        """Create a fresh :class:`PropagationContext`."""
        return PropagationContext[SymbolicLinearRelaxation](self._graph_module)

    def _propagate_operation(self, node: fx.Node, ctx: PropagationContext[SymbolicLinearRelaxation]) -> None:
        """Build symbolic node or evaluate concretely."""
        is_abstract = node.meta.get("is_abstract", True)

        if not is_abstract:
            ctx.store(node, self._evaluate_concrete(node, ctx))
            return

        strategy = self.registry.get_strategy(node, self._graph_module)
        if not isinstance(strategy, BackwardLBPStrategy):
            raise TypeError(f"Expected BackwardLBPStrategy for node {node.name!r}, got {type(strategy).__name__}")
        sym = strategy.build_symbolic(node, ctx)
        ctx.store(node, sym)

    def _handle_output(self, node: fx.Node, ctx: PropagationContext[SymbolicLinearRelaxation]) -> list[LinearBounds]:
        """Backward-concretize each output symbolic relaxation."""
        args, _ = ctx.resolve_args(node)
        output_values = args[0] if isinstance(args[0], (tuple, list)) else args

        results: list[LinearBounds] = []
        for val in output_values:
            if not isinstance(val, SymbolicLinearRelaxation):
                raise TypeError(f"Expected output to be SymbolicLinearRelaxation, got {type(val)}")

            # Determine output shape, dtype, device from the output node's input meta
            output_node = node.args[0]
            if isinstance(output_node, (tuple, list)):
                # Multi-output: use the matching element's meta
                idx = list(output_values).index(val)
                meta_node = node.args[0][idx]
            else:
                meta_node = output_node

            output_shape = meta_node.meta["tensor_meta"]["shape"]
            dtype = meta_node.meta["tensor_meta"]["dtype"]
            device = self._infer_device()

            out_relaxation = OutputLinearRelaxation(inputs=[val], output_shape=output_shape)
            results.append(out_relaxation.concretize(batch_ndim=0, dtype=dtype, device=device))

        return results

    def _infer_device(self) -> torch.device:
        """Infer device from model parameters or default to CPU."""
        for param in self._graph_module.parameters():
            return param.device
        return torch.device("cpu")

    @staticmethod
    def _evaluate_concrete(node: fx.Node, ctx: PropagationContext) -> torch.Tensor:
        """Concretely evaluate a non-abstract node."""
        args, kwargs = ctx.resolve_args(node)
        target = node.target
        if node.op == "call_function":
            return target(*args, **kwargs)
        if node.op == "call_method":
            return getattr(args[0], target)(*args[1:], **kwargs)
        if node.op == "call_module":
            module = ctx.get_module(target)
            return module(*args, **kwargs)
        raise ValueError(f"Cannot evaluate node op={node.op!r}")
