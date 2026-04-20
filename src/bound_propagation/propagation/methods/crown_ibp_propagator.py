"""CROWN-IBP propagator.

Hybrid bound propagation: IBP is used to compute interval bounds at every
intermediate node during a forward walk, and backward LBP uses those IBP
bounds when constructing each operation's linear relaxation. The output
is reported as a linear bound via the backward tape.

This trades some precision versus standard CROWN (which uses recursive
backward concretization for intermediate bounds) for a single forward
sweep with no recursion.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.fx as fx

from ...bounds import IntervalBounds, LinearBounds
from ...regions import SimpleRegion
from ..alpha_optimization import (
    AlphaOptimizationConfig,
    AlphaProvider,
    NullAlphaProvider,
    run_alpha_optimization,
)
from ..backward_lbp import create_default_backward_lbp_registry
from ..backward_lbp.base import BackwardLBPStrategy, PrecomputedBoundsProvider
from ..backward_lbp.tape import BackwardTape
from ..context import PropagationContext
from ..ibp import ForwardIBPStrategy, create_default_ibp_registry
from ..registry import TargetRegistry
from .base import BoundPropagator


class CROWNIBPPropagator(BoundPropagator):
    """CROWN-IBP: IBP intermediate bounds + backward LBP at the output.

    Performs a single forward walk of the graph. For each abstract node:

    1. Runs the IBP strategy to produce :class:`IntervalBounds`, stored
       in a :class:`PropagationContext`.
    2. Runs the backward LBP strategy to build a :class:`BackwardRelaxation`,
       using a :class:`PrecomputedBoundsProvider` backed by the IBP
       context so no recursive concretization occurs.

    At the output, the backward tape is concretized to obtain a
    :class:`LinearBounds` per output.
    """

    def __init__(
        self,
        graph_module: fx.GraphModule,
        ibp_registry: TargetRegistry[ForwardIBPStrategy] | None = None,
        backward_registry: TargetRegistry[BackwardLBPStrategy] | None = None,
        alpha_config: AlphaOptimizationConfig | None = None,
    ) -> None:
        super().__init__(graph_module)
        self._ibp_registry = ibp_registry or create_default_ibp_registry()
        self._backward_registry = backward_registry or create_default_backward_lbp_registry()
        self._alpha_config = alpha_config or AlphaOptimizationConfig()
        if self._alpha_config.optimize_intermediate:
            raise ValueError(
                "CROWNIBPPropagator does not support optimize_intermediate=True: "
                "intermediate bounds come from IBP, which has no alpha-CROWN knobs."
            )

    @property
    def method_name(self) -> str:
        return "crown_ibp"

    @property
    def ibp_registry(self) -> TargetRegistry[ForwardIBPStrategy]:
        return self._ibp_registry

    @property
    def backward_registry(self) -> TargetRegistry[BackwardLBPStrategy]:
        return self._backward_registry

    @property
    def alpha_config(self) -> AlphaOptimizationConfig:
        return self._alpha_config

    def propagate(
        self,
        input_regions: Sequence[SimpleRegion],
        batch_ndim: int = 0,
    ) -> LinearBounds:
        placeholders = self._placeholder_nodes()
        if len(input_regions) != len(placeholders):
            raise ValueError(f"Expected {len(placeholders)} input regions, got {len(input_regions)}")

        regions_list = list(input_regions)
        if not self._alpha_config.enabled:
            return self._propagate_once(regions_list, NullAlphaProvider(), batch_ndim)
        return run_alpha_optimization(
            propagate_once=lambda provider: self._propagate_once(regions_list, provider, batch_ndim),
            config=self._alpha_config,
        )

    def _propagate_once(
        self,
        input_regions: list[SimpleRegion],
        alpha_provider: AlphaProvider,
        batch_ndim: int,
    ) -> LinearBounds:
        placeholders = self._placeholder_nodes()
        ctx = PropagationContext[IntervalBounds](self._graph_module)
        tape = BackwardTape(self._graph_module, input_regions)
        # IBP has no alpha knobs — only the backward tape consumes the provider.
        tape.alpha_provider = alpha_provider
        provider = PrecomputedBoundsProvider.from_context(ctx)

        # Seed placeholder IBP bounds.
        for ph, region in zip(placeholders, input_regions, strict=True):
            lower, upper = region.aabb()
            ctx.store(ph, IntervalBounds(lower, upper))

        for node in self._graph_module.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "get_attr":
                value = ctx.fetch_attr(node.target)
                ctx.store(node, value)
                tape.record_concrete(node, value)
                continue
            if node.op in ("call_function", "call_method", "call_module"):
                self._propagate_operation(node, ctx, tape, provider)
                continue
            if node.op == "output":
                return self._handle_output(node, tape, batch_ndim)

        raise RuntimeError("Graph has no output node")

    def _propagate_operation(
        self,
        node: fx.Node,
        ctx: PropagationContext[IntervalBounds],
        tape: BackwardTape,
        provider: PrecomputedBoundsProvider,
    ) -> None:
        is_abstract = node.meta.get("is_abstract", True)

        if not is_abstract:
            value = self._evaluate_concrete(node, ctx)
            ctx.store(node, value)
            tape.record_concrete(node, value)
            return

        ibp_strategy = self._ibp_registry.get_strategy(node, self._graph_module)
        # IBP has no alpha-CROWN knobs; treat its intermediate bounds as constants
        # to avoid spending autograd on the un-optimized weight path during the
        # alpha-optimization loop.
        with torch.no_grad():
            ctx.store(node, ibp_strategy.propagate_forward(node, ctx))

        bwd_strategy = self._backward_registry.get_strategy(node, self._graph_module)
        if not isinstance(bwd_strategy, BackwardLBPStrategy):
            raise TypeError(f"Expected BackwardLBPStrategy for node {node.name!r}, got {type(bwd_strategy).__name__}")
        relaxation = bwd_strategy.build_relaxation(node, tape, provider)
        tape.record(node, relaxation)

    def _handle_output(self, node: fx.Node, tape: BackwardTape, batch_ndim: int) -> LinearBounds:
        output_arg = node.args[0]
        if not isinstance(output_arg, fx.Node):
            raise TypeError(f"Expected output to be an fx.Node, got {type(output_arg)}")
        return tape.backward_from(output_arg, batch_ndim=batch_ndim)

    @staticmethod
    def _evaluate_concrete(node: fx.Node, ctx: PropagationContext[IntervalBounds]) -> torch.Tensor:
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
