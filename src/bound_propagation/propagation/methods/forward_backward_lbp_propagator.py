"""Forward-Backward LBP propagator.

Hybrid bound propagation: forward LBP is used to compute linear bounds at
every intermediate node during a forward walk, and backward LBP uses those
forward bounds (concretized on demand) when constructing each operation's
linear relaxation. The output is reported as a linear bound via the
backward tape.

Compared to CROWN-IBP, intermediate bounds come from forward LBP rather
than IBP, which is typically tighter at the cost of extra compute in the
forward pass. Compared to standard CROWN, no recursive backward
concretization occurs -- each node is visited exactly once.
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
from ..backward_lbp.base import BackwardLBPStrategy, IntermediateBoundsProvider
from ..backward_lbp.tape import BackwardTape
from ..context import PropagationContext
from ..forward_lbp import ForwardLBPStrategy, create_default_forward_lbp_registry
from ..forward_lbp.utils import create_identity_bounds
from ..registry import TargetRegistry
from .base import BoundPropagator


class _ForwardLBPBoundsProvider:
    """``IntermediateBoundsProvider`` backed by a forward LBP context.

    The forward LBP context stores :class:`LinearBounds` for abstract nodes
    and :class:`torch.Tensor` for concrete nodes (``get_attr`` or
    non-abstract calls). This provider returns :class:`IntervalBounds` by
    concretizing linear bounds and by wrapping tensors as degenerate
    intervals. Concretizations are cached per fx.Node name.
    """

    def __init__(self, ctx: PropagationContext[LinearBounds]) -> None:
        self._ctx = ctx
        self._cache: dict[str, IntervalBounds] = {}

    def __call__(self, node: fx.Node) -> IntervalBounds:
        if node.name in self._cache:
            return self._cache[node.name]

        value = self._ctx.resolve(node)
        if isinstance(value, LinearBounds):
            result = value.concretize()
        elif isinstance(value, IntervalBounds):
            result = value
        elif isinstance(value, torch.Tensor):
            result = IntervalBounds(value, value)
        else:
            raise TypeError(
                f"Cannot convert value of type {type(value).__name__} for node {node.name!r} to IntervalBounds"
            )

        self._cache[node.name] = result
        return result


class ForwardBackwardLBPPropagator(BoundPropagator):
    """Forward-Backward LBP: forward LBP intermediate bounds + backward LBP at the output.

    Performs a single forward walk of the graph. For each abstract node:

    1. Runs the forward LBP strategy to produce :class:`LinearBounds`,
       stored in a :class:`PropagationContext`.
    2. Runs the backward LBP strategy to build a ``BackwardRelaxation``,
       using a provider that lazily concretizes the forward LBP bounds so
       no recursive backward concretization occurs.

    At the output, the backward tape is run to produce a
    :class:`LinearBounds` per output.
    """

    def __init__(
        self,
        graph_module: fx.GraphModule,
        forward_registry: TargetRegistry[ForwardLBPStrategy] | None = None,
        backward_registry: TargetRegistry[BackwardLBPStrategy] | None = None,
        alpha_config: AlphaOptimizationConfig | None = None,
    ) -> None:
        super().__init__(graph_module)
        self._forward_registry = forward_registry or create_default_forward_lbp_registry()
        self._backward_registry = backward_registry or create_default_backward_lbp_registry()
        self._alpha_config = alpha_config or AlphaOptimizationConfig()
        if self._alpha_config.optimize_intermediate:
            raise ValueError(
                "ForwardBackwardLBPPropagator does not support optimize_intermediate=True: "
                "intermediate bounds come from a single forward LBP sweep, not a recursive CROWN pass. "
                "All forward and backward alphas are already co-optimized when enabled=True."
            )

    @property
    def method_name(self) -> str:
        return "forward_backward_lbp"

    @property
    def forward_registry(self) -> TargetRegistry[ForwardLBPStrategy]:
        return self._forward_registry

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
        ctx = PropagationContext[LinearBounds](self._graph_module)
        ctx.alpha_provider = alpha_provider
        tape = BackwardTape(self._graph_module, input_regions)
        tape.alpha_provider = alpha_provider
        provider = _ForwardLBPBoundsProvider(ctx)

        # Seed placeholder forward LBP bounds as identities over each region.
        for idx, (ph, region) in enumerate(zip(placeholders, input_regions, strict=True)):
            ctx.store(ph, create_identity_bounds(idx, region, region.lower.shape))

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
        ctx: PropagationContext[LinearBounds],
        tape: BackwardTape,
        provider: IntermediateBoundsProvider,
    ) -> None:
        is_abstract = node.meta.get("is_abstract", True)

        if not is_abstract:
            value = self._evaluate_concrete(node, ctx)
            ctx.store(node, value)
            tape.record_concrete(node, value)
            return

        fwd_strategy = self._forward_registry.get_strategy(node, self._graph_module)
        ctx.store(node, fwd_strategy.propagate_forward(node, ctx))

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
    def _evaluate_concrete(node: fx.Node, ctx: PropagationContext[LinearBounds]) -> torch.Tensor:
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
