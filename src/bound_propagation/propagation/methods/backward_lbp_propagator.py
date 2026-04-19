"""Backward Linear Bound Propagation (CROWN) propagator.

Walks a torch.fx.GraphModule in forward topological order, building a
BackwardTape of relaxations. At the output, runs the tape's backward
algorithm to produce LinearBounds.

Intermediate bounds at nonlinear nodes are computed via recursive CROWN
(backward through the partial tape) rather than IBP, yielding tighter bounds.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.fx as fx

from ...bounds import LinearBounds
from ...regions import SimpleRegion
from ..alpha_optimization import (
    AlphaOptimizationConfig,
    AlphaProvider,
    NullAlphaProvider,
    run_alpha_optimization,
)
from ..backward_lbp import create_default_backward_lbp_registry
from ..backward_lbp.base import BackwardLBPStrategy, CrownBoundsProvider
from ..backward_lbp.tape import BackwardTape
from ..registry import TargetRegistry
from .base import BoundPropagator


class BackwardLBPPropagator(BoundPropagator):
    """Backward-mode linear bound propagation (CROWN).

    Builds a BackwardTape during a forward graph walk, then runs the
    tape's backward algorithm at the output to obtain LinearBounds.

    When ``alpha_config.enabled`` is ``True``, runs an outer
    projected-gradient-descent loop that optimizes every alpha-capable
    relaxation's free parameter (ReLU crossing slope, sigmoid tangent
    point, McCormick eta, etc.) jointly against a loss computed from the
    final output bounds. Two scopes are supported:

    - ``optimize_intermediate=False``: final-only. Intermediate CROWN
      concretizations run under ``torch.no_grad``; only the outer backward
      pass tracks gradients.
    - ``optimize_intermediate=True``: intermediate. Every recursive
      ``concretize_at`` call also flows gradients through the alpha
      parameters it consults, so the knobs at every layer couple.

    Args:
        graph_module: The traced fx.GraphModule with metadata annotations.
        registry: Strategy registry. If None, uses the built-in default.
        alpha_config: Alpha-CROWN optimization config. When ``None`` or
            ``enabled=False``, runs the plain single-pass CROWN.
    """

    def __init__(
        self,
        graph_module: fx.GraphModule,
        registry: TargetRegistry[BackwardLBPStrategy] | None = None,
        alpha_config: AlphaOptimizationConfig | None = None,
    ) -> None:
        super().__init__(graph_module)
        self._registry = registry or create_default_backward_lbp_registry()
        self._alpha_config = alpha_config or AlphaOptimizationConfig()

    @property
    def method_name(self) -> str:
        return "backward_lbp"

    @property
    def alpha_config(self) -> AlphaOptimizationConfig:
        return self._alpha_config

    def propagate(
        self,
        input_regions: Sequence[SimpleRegion],
    ) -> Sequence[LinearBounds]:
        placeholders = self._placeholder_nodes()
        if len(input_regions) != len(placeholders):
            raise ValueError(f"Expected {len(placeholders)} input regions, got {len(input_regions)}")

        regions_list = list(input_regions)

        if not self._alpha_config.enabled:
            return self._propagate_once(regions_list, NullAlphaProvider())

        return run_alpha_optimization(
            propagate_once=lambda provider: self._propagate_once(regions_list, provider),
            config=self._alpha_config,
        )

    def _propagate_once(
        self,
        input_regions: list[SimpleRegion],
        alpha_provider: AlphaProvider,
    ) -> Sequence[LinearBounds]:
        """Single forward walk + backward tape pass with the given provider.

        Separated out so the alpha-optimization loop can call it repeatedly
        with a learnable :class:`AutoRegisteringAlphaProvider`.
        """
        tape = BackwardTape(self._graph_module, input_regions)
        tape.alpha_provider = alpha_provider
        # Final-only mode wraps intermediate CROWN concretizations in no_grad
        # so the outer optimizer only sees the final-layer dependence on alphas.
        no_grad_concretizations = self._alpha_config.enabled and not self._alpha_config.optimize_intermediate
        provider = CrownBoundsProvider(tape, no_grad_concretizations=no_grad_concretizations)

        # Forward walk: build tape
        for node in self._graph_module.graph.nodes:
            if node.op == "placeholder":
                continue
            elif node.op == "get_attr":
                tape.record_concrete(node, tape.fetch_attr(node.target))
            elif node.op in ("call_function", "call_method", "call_module"):
                self._propagate_operation(node, tape, provider)
            elif node.op == "output":
                return self._handle_output(node, tape)

        raise RuntimeError("Graph has no output node")

    @property
    def registry(self) -> TargetRegistry[BackwardLBPStrategy]:
        return self._registry

    def _propagate_operation(self, node: fx.Node, tape: BackwardTape, provider: CrownBoundsProvider) -> None:
        """Build relaxation or evaluate concretely."""
        is_abstract = node.meta.get("is_abstract", True)

        if not is_abstract:
            tape.record_concrete(node, self._evaluate_concrete(node, tape))
            return

        strategy = self.registry.get_strategy(node, self._graph_module)
        if not isinstance(strategy, BackwardLBPStrategy):
            raise TypeError(f"Expected BackwardLBPStrategy for node {node.name!r}, got {type(strategy).__name__}")
        relaxation = strategy.build_relaxation(node, tape, provider)
        tape.record(node, relaxation)

    def _handle_output(self, node: fx.Node, tape: BackwardTape) -> list[LinearBounds]:
        """Backward from each output node using the tape."""
        args = node.args[0] if isinstance(node.args[0], (tuple, list)) else node.args

        results: list[LinearBounds] = []
        for output_arg in args:
            if not isinstance(output_arg, fx.Node):
                raise TypeError(f"Expected output to be an fx.Node, got {type(output_arg)}")
            results.append(tape.backward_from(output_arg, batch_ndim=0))

        return results

    @staticmethod
    def _evaluate_concrete(node: fx.Node, tape: BackwardTape) -> torch.Tensor:
        """Concretely evaluate a non-abstract node."""
        args, kwargs = tape.resolve_args(node)
        target = node.target
        if node.op == "call_function":
            return target(*args, **kwargs)
        if node.op == "call_method":
            return getattr(args[0], target)(*args[1:], **kwargs)
        if node.op == "call_module":
            module = tape.get_module(target)
            return module(*args, **kwargs)
        raise ValueError(f"Cannot evaluate node op={node.op!r}")
