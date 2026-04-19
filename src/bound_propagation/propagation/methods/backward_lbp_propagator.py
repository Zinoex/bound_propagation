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
from ..backward_lbp import create_default_backward_lbp_registry
from ..backward_lbp.base import BackwardLBPStrategy, CrownBoundsProvider
from ..backward_lbp.tape import BackwardTape
from ..registry import TargetRegistry
from .base import BoundPropagator


class BackwardLBPPropagator(BoundPropagator):
    """Backward-mode linear bound propagation (CROWN).

    Builds a BackwardTape during a forward graph walk, then runs the
    tape's backward algorithm at the output to obtain LinearBounds.

    Args:
        graph_module: The traced fx.GraphModule with metadata annotations.
        registry: Strategy registry. If None, uses the built-in default.
    """

    def __init__(
        self,
        graph_module: fx.GraphModule,
        registry: TargetRegistry[BackwardLBPStrategy] | None = None,
    ) -> None:
        super().__init__(graph_module)
        self._registry = registry or create_default_backward_lbp_registry()

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

        tape = BackwardTape(self._graph_module, list(input_regions))
        provider = CrownBoundsProvider(tape)

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
