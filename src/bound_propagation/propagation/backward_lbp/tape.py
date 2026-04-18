"""Backward tape (Wengert list) for tape-based backward LBP (CROWN).

The tape records relaxations during a forward pass, then runs the backward
algorithm via BFS with A-matrix accumulation to produce LinearBounds.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import torch
import torch.fx as fx

from ...bounds import IntervalBounds, LinearBounds
from ...regions import SimpleRegion
from .base import BackwardRelaxation


class _InputMarker(BackwardRelaxation):
    """Marker stored for placeholder nodes. Never backward_through'd."""

    def predecessor_nodes(self) -> list[fx.Node]:
        return []

    def backward_through(self, A_lower, A_upper, batch_ndim):
        raise RuntimeError("_InputMarker.backward_through should never be called")


class BackwardTape:
    """Wengert list (tape) for backward LBP.

    Manages relaxation storage, argument resolution, and the backward
    BFS algorithm that produces LinearBounds.

    Parameters
    ----------
    graph_module : fx.GraphModule
        The traced fx.GraphModule.
    input_regions : list[SimpleRegion]
        Sequence of SimpleRegion, one per placeholder.
    """

    def __init__(
        self,
        graph_module: fx.GraphModule,
        input_regions: list[SimpleRegion],
    ) -> None:
        self._graph_module = graph_module
        self._store: dict[str, Any] = {}
        self._relaxations: dict[str, BackwardRelaxation] = {}
        self._input_regions: dict[str, SimpleRegion] = {}
        self._input_ids: dict[str, int] = {}

        # Seed placeholders
        placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]
        if len(input_regions) != len(placeholders):
            raise ValueError(f"Expected {len(placeholders)} input regions, got {len(input_regions)}")

        for i, (ph, region) in enumerate(zip(placeholders, input_regions, strict=True)):
            marker = _InputMarker()
            self._store[ph.name] = marker
            self._relaxations[ph.name] = marker
            self._input_regions[ph.name] = region
            self._input_ids[ph.name] = i

        # Cache for concretized interval bounds
        self._interval_cache: dict[str, IntervalBounds] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, node: fx.Node, relaxation: BackwardRelaxation) -> None:
        """Record a relaxation for an abstract node."""
        self._store[node.name] = relaxation
        self._relaxations[node.name] = relaxation

    def record_concrete(self, node: fx.Node, value: Any) -> None:
        """Record a concrete value for a non-abstract node."""
        self._store[node.name] = value

    # ------------------------------------------------------------------
    # Resolution (like PropagationContext)
    # ------------------------------------------------------------------

    def resolve(self, arg: Any) -> Any:
        """Resolve a single argument.

        fx.Node -> stored value; literals -> passthrough.
        """
        if isinstance(arg, fx.Node):
            if arg.name not in self._store:
                raise KeyError(f"Node '{arg.name}' has no stored value (not yet recorded?)")
            return self._store[arg.name]
        if isinstance(arg, (tuple, list)):
            resolved = [self.resolve(a) for a in arg]
            return type(arg)(resolved)
        return arg

    def resolve_args(self, node: fx.Node) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Resolve all positional and keyword arguments for *node*."""
        args = tuple(self.resolve(a) for a in node.args)
        kwargs = {k: self.resolve(v) for k, v in node.kwargs.items()}
        return args, kwargs

    def get_module(self, target: str) -> torch.nn.Module:
        """Retrieve a sub-module from the GraphModule by qualified name."""
        return self._graph_module.get_submodule(target)

    def fetch_attr(self, target: str) -> Any:
        """Walk the module hierarchy to fetch a parameter, buffer, or constant."""
        atoms = target.split(".")
        obj: Any = self._graph_module
        for atom in atoms:
            obj = getattr(obj, atom)
        return obj

    # ------------------------------------------------------------------
    # Backward algorithm
    # ------------------------------------------------------------------

    def backward_from(self, node: fx.Node, batch_ndim: int) -> LinearBounds:
        """Run backward LBP from *node* to inputs.

        Builds the backward subgraph, computes pending counts, then
        runs BFS with A-matrix accumulation and bias threading.

        Parameters
        ----------
        node : fx.Node
            The node to backward-propagate from.
        batch_ndim : int
            Number of leading batch dimensions.

        Returns
        -------
        LinearBounds
            LinearBounds with linear terms for each input placeholder.
        """
        subgraph = self._backward_subgraph(node)
        pending = self._compute_pending(subgraph)

        # Build identity A-matrix for the start node
        shape = node.meta["tensor_meta"]["shape"]
        feature_shape = shape[batch_ndim:]
        numel = math.prod(feature_shape)
        dtype = node.meta["tensor_meta"]["dtype"]
        device = self._infer_device()

        batch_ones = (1,) * batch_ndim
        identity_shape = (*batch_ones, *feature_shape, *feature_shape) if feature_shape else (*batch_ones, 1)
        identity = torch.eye(numel, dtype=dtype, device=device).reshape(identity_shape)

        # Initialize accumulated A-matrices and bias
        accumulated_A: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            node.name: (identity, identity),
        }
        bias_shape = (*batch_ones, *feature_shape)
        bias_lower = torch.zeros(bias_shape, dtype=dtype, device=device)
        bias_upper = torch.zeros(bias_shape, dtype=dtype, device=device)

        # Collect input A-matrices
        input_A: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # BFS backward
        queue: deque[fx.Node] = deque([node])

        while queue:
            current = queue.popleft()
            A_lower, A_upper = accumulated_A.pop(current.name)

            # Check if this is a placeholder (input node)
            if current.name in self._input_regions:
                input_A[current.name] = (A_lower, A_upper)
                continue

            # Get relaxation and backward through it
            if current.name not in self._relaxations:
                raise RuntimeError(f"Node '{current.name}' is in the backward subgraph but has no relaxation")

            relaxation = self._relaxations[current.name]
            contributions = relaxation.backward_through(A_lower, A_upper, batch_ndim)

            # Accumulate bias
            bias_lower = bias_lower + contributions.bias_lower
            bias_upper = bias_upper + contributions.bias_upper

            # Distribute A-matrices to predecessors
            for pred_node, (delta_A_l, delta_A_u) in contributions.a_terms.items():
                pred_name = pred_node.name
                if pred_name in accumulated_A:
                    old_l, old_u = accumulated_A[pred_name]
                    accumulated_A[pred_name] = (old_l + delta_A_l, old_u + delta_A_u)
                else:
                    accumulated_A[pred_name] = (delta_A_l, delta_A_u)

                pending[pred_name] -= 1
                if pending[pred_name] == 0:
                    queue.append(pred_node)

        # Collect input A-matrices in placeholder order
        regions: list[SimpleRegion] = []
        linear_lower: list[torch.Tensor] = []
        linear_upper: list[torch.Tensor] = []
        input_ids: list[int] = []

        for ph_node in self._placeholder_nodes():
            if ph_node.name in input_A:
                A_l, A_u = input_A[ph_node.name]
                regions.append(self._input_regions[ph_node.name])
                linear_lower.append(A_l)
                linear_upper.append(A_u)
                input_ids.append(self._input_ids[ph_node.name])

        return LinearBounds(
            regions=regions or None,
            linear_lower=linear_lower or None,
            bias_lower=bias_lower,
            linear_upper=linear_upper or None,
            bias_upper=bias_upper,
            input_ids=input_ids or None,
        )

    def concretize_at(self, node: fx.Node, batch_ndim: int = 0) -> IntervalBounds:
        """Backward-concretize at *node* and cache the result.

        Runs backward_from + concretize. Results are cached so repeated
        concretizations of the same node are free.

        Parameters
        ----------
        node : fx.Node
            The node to concretize.
        batch_ndim : int
            Number of leading batch dimensions.

        Returns
        -------
        IntervalBounds
            Cached IntervalBounds for the node.
        """
        if node.name in self._interval_cache:
            return self._interval_cache[node.name]

        linear_bounds = self.backward_from(node, batch_ndim)
        result = linear_bounds.concretize()
        self._interval_cache[node.name] = result
        return result

    # ------------------------------------------------------------------
    # Subgraph helpers
    # ------------------------------------------------------------------

    def _backward_subgraph(self, start: fx.Node) -> set[fx.Node]:
        """BFS backward following relaxation.predecessor_nodes(), not fx.Node.args.

        This avoids deadlocks from chain-breaking ops whose fx.Node.args
        still point to predecessor nodes even though the relaxation has
        no predecessors (e.g. IntervalLeafRelaxation).
        """
        visited: set[fx.Node] = set()
        queue: deque[fx.Node] = deque([start])
        visited.add(start)

        while queue:
            current = queue.popleft()

            # Placeholders have no further predecessors
            if current.name in self._input_regions:
                continue

            # Concrete-only nodes have no relaxation
            if current.name not in self._relaxations:
                continue

            relaxation = self._relaxations[current.name]
            for pred in relaxation.predecessor_nodes():
                if pred not in visited:
                    visited.add(pred)
                    queue.append(pred)

        return visited

    def _compute_pending(self, subgraph: set[fx.Node]) -> dict[str, int]:
        """Count incoming edges per node within the subgraph.

        For each node in the subgraph, pending[node.name] = number of
        nodes in the subgraph that list it as a predecessor.
        """
        pending: dict[str, int] = {node.name: 0 for node in subgraph}

        for node in subgraph:
            if node.name in self._input_regions:
                continue
            if node.name not in self._relaxations:
                continue

            relaxation = self._relaxations[node.name]
            for pred in relaxation.predecessor_nodes():
                if pred in subgraph:
                    pending[pred.name] += 1

        return pending

    def _placeholder_nodes(self) -> list[fx.Node]:
        """Return placeholder nodes in graph order."""
        return [n for n in self._graph_module.graph.nodes if n.op == "placeholder"]

    def _infer_device(self) -> torch.device:
        """Infer device from model parameters or default to CPU."""
        for param in self._graph_module.parameters():
            return param.device
        return torch.device("cpu")
