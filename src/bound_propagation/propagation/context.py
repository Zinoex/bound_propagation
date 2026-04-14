"""Propagation context for bound propagation over torch.fx graphs.

PropagationContext manages bounds storage, argument resolution, and
refcount-based memory management during graph traversal.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.fx as fx

from ..bounds import AbstractBounds


class PropagationContext:
    """Context for propagating bounds through a torch.fx graph.

    Manages:
    - A bounds store keyed by fx.Node SSA names
    - Resolution of fx.Node args/kwargs to stored values or literals
    - Module/attribute lookup on the GraphModule
    - Refcount-based cleanup of intermediate bounds

    Args:
        graph_module: The traced torch.fx.GraphModule being propagated.
    """

    def __init__(self, graph_module: fx.GraphModule) -> None:
        self._graph_module = graph_module
        self._store: dict[str, AbstractBounds | torch.Tensor | torch.types.Number] = {}
        self._refcounts: dict[str, int] = {}
        self._init_refcounts()

    # ------------------------------------------------------------------
    # Refcount initialization
    # ------------------------------------------------------------------

    def _init_refcounts(self) -> None:
        """Count actual arg/kwarg references for each node.

        We count every occurrence of a node in the args/kwargs of its
        consumers, not just ``len(node.users)``, because a single consumer
        can reference a value multiple times (e.g. ``x + x``).
        """
        counts: dict[str, int] = {}

        for node in self._graph_module.graph.nodes:
            counts.setdefault(node.name, 0)

        for node in self._graph_module.graph.nodes:
            self._count_refs_in(node.args, counts)
            self._count_refs_in(tuple(node.kwargs.values()), counts)

        self._refcounts = counts

    @staticmethod
    def _count_refs_in(args: tuple[Any, ...], counts: dict[str, int]) -> None:
        """Recursively count fx.Node references in an args structure."""
        for arg in args:
            if isinstance(arg, fx.Node):
                counts[arg.name] = counts.get(arg.name, 0) + 1
            elif isinstance(arg, (tuple, list)):
                PropagationContext._count_refs_in(tuple(arg), counts)

    # ------------------------------------------------------------------
    # Store / resolve
    # ------------------------------------------------------------------

    def store(self, node: fx.Node, value: AbstractBounds | torch.Tensor | torch.types.Number) -> None:
        """Store a bound or concrete value for *node*."""
        self._store[node.name] = value

    def resolve(self, arg: Any) -> Any:
        """Resolve a single argument.

        - ``fx.Node`` → look up stored value (raises if missing)
        - Literal (int, float, NoneType, slice, etc.) → returned as-is
        - ``tuple``/``list`` of the above → recursively resolved
        """
        if isinstance(arg, fx.Node):
            if arg.name not in self._store:
                raise KeyError(f"Node '{arg.name}' has no stored value (not yet propagated?)")
            return self._store[arg.name]
        if isinstance(arg, (tuple, list)):
            resolved = [self.resolve(a) for a in arg]
            return type(arg)(resolved)
        # Literal passthrough (int, float, None, slice, bool, torch.dtype, ...)
        return arg

    def resolve_args(self, node: fx.Node) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Resolve all positional and keyword arguments for *node*.

        Returns:
            ``(args, kwargs)`` with all ``fx.Node`` references replaced by
            their stored values and literals passed through unchanged.
        """
        args = tuple(self.resolve(a) for a in node.args)
        kwargs = {k: self.resolve(v) for k, v in node.kwargs.items()}
        return args, kwargs

    # ------------------------------------------------------------------
    # Module / attribute access
    # ------------------------------------------------------------------

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

    @property
    def graph_module(self) -> fx.GraphModule:
        """The underlying ``fx.GraphModule``."""
        return self._graph_module

    # ------------------------------------------------------------------
    # Refcount-based cleanup
    # ------------------------------------------------------------------

    def release(self, node: fx.Node) -> None:
        """Decrement refcounts for *node*'s inputs and free when zero.

        Call this after processing *node* to allow upstream values
        whose last consumer was *node* to be garbage-collected.
        """
        self._decrement_refs_in(node.args)
        self._decrement_refs_in(tuple(node.kwargs.values()))

    def _decrement_refs_in(self, args: tuple[Any, ...]) -> None:
        for arg in args:
            if isinstance(arg, fx.Node):
                self._refcounts[arg.name] -= 1
                if self._refcounts[arg.name] <= 0:
                    self._store.pop(arg.name, None)
            elif isinstance(arg, (tuple, list)):
                self._decrement_refs_in(tuple(arg))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __contains__(self, node: fx.Node) -> bool:
        return node.name in self._store

    def __getitem__(self, node: fx.Node) -> AbstractBounds | torch.Tensor | torch.types.Number:
        return self._store[node.name]
