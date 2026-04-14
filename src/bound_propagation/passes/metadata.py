"""Interpreter-based metadata pass for torch.fx graphs.

Annotates each node's ``node.meta`` with shape, dtype, and
``is_abstract`` (whether bounds must be propagated through this node).
Runs once before bound propagation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.fx as fx


class MetadataPass(fx.Interpreter):
    """Annotate fx graph nodes with tensor metadata and abstractness.

    After ``run(*example_inputs, abstract_mask=...)``, every node in the
    graph will have its ``node.meta`` populated with:

    - ``"tensor_meta"``: a dict with ``shape`` (``tuple[int, ...]``) and
      ``dtype`` (``torch.dtype``).  Only present for tensor-valued nodes.
    - ``"is_abstract"``: ``True`` for nodes whose value depends on an
      abstract (bounded) input, ``False`` for constant/parameter paths.

    ``abstract_mask`` is a sequence of bools, one per placeholder, that
    indicates which inputs are abstract.  If omitted every input is
    treated as abstract.

    Example::

        gm = torch.fx.symbolic_trace(my_fn)
        meta = MetadataPass(gm)
        meta.run(example_x, example_y, abstract_mask=[True, True])
    """

    def __init__(self, module: fx.GraphModule) -> None:
        super().__init__(module)
        self._abstract_mask: list[bool] | None = None
        self._placeholder_idx: int = 0

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(
        self,
        *args: Any,
        abstract_mask: list[bool] | None = None,
    ) -> Any:
        """Execute the pass with *example_inputs*.

        Args:
            *args: Concrete example tensors for each placeholder.
            abstract_mask: Per-placeholder flag indicating abstract inputs.
                If ``None``, all inputs are abstract.
        """
        self._abstract_mask = abstract_mask
        self._placeholder_idx = 0
        return super().run(*args)

    # ------------------------------------------------------------------
    # Per-op hooks
    # ------------------------------------------------------------------

    def run_node(self, n: fx.Node) -> Any:
        """Execute one node and record metadata."""
        result = super().run_node(n)
        self._annotate_tensor_meta(n, result)
        self._annotate_is_abstract(n)
        return result

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate_tensor_meta(node: fx.Node, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            node.meta["tensor_meta"] = {
                "shape": tuple(value.shape),
                "dtype": value.dtype,
            }

    def _annotate_is_abstract(self, node: fx.Node) -> None:
        if node.op == "placeholder":
            if self._abstract_mask is not None:
                node.meta["is_abstract"] = self._abstract_mask[self._placeholder_idx]
            else:
                node.meta["is_abstract"] = True
            self._placeholder_idx += 1

        elif node.op == "get_attr":
            node.meta["is_abstract"] = False

        elif node.op in ("call_function", "call_method", "call_module"):
            # Abstract if *any* input is abstract
            node.meta["is_abstract"] = self._any_input_abstract(node)

        elif node.op == "output":
            node.meta["is_abstract"] = self._any_input_abstract(node)

        else:
            node.meta["is_abstract"] = False

    def _any_input_abstract(self, node: fx.Node) -> bool:
        """Return True if any fx.Node argument of *node* is abstract."""
        return self._check_abstract_in(node.args) or self._check_abstract_in(tuple(node.kwargs.values()))

    def _check_abstract_in(self, args: tuple[Any, ...]) -> bool:
        for arg in args:
            if isinstance(arg, fx.Node):
                if arg.meta.get("is_abstract", False):
                    return True
            elif isinstance(arg, (tuple, list)):
                if self._check_abstract_in(tuple(arg)):
                    return True
        return False
