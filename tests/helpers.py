"""Test helpers for calling strategies with the new fx.Node + PropagationContext API."""

from __future__ import annotations

from typing import Any

import torch
import torch.fx as fx

from bound_propagation.bounds import AbstractBounds
from bound_propagation.propagation.context import PropagationContext
from bound_propagation.propagation.strategy import ForwardBoundingStrategy


def propagate(strategy: ForwardBoundingStrategy[Any], *inputs: Any, **kwargs: Any) -> Any:
    """Call a strategy using a minimal fx graph and PropagationContext.

    Each input becomes a placeholder node whose value is stored in the
    context.  Lists/tuples of inputs are recursively handled so that
    ``torch.cat([a, b], dim=0)`` style signatures work correctly.

    Any keyword arguments are forwarded as ``node.kwargs``.

    Examples::

        # Unary: relu(bounds)
        result = propagate(IBPRelu(), bounds)

        # Binary: add(left, right)
        result = propagate(IBPAdd(), left, right)

        # With kwargs: sum(bounds, dim=1, keepdim=True)
        result = propagate(IBPSum(), bounds, dim=1, keepdim=True)

        # List input: cat([a, b], dim=0)
        result = propagate(ForwardLBPConcat(), [a, b], dim=0)
    """
    graph = fx.Graph()
    store_map: dict[fx.Node, Any] = {}
    counter = 0

    def _make_arg(inp: Any) -> Any:
        nonlocal counter
        if isinstance(inp, (list, tuple)):
            return type(inp)(_make_arg(item) for item in inp)
        ph = graph.placeholder(f"input_{counter}")
        counter += 1
        store_map[ph] = inp
        return ph

    node_args = tuple(_make_arg(inp) for inp in inputs)

    op = graph.call_function(torch.ops.aten.add.Tensor, args=node_args, kwargs=kwargs)
    graph.output(op)

    gm = fx.GraphModule(torch.nn.Module(), graph)
    ctx = PropagationContext(gm)

    for ph, value in store_map.items():
        ctx.store(ph, value)

    return strategy.propagate_forward(op, ctx)
