"""Tensor-constructor functions that produce constant tensors.

These functions (``torch.zeros``, ``torch.ones``, ``torch.full`` and the
``*_like`` variants) do not depend on the *values* of any abstract
(bounded) argument, only on shape/dtype/device. The metadata pass marks
them ``is_abstract=False`` so propagators evaluate them concretely.

The registry treats them as supported without requiring a strategy:
they are handled entirely by the propagator's concrete-evaluation path
via :func:`evaluate_constant_producer`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx

CONSTANT_PRODUCING_TARGETS: frozenset[Callable[..., Any]] = frozenset(
    {
        torch.zeros,
        torch.zeros_like,
        torch.ones,
        torch.ones_like,
        torch.full,
        torch.full_like,
        torch.empty,
        torch.empty_like,
        torch.eye,
        torch.arange,
        torch.linspace,
    }
)


def is_constant_producing(target: Any) -> bool:
    """Return ``True`` if *target* is a tensor-constructor treated as constant."""
    return target in CONSTANT_PRODUCING_TARGETS


def evaluate_constant_producer(node: fx.Node) -> torch.Tensor:
    """Evaluate a :data:`CONSTANT_PRODUCING_TARGETS` node concretely.

    Substitutes any ``fx.Node`` argument with a zero tensor of its
    annotated shape and dtype (from :class:`MetadataPass`), then calls
    ``node.target``. The result depends only on shape/dtype of the
    template, matching the value-independent semantics of tensor
    constructors like ``torch.zeros_like``.
    """
    if node.op != "call_function":
        raise ValueError(f"evaluate_constant_producer expects a call_function node, got {node.op!r}")
    if node.target not in CONSTANT_PRODUCING_TARGETS:
        raise ValueError(f"Node target {node.target!r} is not a constant-producing tensor constructor")

    args = tuple(_materialize(a) for a in node.args)
    kwargs = {k: _materialize(v) for k, v in node.kwargs.items()}
    return node.target(*args, **kwargs)  # ty:ignore[call-non-callable]


def _materialize(arg: Any) -> Any:
    """Replace ``fx.Node`` references with zero templates of matching shape/dtype."""
    if isinstance(arg, fx.Node):
        meta = arg.meta.get("tensor_meta")
        if meta is None:
            raise RuntimeError(
                f"fx.Node {arg.name!r} has no tensor_meta; run MetadataPass before evaluating constant producers."
            )
        return torch.zeros(meta["shape"], dtype=meta["dtype"])
    if isinstance(arg, tuple):
        return tuple(_materialize(a) for a in arg)
    if isinstance(arg, list):
        return [_materialize(a) for a in arg]
    return arg
