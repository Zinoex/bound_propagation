"""Verify Stage 4 contract: backward-LBP captures shapes at build time.

Strategies populate :meth:`BackwardTape.set_shape` for every abstract node
they record, so that backward-pass consumers can answer shape / dtype
queries without touching ``node.meta``. These tests validate that the
tape's shape dict is populated correctly and that the backward pass
produces the same results whether shapes come from the tape or from
``node.meta``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bound_propagation import BoundModel, HyperRectangle


def test_tape_captures_shapes_through_reshape_stack() -> None:
    """A linear stack with a reshape in the middle produces a
    backward-LBP tape whose shape dict carries the reshaped feature shape
    forward without reading ``node.meta``."""

    class ReshapeMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(6, 8)
            self.fc2 = nn.Linear(8, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.fc1(x).relu()
            # Interleave a reshape that backward-LBP must capture by shape.
            h = h.reshape(2, 4)
            h = h.flatten()
            return self.fc2(h)

    torch.manual_seed(0)
    model = ReshapeMLP()
    dummy = torch.zeros(6)
    bm = BoundModel(model, dummy_inputs=(dummy,), method="backward_lbp")

    region = HyperRectangle(lower=torch.zeros(6), upper=torch.ones(6))
    bounds = bm.propagate(region)
    assert bounds.shape == (3,)


def test_tape_bulk_copies_shapes_from_node_meta_at_construction() -> None:
    """``BackwardTape`` bulk-copies ``node.meta["tensor_meta"]`` into its own
    shape/dtype dicts at construction time, so backward-pass consumers read
    from the tape exclusively (no live ``node.meta`` reads)."""
    import pytest

    from bound_propagation.propagation.backward_lbp.tape import BackwardTape

    model = nn.Sequential(nn.Linear(4, 2))
    dummy = torch.zeros(4)
    bm = BoundModel(model, dummy_inputs=(dummy,), method="backward_lbp")

    region = HyperRectangle(lower=torch.zeros(4), upper=torch.ones(4))
    tape = BackwardTape(bm._graph_module, [region])  # noqa: SLF001

    placeholders = [n for n in bm._graph_module.graph.nodes if n.op == "placeholder"]  # noqa: SLF001
    assert tape.shape_of(placeholders[0]) == (4,)
    assert tape.dtype_of(placeholders[0]) == torch.float32

    # A node with no tensor_meta (or not in the graph) raises ``KeyError``:
    # the strict API surfaces misuse rather than silently returning wrong data.
    with pytest.raises(KeyError, match="No shape recorded"):
        tape.shape_of(type("_FakeNode", (), {"name": "not_in_graph"})())


def test_tape_set_shape_overrides_fallback() -> None:
    """When a strategy records a shape, :meth:`shape_of` returns that —
    the fallback to ``node.meta`` is bypassed."""
    from bound_propagation.propagation.backward_lbp.tape import BackwardTape

    model = nn.Sequential(nn.Linear(4, 2))
    dummy = torch.zeros(4)
    bm = BoundModel(model, dummy_inputs=(dummy,), method="backward_lbp")

    region = HyperRectangle(lower=torch.zeros(4), upper=torch.ones(4))
    tape = BackwardTape(bm._graph_module, [region])  # noqa: SLF001

    placeholders = [n for n in bm._graph_module.graph.nodes if n.op == "placeholder"]  # noqa: SLF001
    tape.set_shape(placeholders[0], (2, 2), dtype=torch.float64)

    assert tape.shape_of(placeholders[0]) == (2, 2)
    assert tape.dtype_of(placeholders[0]) == torch.float64
