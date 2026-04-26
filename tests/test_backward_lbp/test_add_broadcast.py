"""Regression: ``AddRelaxation`` / ``SubRelaxation`` must reduce the
upstream ``A`` across broadcast axes when ``y = a + b`` (or ``y = a - b``)
involves operands of different shape.

Before the fix, the backward strategies passed ``A`` through unchanged to
both predecessors. When one predecessor was scalar-shaped (e.g. via
``squeeze(-1)``) and the other was vector-shaped, the squeeze branch's
backward step would re-introduce the squeezed axis, leaving an extra
``(1,)`` in ``input_shape``. Accumulating that into the placeholder's A
hit ``Cannot add operators with different input shapes: (1,) vs (1, 1)``.

The fix sums A across broadcast axes per predecessor (mirror of pytorch's
broadcast forward semantics).
"""

from __future__ import annotations

import torch
from torch import nn

from bound_propagation import BoundModel, HyperRectangle


def _propagate(fn, *, lo: float, hi: float, method: str):
    dummy = torch.zeros(1)
    model = BoundModel(fn, dummy_inputs=(dummy,), method=method)
    region = HyperRectangle(lower=torch.tensor([lo]), upper=torch.tensor([hi]))
    return model.propagate(region)


def _assert_sound(fn, *, lo: float, hi: float, method: str, n: int = 30) -> None:
    bounds = _propagate(fn, lo=lo, hi=hi, method=method).concretize()
    for x in torch.linspace(lo, hi, n):
        with torch.no_grad():
            y = fn(x.reshape(1))
        assert torch.all(bounds.lower <= y + 1e-5), f"{method} lower violated at x={x}: {bounds.lower} > {y}"
        assert torch.all(y <= bounds.upper + 1e-5), f"{method} upper violated at x={x}: {y} > {bounds.upper}"


class _NetThenSqueezeAdd(nn.Module):
    """``y = sin(x) + l2(relu(l1(x))).squeeze(-1)`` — the original failing pattern."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.l1 = nn.Linear(1, 4)
        self.l2 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = self.l2(torch.relu(self.l1(x))).squeeze(-1)
        return (a + b).reshape(1)


class TestAddBroadcastBackwardLBP:
    """``add`` with one scalar branch + one vector branch."""

    def test_squeeze_then_add_backward_lbp_succeeds(self) -> None:
        model = _NetThenSqueezeAdd()
        model.eval()
        _assert_sound(model, lo=-1.0, hi=1.0, method="backward_lbp")

    def test_squeeze_then_add_forward_backward_lbp_succeeds(self) -> None:
        model = _NetThenSqueezeAdd()
        model.eval()
        _assert_sound(model, lo=-1.0, hi=1.0, method="forward_backward_lbp")

    def test_squeeze_then_add_crown_ibp_succeeds(self) -> None:
        model = _NetThenSqueezeAdd()
        model.eval()
        _assert_sound(model, lo=-1.0, hi=1.0, method="crown_ibp")

    def test_scalar_plus_vector_function(self) -> None:
        """A pure functional version exercising the same broadcast pattern."""

        def fn(x: torch.Tensor) -> torch.Tensor:
            scalar = torch.sin(x).squeeze(-1)  # shape ()
            vector = torch.relu(x)  # shape (1,)
            return (scalar + vector).reshape(1)

        for method in ("backward_lbp", "forward_backward_lbp", "crown_ibp"):
            _assert_sound(fn, lo=-1.0, hi=1.0, method=method)

    def test_sub_with_broadcast_branches(self) -> None:
        """``y = a - b`` with one scalar branch and one vector branch."""

        def fn(x: torch.Tensor) -> torch.Tensor:
            scalar = torch.sigmoid(x).squeeze(-1)
            vector = torch.tanh(x)
            return (vector - scalar).reshape(1)

        for method in ("backward_lbp", "forward_backward_lbp", "crown_ibp"):
            _assert_sound(fn, lo=-1.0, hi=1.0, method=method)

    def test_left_scalar_right_vector_sub(self) -> None:
        """The scalar on the left side (negated branch in ``y = a - b``)."""

        def fn(x: torch.Tensor) -> torch.Tensor:
            scalar = torch.sigmoid(x).squeeze(-1)
            vector = torch.tanh(x)
            return (scalar - vector).reshape(1)

        for method in ("backward_lbp", "forward_backward_lbp", "crown_ibp"):
            _assert_sound(fn, lo=-1.0, hi=1.0, method=method)
