"""Regression: backward LBP through ``constant + x`` / ``c * x`` produced a
bias contribution shaped after the constant's rank, not the propagated A's
input rank. For scalar constants composed with a multi-feature x, the bias
was off-rank, then broadcast incorrectly when added into the accumulator.

This file covers ``relu(stack([x-1, x, x+1])).mean()``-style patterns where the
combination of ``stack`` (creating multi-feature axes), ``elementwise``, and
``reduction`` exposes the bug downstream of ``constant + x``.
"""

from __future__ import annotations

import torch

from bound_propagation import BoundModel, HyperRectangle


def _propagate(fn, *, lo: float, hi: float, method: str = "backward_lbp"):
    dummy = torch.zeros(1)
    model = BoundModel(fn, dummy_inputs=(dummy,), method=method)
    region = HyperRectangle(lower=torch.tensor([lo]), upper=torch.tensor([hi]))
    return model.propagate(region)


class TestScalarConstantAddBackwardShape:
    def test_relu_stack_offsets_mean(self) -> None:
        """``mean(relu([x-1, x, x+1]))`` — minimal failing case."""

        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(torch.stack([x - 1.0, x, x + 1.0])).mean()

        bounds = _propagate(fn, lo=-2.0, hi=2.0)
        # Just need to construct without raising; verify lower <= upper.
        concrete = bounds.concretize()
        assert torch.all(concrete.lower <= concrete.upper + 1e-6)

    def test_sigmoid_stack_offsets_sum(self) -> None:
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(torch.stack([x, x + 2.0, x - 2.0])).sum()

        bounds = _propagate(fn, lo=-3.0, hi=3.0)
        concrete = bounds.concretize()
        assert torch.all(concrete.lower <= concrete.upper + 1e-6)

    def test_scalar_scale_then_relu_then_sum(self) -> None:
        """Triggers the same shape bug via ``ScaleRelaxation`` (scalar ``c * x``)."""

        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(torch.stack([x * 0.5, x * 1.0, x * 2.0])).sum()

        bounds = _propagate(fn, lo=-2.0, hi=2.0)
        concrete = bounds.concretize()
        assert torch.all(concrete.lower <= concrete.upper + 1e-6)

    def test_soundness_against_samples(self) -> None:
        """Sample the input region; the bound must contain every f(x)."""

        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.relu(torch.stack([x - 1.0, x, x + 1.0])).mean()

        bounds = _propagate(fn, lo=-2.0, hi=2.0).concretize()
        torch.manual_seed(0)
        for x in torch.linspace(-2.0, 2.0, 50):
            with torch.no_grad():
                y = fn(x.reshape(1))
            assert torch.all(bounds.lower <= y + 1e-6), f"lower violated at x={x}: {bounds.lower} > {y}"
            assert torch.all(y <= bounds.upper + 1e-6), f"upper violated at x={x}: {y} > {bounds.upper}"
