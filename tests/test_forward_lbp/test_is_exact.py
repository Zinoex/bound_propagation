"""End-to-end ``is_exact`` propagation through forward LBP.

Verifies the phase 5 contract: a stack of pure-affine layers (``nn.Linear``)
produces a ``LinearBounds`` with ``is_exact == True``; inserting a nonlinear
layer (``nn.ReLU``) breaks exactness.

Together with the property-level tests in ``test_bounds/test_linear_bounds.py``
this confirms the ``ForwardLBPLinear`` affine fast path threads the shared
operator and bias tensor through the chain.
"""

from __future__ import annotations

import torch
from torch import nn

from bound_propagation import BoundModel, HyperRectangle
from bound_propagation.bounds import LinearBounds


def _propagate(model: nn.Module, *, in_features: int = 4) -> LinearBounds:
    dummy = torch.zeros(in_features)
    bound_model = BoundModel(model, dummy_inputs=(dummy,), method="forward_lbp")
    region = HyperRectangle(
        lower=torch.full((in_features,), -0.5),
        upper=torch.full((in_features,), 0.5),
    )
    bounds = bound_model.propagate(region)
    assert isinstance(bounds, LinearBounds)
    return bounds


class TestForwardLBPIsExactPropagation:
    def test_single_linear_is_exact(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 3)
        assert _propagate(model).is_exact is True

    def test_three_linear_stack_is_exact(self) -> None:
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(4, 5), nn.Linear(5, 6), nn.Linear(6, 2))
        assert _propagate(model).is_exact is True

    def test_relu_inserted_breaks_exactness(self) -> None:
        torch.manual_seed(2)
        model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2))
        # The intermediate Linear output crosses zero for some neurons under
        # the centered region [-0.5, 0.5], so ReLU's relaxation is non-trivial.
        assert _propagate(model).is_exact is False

    def test_relu_then_linear_is_not_exact(self) -> None:
        torch.manual_seed(3)
        model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 6), nn.Linear(6, 2))
        assert _propagate(model).is_exact is False

    def test_exact_bound_concretizes_to_zero_width_when_inputs_pinned(self) -> None:
        """An exact bound on a pinned input region should have zero-width concretization."""
        torch.manual_seed(4)
        model = nn.Linear(3, 2)
        dummy = torch.zeros(3)
        bound_model = BoundModel(model, dummy_inputs=(dummy,), method="forward_lbp")
        # Degenerate region: lower == upper (a single point).
        point = torch.tensor([0.1, -0.2, 0.3])
        region = HyperRectangle(lower=point, upper=point)
        bounds = bound_model.propagate(region)
        assert bounds.is_exact is True
        concrete = bounds.concretize()
        assert torch.allclose(concrete.lower, concrete.upper)
        # And matches the deterministic forward pass.
        with torch.no_grad():
            expected = model(point)
        assert torch.allclose(concrete.lower, expected)
