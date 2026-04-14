from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.tan import ForwardLBPTan
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate


def _make_linear_bounds(region: HyperRectangle) -> LinearBounds:
    """Create identity linear bounds from a region."""
    dim = region.lower.numel()
    return LinearBounds(
        regions=[region],
        linear_lower=torch.eye(dim),
        bias_lower=torch.zeros(dim),
        linear_upper=torch.eye(dim),
        bias_upper=torch.zeros(dim),
    )


def test_tan_small_positive_interval() -> None:
    """Test tan on a small positive interval."""
    # Region: x ∈ [0.1, 0.5]
    # tan([0.1, 0.5]) ≈ [0.1003, 0.5463]
    region = HyperRectangle(lower=torch.tensor([0.1]), upper=torch.tensor([0.5]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPTan()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    assert torch.all(lower >= 0.09)
    assert torch.all(lower <= 0.11)
    assert torch.all(upper >= 0.54)
    assert torch.all(upper <= 0.56)


def test_tan_small_negative_interval() -> None:
    """Test tan on a small negative interval."""
    # Region: x ∈ [-0.5, -0.1]
    # tan([-0.5, -0.1]) ≈ [-0.5463, -0.1003]
    region = HyperRectangle(lower=torch.tensor([-0.5]), upper=torch.tensor([-0.1]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPTan()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    assert torch.all(lower >= -0.56)
    assert torch.all(lower <= -0.54)
    assert torch.all(upper >= -0.11)
    assert torch.all(upper <= -0.09)


def test_tan_at_zero() -> None:
    """Test tan at zero."""
    # Region: x ∈ [0, 0]
    # tan([0, 0]) = [0, 0]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([0.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPTan()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(upper, torch.tensor([0.0]), atol=1e-6)


def test_tan_crossing_zero() -> None:
    """Test tan on an interval crossing zero."""
    # Region: x ∈ [-0.3, 0.3]
    # tan([-0.3, 0.3]) ≈ [-0.309, 0.309]
    region = HyperRectangle(lower=torch.tensor([-0.3]), upper=torch.tensor([0.3]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPTan()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    assert torch.all(lower >= -0.32)
    assert torch.all(lower <= -0.30)
    assert torch.all(upper >= 0.30)
    assert torch.all(upper <= 0.32)
