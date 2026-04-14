from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.sigmoid import ForwardLBPSigmoid
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate


def _make_linear_bounds(region: HyperRectangle) -> LinearBounds:
    """Create identity linear bounds from a region."""
    dim = region.lower.numel()
    return LinearBounds(
        region=region,
        linear_lower=torch.eye(dim),
        bias_lower=torch.zeros(dim),
        linear_upper=torch.eye(dim),
        bias_upper=torch.zeros(dim),
    )


def test_sigmoid_positive_interval() -> None:
    """Test sigmoid on a positive interval."""
    # Region: x ∈ [1, 2]
    # sigmoid([1, 2]) ≈ [0.731, 0.881]
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([2.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSigmoid()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    # sigmoid(1) ≈ 0.731, sigmoid(2) ≈ 0.881
    assert torch.all(lower >= 0.72)
    assert torch.all(lower <= 0.74)
    assert torch.all(upper >= 0.87)
    assert torch.all(upper <= 0.90)


def test_sigmoid_negative_interval() -> None:
    """Test sigmoid on a negative interval."""
    # Region: x ∈ [-2, -1]
    # sigmoid([-2, -1]) ≈ [0.119, 0.269]
    region = HyperRectangle(lower=torch.tensor([-2.0]), upper=torch.tensor([-1.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSigmoid()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    assert torch.all(lower >= 0.10)
    assert torch.all(lower <= 0.13)
    assert torch.all(upper >= 0.26)
    assert torch.all(upper <= 0.28)


def test_sigmoid_mixed_sign_interval() -> None:
    """Test sigmoid on an interval crossing zero."""
    # Region: x ∈ [-1, 1]
    # sigmoid([-1, 1]) ≈ [0.269, 0.731]
    region = HyperRectangle(lower=torch.tensor([-1.0]), upper=torch.tensor([1.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSigmoid()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    assert torch.all(lower >= 0.26)
    assert torch.all(lower <= 0.28)
    assert torch.all(upper >= 0.72)
    assert torch.all(upper <= 0.74)


def test_sigmoid_at_zero() -> None:
    """Test sigmoid at zero."""
    # Region: x ∈ [0, 0]
    # sigmoid([0, 0]) = [0.5, 0.5]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([0.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSigmoid()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.5]), atol=1e-6)
    assert torch.allclose(upper, torch.tensor([0.5]), atol=1e-6)


def test_sigmoid_large_positive_interval() -> None:
    """Test sigmoid on a large positive interval."""
    # Region: x ∈ [3, 5]
    # sigmoid([3, 5]) ≈ [0.953, 0.993]
    region = HyperRectangle(lower=torch.tensor([3.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSigmoid()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    assert torch.all(lower >= 0.94)
    assert torch.all(lower <= 0.96)
    assert torch.all(upper >= 0.99)
    assert torch.all(upper <= 1.0)
