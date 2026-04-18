from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.elementwise import ForwardLBPSigmoid
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate


def _make_linear_bounds(region: HyperRectangle) -> LinearBounds:
    """Create identity linear bounds from a region."""
    dim = region.lower.numel()
    return LinearBounds(
        regions=[region],
        input_ids=[0],
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

    lower_x = torch.tensor(1.0)
    upper_x = torch.tensor(2.0)
    lower_s = torch.sigmoid(lower_x)
    upper_s = torch.sigmoid(upper_x)
    secant_slope = (upper_s - lower_s) / (upper_x - lower_x)
    midpoint = (lower_x + upper_x) / 2.0
    midpoint_s = torch.sigmoid(midpoint)
    midpoint_prime = midpoint_s * (1.0 - midpoint_s)

    assert torch.allclose(result.linear_lower, secant_slope.reshape(1, 1))
    assert torch.allclose(result.bias_lower, (lower_s - secant_slope * lower_x).reshape(1))
    assert torch.allclose(result.linear_upper, midpoint_prime.reshape(1, 1))
    assert torch.allclose(result.bias_upper, (midpoint_s - midpoint_prime * midpoint).reshape(1))

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

    lower_x = torch.tensor(-1.0)
    upper_x = torch.tensor(1.0)
    lower_s = torch.sigmoid(lower_x)
    upper_s = torch.sigmoid(upper_x)
    lower_prime = lower_s * (1.0 - lower_s)
    upper_prime = upper_s * (1.0 - upper_s)

    assert torch.allclose(result.linear_lower, lower_prime.reshape(1, 1))
    assert torch.allclose(result.bias_lower, (lower_s - lower_prime * lower_x).reshape(1))
    assert torch.allclose(result.linear_upper, upper_prime.reshape(1, 1))
    assert torch.allclose(result.bias_upper, (upper_s - upper_prime * upper_x).reshape(1))

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

    assert torch.allclose(result.linear_lower, torch.tensor([[0.0]]), atol=1e-6)
    assert torch.allclose(result.bias_lower, torch.tensor([0.5]), atol=1e-6)
    assert torch.allclose(result.linear_upper, torch.tensor([[0.0]]), atol=1e-6)
    assert torch.allclose(result.bias_upper, torch.tensor([0.5]), atol=1e-6)

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
