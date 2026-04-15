from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.tanh import ForwardLBPTanh
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


def test_tanh_positive_interval() -> None:
    """Test tanh on a positive interval."""
    # Region: x ∈ [1, 2]
    # tanh([1, 2]) ≈ [0.762, 0.964]
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([2.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPTanh()
    result = propagate(strategy, bounds)

    lower_x = torch.tensor(1.0)
    upper_x = torch.tensor(2.0)
    lower_t = torch.tanh(lower_x)
    upper_t = torch.tanh(upper_x)
    secant_slope = (upper_t - lower_t) / (upper_x - lower_x)
    midpoint = (lower_x + upper_x) / 2.0
    midpoint_t = torch.tanh(midpoint)
    midpoint_prime = 1.0 - midpoint_t * midpoint_t

    assert torch.allclose(result.linear_lower, secant_slope.reshape(1, 1))
    assert torch.allclose(result.bias_lower, (lower_t - secant_slope * lower_x).reshape(1))
    assert torch.allclose(result.linear_upper, midpoint_prime.reshape(1, 1))
    assert torch.allclose(result.bias_upper, (midpoint_t - midpoint_prime * midpoint).reshape(1))

    lower, upper = result.concretize()
    assert torch.all(lower >= 0.75)
    assert torch.all(lower <= 0.77)
    assert torch.all(upper >= 0.96)
    assert torch.all(upper <= 1.0)


def test_tanh_negative_interval() -> None:
    """Test tanh on a negative interval."""
    # Region: x ∈ [-2, -1]
    # tanh([-2, -1]) ≈ [-0.964, -0.762]
    region = HyperRectangle(lower=torch.tensor([-2.0]), upper=torch.tensor([-1.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPTanh()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    assert torch.all(lower >= -1.0)
    assert torch.all(lower <= -0.96)
    assert torch.all(upper >= -0.77)
    assert torch.all(upper <= -0.75)


def test_tanh_mixed_sign_interval() -> None:
    """Test tanh on an interval crossing zero."""
    # Region: x ∈ [-1, 1]
    # tanh([-1, 1]) ≈ [-0.762, 0.762]
    region = HyperRectangle(lower=torch.tensor([-1.0]), upper=torch.tensor([1.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPTanh()
    result = propagate(strategy, bounds)

    lower_x = torch.tensor(-1.0)
    upper_x = torch.tensor(1.0)
    lower_t = torch.tanh(lower_x)
    upper_t = torch.tanh(upper_x)
    lower_prime = 1.0 - lower_t * lower_t
    upper_prime = 1.0 - upper_t * upper_t

    assert torch.allclose(result.linear_lower, lower_prime.reshape(1, 1))
    assert torch.allclose(result.bias_lower, (lower_t - lower_prime * lower_x).reshape(1))
    assert torch.allclose(result.linear_upper, upper_prime.reshape(1, 1))
    assert torch.allclose(result.bias_upper, (upper_t - upper_prime * upper_x).reshape(1))

    lower, upper = result.concretize()
    assert torch.all(lower >= -0.77)
    assert torch.all(lower <= -0.75)
    assert torch.all(upper >= 0.75)
    assert torch.all(upper <= 0.77)


def test_tanh_at_zero() -> None:
    """Test tanh at zero."""
    # Region: x ∈ [0, 0]
    # tanh([0, 0]) = [0, 0]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([0.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPTanh()
    result = propagate(strategy, bounds)

    assert torch.allclose(result.linear_lower, torch.tensor([[0.0]]), atol=1e-6)
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(result.linear_upper, torch.tensor([[0.0]]), atol=1e-6)
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]), atol=1e-6)

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(upper, torch.tensor([0.0]), atol=1e-6)


def test_tanh_large_positive_interval() -> None:
    """Test tanh on a large positive interval."""
    # Region: x ∈ [3, 5]
    # tanh([3, 5]) ≈ [0.995, 0.9999]
    region = HyperRectangle(lower=torch.tensor([3.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPTanh()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    assert torch.all(lower >= 0.99)
    assert torch.all(lower <= 1.0)
    assert torch.all(upper >= 0.999)
    assert torch.all(upper <= 1.01)
