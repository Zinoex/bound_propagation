from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.reciprocal import ForwardLBPReciprocal
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


def test_reciprocal_positive_interval() -> None:
    """Test reciprocal on a positive interval."""
    # Region: x ∈ [2, 4]
    # Bounds: lower = x, upper = x
    # reciprocal([2, 4]) = [1/4, 1/2] = [0.25, 0.5]
    # reciprocal is convex for x > 0
    # Lower bound: tangent line at midpoint d = 3
    # f(d) = 1/3, f'(d) = -1/9
    # y = -1/9 * (x - 3) + 1/3 = -1/9 * x + 1/3 + 3/9 = -1/9 * x + 2/3
    # Upper bound: secant line
    # slope = (1/4 - 1/2) / (4 - 2) = -0.25/2 = -0.125
    # y = -0.125 * (x - 2) + 0.5 = -0.125 * x + 0.25 + 0.5 = -0.125 * x + 0.75
    region = HyperRectangle(lower=torch.tensor([2.0]), upper=torch.tensor([4.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPReciprocal()
    result = propagate(strategy, bounds)

    # Should have linear relaxation
    assert result.linear_lower is not None
    assert result.linear_upper is not None

    assert torch.allclose(result.linear_lower, torch.tensor([[-1.0 / 9.0]]))
    assert torch.allclose(result.bias_lower, torch.tensor([2.0 / 3.0]))
    assert torch.allclose(result.linear_upper, torch.tensor([[-0.125]]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.75]))

    lower, upper = result.concretize()
    # At x=2: lower = -1/9 * 2 + 2/3 = -2/9 + 6/9 = 4/9 ≈ 0.444
    #         upper = -0.125 * 2 + 0.75 = 0.5
    # At x=4: lower = -1/9 * 4 + 2/3 = -4/9 + 6/9 = 2/9 ≈ 0.222
    #         upper = -0.125 * 4 + 0.75 = 0.25
    # Overall: lower >= 0.222, upper <= 0.5
    assert torch.all(lower >= 0.22)
    assert torch.all(lower <= 0.45)
    assert torch.all(upper >= 0.24)
    assert torch.all(upper <= 0.51)


def test_reciprocal_negative_interval() -> None:
    """Test reciprocal on a negative interval."""
    # Region: x ∈ [-4, -2]
    # Bounds: lower = x, upper = x
    # reciprocal([-4, -2]) = [-0.5, -0.25]
    # reciprocal is convex for x < 0
    # Lower bound: secant line
    # slope = (-0.5 - (-0.25)) / (-2 - (-4)) = -0.25/2 = -0.125
    # y = -0.125x - 0.75
    # At x=-4: -0.125*(-4) - 0.75 = 0.5 - 0.75 = -0.25
    # At x=-2: -0.125*(-2) - 0.75 = 0.25 - 0.75 = -0.5
    # Upper bound: tangent at d=-3
    # f(-3) = -1/3, f'(-3) = -1/9
    # y = -1/9 * x - 2/3
    # At x=-4: -1/9*(-4) - 2/3 = 4/9 - 6/9 = -2/9 ≈ -0.222
    # At x=-2: -1/9*(-2) - 2/3 = 2/9 - 6/9 = -4/9 ≈ -0.444
    region = HyperRectangle(lower=torch.tensor([-4.0]), upper=torch.tensor([-2.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPReciprocal()
    result = propagate(strategy, bounds)

    assert torch.allclose(result.linear_lower, torch.tensor([[-0.125]]))
    assert torch.allclose(result.bias_lower, torch.tensor([-0.75]))
    assert torch.allclose(result.linear_upper, torch.tensor([[-1.0 / 9.0]]))
    assert torch.allclose(result.bias_upper, torch.tensor([-2.0 / 3.0]))

    lower, upper = result.concretize()
    # Lower bound should be in [-0.5, -0.25]
    assert torch.all(lower >= -0.51)
    assert torch.all(lower <= -0.24)
    # Upper bound should be in [-0.444, -0.222]
    assert torch.all(upper >= -0.45)
    assert torch.all(upper <= -0.21)


def test_reciprocal_positive_small_interval() -> None:
    """Test reciprocal on a small positive interval."""
    # Region: x ∈ [0.25, 0.5]
    # Bounds: lower = x, upper = x
    # reciprocal([0.25, 0.5]) = [2, 4]
    # For positive x: lower=tangent at midpoint, upper=secant
    # Midpoint d = 0.375: f(d) = 2.667, f'(d) ≈ -7.111
    # Lower (tangent): y = -7.111(x - 0.375) + 2.667
    # At x=0.25: y ≈ 3.556, at x=0.5: y ≈ 1.778
    # Upper (secant): slope = (2-4)/(0.5-0.25) = -8, y = -8x + 6
    region = HyperRectangle(lower=torch.tensor([0.25]), upper=torch.tensor([0.5]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPReciprocal()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    # Lower bound should be around 1.778
    assert torch.all(lower >= 1.7)
    assert torch.all(lower <= 1.9)
    # Upper bound should be around 4
    assert torch.all(upper >= 3.5)
    assert torch.all(upper <= 4.1)


def test_reciprocal_crossing_zero() -> None:
    """Test reciprocal on an interval crossing zero (unbounded)."""
    # Region: x ∈ [-2, 3]
    # Bounds: lower = x, upper = x
    # reciprocal([-2, 3]) = [-inf, inf] (unbounded)
    region = HyperRectangle(lower=torch.tensor([-2.0]), upper=torch.tensor([3.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPReciprocal()
    result = propagate(strategy, bounds)

    assert torch.allclose(result.linear_lower, torch.tensor([[0.0]]))
    assert torch.isneginf(result.bias_lower).all()
    assert torch.allclose(result.linear_upper, torch.tensor([[0.0]]))
    assert torch.isposinf(result.bias_upper).all()

    lower, upper = result.concretize()
    # Should be unbounded
    assert torch.isneginf(lower).all()
    assert torch.isposinf(upper).all()


def test_reciprocal_zero_lower_bound() -> None:
    """Test reciprocal with zero as lower bound (edge case)."""
    # Region: x ∈ [0, 2]
    # Bounds: lower = x, upper = x
    # reciprocal([0, 2]) has division by zero at x=0
    # Implementation treats this as an edge case with alpha=0, beta=0
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([2.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPReciprocal()
    result = propagate(strategy, bounds)

    assert torch.allclose(result.linear_lower, torch.tensor([[0.0]]))
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]))
    assert torch.allclose(result.linear_upper, torch.tensor([[0.0]]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]))

    lower, upper = result.concretize()
    # Implementation returns safe bounds (0, 0) for intervals touching zero
    assert torch.all(lower >= -0.1)
    assert torch.all(upper <= 0.1)


def test_reciprocal_zero_upper_bound() -> None:
    """Test reciprocal with zero as upper bound (edge case)."""
    # Region: x ∈ [-2, 0]
    # Bounds: lower = x, upper = x
    # reciprocal([-2, 0]) has division by zero at x=0
    # Implementation treats this as an edge case with alpha=0, beta=0
    region = HyperRectangle(lower=torch.tensor([-2.0]), upper=torch.tensor([0.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPReciprocal()
    result = propagate(strategy, bounds)

    assert torch.allclose(result.linear_lower, torch.tensor([[0.0]]))
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]))
    assert torch.allclose(result.linear_upper, torch.tensor([[0.0]]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]))

    lower, upper = result.concretize()
    # Implementation returns safe bounds (0, 0) for intervals touching zero
    assert torch.all(lower >= -0.1)
    assert torch.all(upper <= 0.1)


def test_reciprocal_point_interval() -> None:
    """Test reciprocal on a point interval (zero width)."""
    # Region: x ∈ [4, 4]
    # Bounds: lower = x, upper = x
    # reciprocal([4, 4]) = [0.25, 0.25]
    region = HyperRectangle(lower=torch.tensor([4.0]), upper=torch.tensor([4.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPReciprocal()
    result = propagate(strategy, bounds)

    assert torch.allclose(result.linear_lower, torch.tensor([[-1.0 / 16.0]]))
    assert torch.allclose(result.bias_lower, torch.tensor([0.5]))
    assert torch.allclose(result.linear_upper, torch.tensor([[0.0]]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.25]))

    lower, upper = result.concretize()
    # For zero-width interval, relaxation should be tight
    assert torch.allclose(lower, torch.tensor([0.25]), atol=1e-6)
    assert torch.allclose(upper, torch.tensor([0.25]), atol=1e-6)
