from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.abs import ForwardLBPAbs
from bound_propagation.regions import HyperRectangle


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


def test_abs_positive_interval() -> None:
    """Test abs on a positive interval."""
    # Region: x ∈ [2, 5]
    # Bounds: lower = x, upper = x
    # abs([2, 5]) = [2, 5] (identity since all positive)
    region = HyperRectangle(lower=torch.tensor([2.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPAbs()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # For all positive, abs(x) = x, so alpha = 1, beta = 0
    assert torch.allclose(result.linear_lower, torch.tensor([[1.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[1.0]]))
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([2.0]))
    assert torch.allclose(upper, torch.tensor([5.0]))


def test_abs_negative_interval() -> None:
    """Test abs on a negative interval."""
    # Region: x ∈ [-5, -2]
    # Bounds: lower = x, upper = x
    # abs([-5, -2]) = [2, 5]
    region = HyperRectangle(lower=torch.tensor([-5.0]), upper=torch.tensor([-2.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPAbs()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # For all negative, abs(x) = -x, so alpha = -1, beta = 0
    assert torch.allclose(result.linear_lower, torch.tensor([[-1.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[-1.0]]))
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]))

    lower, upper = result.concretize()
    # At x=-5: abs(-5) = 5, at x=-2: abs(-2) = 2
    # lower = -1 * (-2) + 0 = 2
    # upper = -1 * (-5) + 0 = 5
    assert torch.allclose(lower, torch.tensor([2.0]))
    assert torch.allclose(upper, torch.tensor([5.0]))


def test_abs_mixed_sign_interval() -> None:
    """Test abs on an interval containing zero."""
    # Region: x ∈ [-3, 4]
    # Bounds: lower = x, upper = x
    # abs([-3, 4]) = [0, 4]
    # Lower bound: 0
    # Upper bound: line connecting (-3, 3) and (4, 4)
    # slope = (4 - 3) / (4 - (-3)) = 1/7 ≈ 0.142857
    # y = (1/7)(x - (-3)) + 3 = (1/7)x + 3/7 + 3 = (1/7)x + 24/7
    region = HyperRectangle(lower=torch.tensor([-3.0]), upper=torch.tensor([4.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPAbs()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # Lower bound should be at least 0
    assert torch.all(lower >= 0.0)
    # At x=-3: abs(-3) = 3, at x=4: abs(4) = 4
    # Upper bound should be at most 4
    assert torch.all(upper <= 4.0)
    # At x=0: should be around 0
    # Test at endpoints
    assert torch.all(lower <= 1.0)  # conservative, crosses zero


def test_abs_mixed_sign_larger_negative() -> None:
    """Test abs of interval containing zero with larger negative magnitude."""
    # Region: x ∈ [-7, 3]
    # Bounds: lower = x, upper = x
    # abs([-7, 3]) = [0, 7]
    region = HyperRectangle(lower=torch.tensor([-7.0]), upper=torch.tensor([3.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPAbs()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.all(lower >= 0.0)
    assert torch.all(upper <= 7.1)  # slight tolerance
    # At x=-7: abs(-7) = 7
    # Upper bound at x=-7 should be at most 7
    assert torch.all(upper >= 6.9)


def test_abs_symmetric_interval() -> None:
    """Test abs of symmetric interval around zero."""
    # Region: x ∈ [-5, 5]
    # Bounds: lower = x, upper = x
    # abs([-5, 5]) = [0, 5]
    region = HyperRectangle(lower=torch.tensor([-5.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPAbs()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # Lower should be 0 at x=0
    assert torch.all(lower >= 0.0)
    assert torch.all(lower <= 0.5)  # should be close to 0
    # Upper should be 5 at both ends
    assert torch.allclose(upper, torch.tensor([5.0]), atol=0.1)


def test_abs_zero_interval() -> None:
    """Test abs of zero interval."""
    # Region: x ∈ [0, 0]
    # Bounds: lower = x, upper = x
    # abs([0, 0]) = [0, 0]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([0.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPAbs()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(upper, torch.tensor([0.0]), atol=1e-6)


def test_abs_point_interval() -> None:
    """Test abs of point interval (zero width)."""
    # Region: x ∈ [3, 3]
    # Bounds: lower = x, upper = x
    # abs([3, 3]) = [3, 3]
    region = HyperRectangle(lower=torch.tensor([3.0]), upper=torch.tensor([3.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPAbs()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([3.0]), atol=1e-6)
    assert torch.allclose(upper, torch.tensor([3.0]), atol=1e-6)
