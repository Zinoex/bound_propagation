from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.exp import ForwardLBPExp
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


def test_exp_positive_interval() -> None:
    """Test exp on a positive interval."""
    # Region: x ∈ [1, 2]
    # exp([1, 2]) = [e, e^2] ≈ [2.718, 7.389]
    # exp is convex, so:
    # Lower bound: secant line
    # Upper bound: tangent line (at upper for convex)
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([2.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPExp()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # exp(1) ≈ 2.718, exp(2) ≈ 7.389
    assert torch.all(lower >= 2.7)
    assert torch.all(lower <= 2.8)
    assert torch.all(upper >= 7.3)
    assert torch.all(upper <= 7.5)


def test_exp_negative_interval() -> None:
    """Test exp on a negative interval."""
    # Region: x ∈ [-2, -1]
    # exp([-2, -1]) = [e^-2, e^-1] ≈ [0.135, 0.368]
    region = HyperRectangle(lower=torch.tensor([-2.0]), upper=torch.tensor([-1.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPExp()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # exp(-2) ≈ 0.135, exp(-1) ≈ 0.368
    assert torch.all(lower >= 0.13)
    assert torch.all(lower <= 0.14)
    assert torch.all(upper >= 0.36)
    assert torch.all(upper <= 0.38)


def test_exp_mixed_sign_interval() -> None:
    """Test exp on an interval crossing zero."""
    # Region: x ∈ [-1, 1]
    # exp([-1, 1]) = [e^-1, e] ≈ [0.368, 2.718]
    region = HyperRectangle(lower=torch.tensor([-1.0]), upper=torch.tensor([1.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPExp()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.all(lower >= 0.36)
    assert torch.all(lower <= 0.38)
    assert torch.all(upper >= 2.7)
    assert torch.all(upper <= 2.75)


def test_exp_zero_interval() -> None:
    """Test exp at zero."""
    # Region: x ∈ [0, 0]
    # exp([0, 0]) = [1, 1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([0.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPExp()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([1.0]), atol=1e-6)
    assert torch.allclose(upper, torch.tensor([1.0]), atol=1e-6)


def test_exp_large_positive_interval() -> None:
    """Test exp on a large positive interval."""
    # Region: x ∈ [2, 3]
    # exp([2, 3]) = [e^2, e^3] ≈ [7.389, 20.086]
    region = HyperRectangle(lower=torch.tensor([2.0]), upper=torch.tensor([3.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPExp()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.all(lower >= 7.3)
    assert torch.all(lower <= 7.5)
    assert torch.all(upper >= 20.0)
    assert torch.all(upper <= 20.2)
