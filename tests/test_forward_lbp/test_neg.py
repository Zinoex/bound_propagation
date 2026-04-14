from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.neg import ForwardLBPNeg
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


def test_neg_positive_interval() -> None:
    """Test negation of a positive interval."""
    # Region: x ∈ [2, 5]
    # Bounds: lower = x, upper = x
    # -x ∈ [-5, -2]
    region = HyperRectangle(lower=torch.tensor([2.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPNeg()
    result = propagate(strategy, bounds)

    # Linear: negated identity -1
    assert torch.allclose(result.linear_lower, torch.tensor([[-1.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[-1.0]]))
    # Bias: 0 (unchanged)
    assert torch.allclose(result.bias_lower, torch.tensor([0.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]))

    # At x=2: -2, At x=5: -5
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([-5.0]))
    assert torch.allclose(upper, torch.tensor([-2.0]))


def test_neg_negative_interval() -> None:
    """Test negation of a negative interval."""
    # Region: x ∈ [-3, -1]
    # Bounds: lower = x, upper = x
    # -x ∈ [1, 3]
    region = HyperRectangle(lower=torch.tensor([-3.0]), upper=torch.tensor([-1.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPNeg()
    result = propagate(strategy, bounds)

    # Linear: -1
    assert torch.allclose(result.linear_lower, torch.tensor([[-1.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[-1.0]]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([1.0]))
    assert torch.allclose(upper, torch.tensor([3.0]))


def test_neg_crossing_zero() -> None:
    """Test negation of an interval crossing zero."""
    # Region: x ∈ [-2, 3]
    # Bounds: lower = x, upper = x
    # -x ∈ [-3, 2]
    region = HyperRectangle(lower=torch.tensor([-2.0]), upper=torch.tensor([3.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPNeg()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([-3.0]))
    assert torch.allclose(upper, torch.tensor([2.0]))


def test_neg_with_bias() -> None:
    """Test negation with non-identity linear bounds."""
    # Region: x ∈ [0, 2]
    # Bounds: lower = 2x + 1, upper = 2x + 5
    # -(2x + 1) = -2x - 1, -(2x + 5) = -2x - 5
    # At x=0: -[1, 5] = [-5, -1]
    # At x=2: -[5, 9] = [-9, -5]
    # Overall: [-9, -1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([2.0]))

    bounds = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[2.0]]),
        bias_lower=torch.tensor([1.0]),
        linear_upper=torch.tensor([[2.0]]),
        bias_upper=torch.tensor([5.0]),
    )

    strategy = ForwardLBPNeg()
    result = propagate(strategy, bounds)

    # Linear: -2 (swapped because negation swaps bounds)
    assert torch.allclose(result.linear_lower, torch.tensor([[-2.0]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[-2.0]]))
    # Bias: -5, -1 (swapped)
    assert torch.allclose(result.bias_lower, torch.tensor([-5.0]))
    assert torch.allclose(result.bias_upper, torch.tensor([-1.0]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([-9.0]))
    assert torch.allclose(upper, torch.tensor([-1.0]))


def test_neg_multidimensional() -> None:
    """Test negation on multidimensional bounds."""
    # Region: x0 ∈ [1, 3], x1 ∈ [-2, 4], x2 ∈ [-1, 0]
    # Bounds: identity
    # -x: ([-3, -1], [-4, 2], [0, 1])
    region = HyperRectangle(lower=torch.tensor([1.0, -2.0, -1.0]), upper=torch.tensor([3.0, 4.0, 0.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPNeg()
    result = propagate(strategy, bounds)

    # Linear: -I
    assert torch.allclose(result.linear_lower, -torch.eye(3))
    assert torch.allclose(result.linear_upper, -torch.eye(3))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([-3.0, -4.0, 0.0]))
    assert torch.allclose(upper, torch.tensor([-1.0, 2.0, 1.0]))


def test_neg_zero_interval() -> None:
    """Test negation of zero interval."""
    # Region: x ∈ [0, 0]
    # Bounds: lower = x, upper = x
    # -x = [0, 0]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([0.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPNeg()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([0.0]))
    assert torch.allclose(upper, torch.tensor([0.0]))
