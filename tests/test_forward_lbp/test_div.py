from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.div import ForwardLBPDiv
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


def test_div_abstract_abstract_positive() -> None:
    """Test division of two abstract positive intervals."""
    # Region: x0 ∈ [6, 12], x1 ∈ [2, 3]
    # Division: x0 / x1
    # Result: [6/3, 12/2] = [2, 6]
    region = HyperRectangle(lower=torch.tensor([6.0, 2.0]), upper=torch.tensor([12.0, 3.0]))

    bounds_a = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[1.0, 0.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[1.0, 0.0]]),
        bias_upper=torch.tensor([0.0]),
    )
    bounds_b = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[0.0, 1.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[0.0, 1.0]]),
        bias_upper=torch.tensor([0.0]),
    )

    strategy = ForwardLBPDiv()
    result = propagate(strategy, bounds_a, bounds_b)

    # Division loses linear dependency
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([2.0]))
    assert torch.allclose(upper, torch.tensor([6.0]))


def test_div_abstract_constant_positive() -> None:
    """Test division of abstract by positive constant."""
    # Region: x ∈ [4, 8]
    # Division: x / 2
    # Result: [2, 4]
    region = HyperRectangle(lower=torch.tensor([4.0]), upper=torch.tensor([8.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPDiv()
    result = propagate(strategy, bounds, torch.tensor(2.0))

    # Should preserve linear structure: x/2
    assert result.linear_lower is not None
    assert result.linear_upper is not None
    assert torch.allclose(result.linear_lower, torch.tensor([[0.5]]))
    assert torch.allclose(result.linear_upper, torch.tensor([[0.5]]))

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([2.0]))
    assert torch.allclose(upper, torch.tensor([4.0]))


def test_div_abstract_constant_negative() -> None:
    """Test division of abstract by negative constant."""
    # Region: x ∈ [4, 8]
    # Division: x / (-2)
    # Result: [-4, -2]
    region = HyperRectangle(lower=torch.tensor([4.0]), upper=torch.tensor([8.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPDiv()
    result = propagate(strategy, bounds, torch.tensor(-2.0))

    # Linear bounds should be flipped: x/(-2) = -x/2
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([-4.0]))
    assert torch.allclose(upper, torch.tensor([-2.0]))


def test_div_crossing_zero_divisor() -> None:
    """Test division when divisor crosses zero (unbounded)."""
    # Region: x0 ∈ [4, 8], x1 ∈ [-1, 1]
    # Division: x0 / x1 (x1 crosses zero)
    # Result: [-inf, inf]
    region = HyperRectangle(lower=torch.tensor([4.0, -1.0]), upper=torch.tensor([8.0, 1.0]))

    bounds_a = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[1.0, 0.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[1.0, 0.0]]),
        bias_upper=torch.tensor([0.0]),
    )
    bounds_b = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[0.0, 1.0]]),
        bias_lower=torch.tensor([0.0]),
        linear_upper=torch.tensor([[0.0, 1.0]]),
        bias_upper=torch.tensor([0.0]),
    )

    strategy = ForwardLBPDiv()
    result = propagate(strategy, bounds_a, bounds_b)

    lower, upper = result.concretize()
    assert torch.isneginf(lower).all()
    assert torch.isposinf(upper).all()
