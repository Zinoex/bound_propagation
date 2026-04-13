from __future__ import annotations

import math

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.cos import ForwardLBPCosStrategy
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


def test_cos_small_positive_interval() -> None:
    """Test cos on a small interval in [0, π/2] where cos is monotone decreasing."""
    # Region: x ∈ [0, π/4]
    # Bounds: identity
    # cos([0, π/4]) = [cos(π/4), cos(0)] = [√2/2, 1] ≈ [0.707, 1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([math.pi / 4]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCosStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Should have linear relaxation
    assert result.linear_lower is not None
    assert result.linear_upper is not None

    lower, upper = result.concretize()
    # cos(π/4) ≈ 0.707, cos(0) = 1
    assert torch.all(lower >= 0.70)
    assert torch.all(lower <= 0.72)
    assert torch.all(upper >= 0.99)
    assert torch.all(upper <= 1.01)


def test_cos_decreasing_interval() -> None:
    """Test cos on interval where it's monotone decreasing."""
    # Region: x ∈ [0, π/2]
    # Bounds: identity
    # cos([0, π/2]) = [cos(π/2), cos(0)] = [0, 1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([math.pi / 2]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCosStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # cos(π/2) ≈ 0, cos(0) = 1
    assert torch.all(lower >= -0.01)
    assert torch.all(lower <= 0.01)
    assert torch.all(upper >= 0.99)
    assert torch.all(upper <= 1.01)


def test_cos_crossing_minimum() -> None:
    """Test cos crossing its minimum at π."""
    # Region: x ∈ [π/2, 3π/2]
    # Bounds: identity
    # cos(π) = -1 (minimum in this interval)
    # cos([π/2, 3π/2]) = [-1, 0] (since cos(π/2) ≈ 0, cos(3π/2) ≈ 0)
    region = HyperRectangle(lower=torch.tensor([math.pi / 2]), upper=torch.tensor([3 * math.pi / 2]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCosStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # Should contain [-1, 0]
    assert torch.all(lower >= -1.01)
    assert torch.all(lower <= -0.99)
    assert torch.all(upper >= -0.01)
    assert torch.all(upper <= 0.01)


def test_cos_full_period() -> None:
    """Test cos over a full period."""
    # Region: x ∈ [0, 2π]
    # Bounds: identity
    # cos([0, 2π]) = [-1, 1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([2 * math.pi]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCosStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # Should contain full range [-1, 1]
    assert torch.all(lower >= -1.01)
    assert torch.all(lower <= -0.99)
    assert torch.all(upper >= 0.99)
    assert torch.all(upper <= 1.01)


def test_cos_negative_interval() -> None:
    """Test cos on a negative interval."""
    # Region: x ∈ [-π/4, 0]
    # Bounds: identity
    # cos([-π/4, 0]) = [cos(π/4), cos(0)] = [√2/2, 1] ≈ [0.707, 1]
    # (cos is even, so cos(-x) = cos(x))
    region = HyperRectangle(lower=torch.tensor([-math.pi / 4]), upper=torch.tensor([0.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCosStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    assert torch.all(lower >= 0.70)
    assert torch.all(lower <= 0.72)
    assert torch.all(upper >= 0.99)
    assert torch.all(upper <= 1.01)


def test_cos_zero_width() -> None:
    """Test cos on a point interval."""
    # Region: x ∈ [π/3, π/3]
    # Bounds: identity
    # cos(π/3) = 0.5
    region = HyperRectangle(lower=torch.tensor([math.pi / 3]), upper=torch.tensor([math.pi / 3]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCosStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # For point interval, should be tight
    assert torch.allclose(lower, torch.tensor([0.5]), atol=1e-5)
    assert torch.allclose(upper, torch.tensor([0.5]), atol=1e-5)
