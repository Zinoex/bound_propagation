from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.sqrt import ForwardLBPSqrtStrategy
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


def test_sqrt_positive_interval() -> None:
    """Test sqrt on a positive interval."""
    # Region: x ∈ [1, 4]
    # Bounds: lower = x, upper = x
    # sqrt is concave, so:
    # - Lower bound uses tangent line (alpha_lower * x + beta_lower)
    # - Upper bound uses secant line (alpha_upper * x + beta_upper)
    # sqrt([1, 4]) = [1, 2]
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([4.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSqrtStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Should have linear relaxation
    assert result.linear_lower is not None
    assert result.linear_upper is not None

    # Compute expected bounds
    # For sqrt: derivative at x is 1/(2*sqrt(x))
    # Upper bound (secant): slope = (sqrt(4) - sqrt(1)) / (4 - 1) = (2 - 1) / 3 = 1/3
    # Passes through (1, 1): y = (1/3)(x - 1) + 1 = (1/3)x + 2/3
    # Lower bound (tangent at upper): slope at x=4 is 1/(2*sqrt(4)) = 1/4
    # y = (1/4)(x - 4) + 2 = (1/4)x + 1

    lower, upper = result.concretize()
    # At x=1: lower ≈ 1/4*1 + 1 = 1.25, upper = 1/3*1 + 2/3 = 1
    # At x=4: lower = 1/4*4 + 1 = 2, upper = 1/3*4 + 2/3 ≈ 2
    # So overall: [1, 2]
    assert torch.all(lower >= 0.99)  # slightly conservative
    assert torch.all(upper <= 2.01)
    # Verify it's within the actual range
    assert torch.all(lower <= torch.sqrt(region.upper))
    assert torch.all(upper >= torch.sqrt(region.lower))


def test_sqrt_small_interval() -> None:
    """Test sqrt on a small interval (near point interval)."""
    #  Region: x ∈ [4, 4.1]
    # Bounds: identity
    # sqrt([4, 4.1]) ≈ [2, 2.025]
    region = HyperRectangle(lower=torch.tensor([4.0]), upper=torch.tensor([4.1]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSqrtStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # sqrt(4) = 2, sqrt(4.1) ≈ 2.0248
    assert torch.all(lower >= 1.99)
    assert torch.all(lower <= 2.01)
    assert torch.all(upper >= 2.02)
    assert torch.all(upper <= 2.03)


def test_sqrt_zero_width_interval() -> None:
    """Test sqrt on a point interval (zero width)."""
    # Region: x ∈ [9, 9]
    # Bounds: identity
    # sqrt([9, 9]) = [3, 3]
    region = HyperRectangle(lower=torch.tensor([9.0]), upper=torch.tensor([9.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSqrtStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # For zero-width interval, relaxation should be tight
    assert torch.allclose(lower, torch.tensor([3.0]), atol=1e-6)
    assert torch.allclose(upper, torch.tensor([3.0]), atol=1e-6)


def test_sqrt_large_interval() -> None:
    """Test sqrt on a large interval."""
    # Region: x ∈ [1, 100]
    # Bounds: identity
    # sqrt([1, 100]) = [1, 10]
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([100.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSqrtStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # Due to relaxation, bounds should contain [1, 10]
    assert torch.all(lower <= 1.01)  # conservative lower
    assert torch.all(upper >= 9.99)  # conservative upper


def test_sqrt_with_bias() -> None:
    """Test sqrt with non-identity linear bounds."""
    # Region: x ∈ [0, 3]
    # Bounds: lower = 2x + 1, upper = 2x + 9
    # At x=0: input ∈ [1, 9], sqrt ∈ [1, 3]
    # At x=3: input ∈ [7, 15], sqrt ∈ [sqrt(7), sqrt(15)] ≈ [2.65, 3.87]
    # Conservative interval: [1, 3.87]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([3.0]))

    bounds = LinearBounds(
        region=region,
        linear_lower=torch.tensor([[2.0]]),
        bias_lower=torch.tensor([1.0]),
        linear_upper=torch.tensor([[2.0]]),
        bias_upper=torch.tensor([9.0]),
    )

    strategy = ForwardLBPSqrtStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # Min: sqrt(1) = 1, Max: sqrt(15) ≈ 3.87
    assert torch.all(lower >= 0.99)
    assert torch.all(lower <= 1.01)
    assert torch.all(upper >= 3.85)
    assert torch.all(upper <= 3.90)


def test_sqrt_multidimensional() -> None:
    """Test sqrt on multidimensional bounds."""
    # Region: x0 ∈ [1, 4], x1 ∈ [9, 16]
    # Bounds: identity
    # sqrt([1, 4], [9, 16]) = ([1, 2], [3, 4])
    region = HyperRectangle(lower=torch.tensor([1.0, 9.0]), upper=torch.tensor([4.0, 16.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSqrtStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # Element 0: sqrt([1, 4]) = [1, 2]
    assert torch.all(lower[0] >= 0.99)
    assert torch.all(upper[0] <= 2.01)
    # Element 1: sqrt([9, 16]) = [3, 4]
    assert torch.all(lower[1] >= 2.99)
    assert torch.all(upper[1] <= 4.01)
