from __future__ import annotations

import math

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.sin import ForwardLBPSinStrategy
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


def test_sin_small_positive_interval() -> None:
    """Test sin on a small interval in [0, π/2] where sin is monotone increasing."""
    # Region: x ∈ [0, π/4]
    # sin([0, π/4]) = [sin(0), sin(π/4)] = [0, √2/2] ≈ [0, 0.707]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([math.pi / 4]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    # Should have linear relaxation
    assert result.linear_lower is not None
    assert result.linear_upper is not None

    lower, upper = result.concretize()
    # sin(0) = 0, sin(π/4) ≈ 0.707
    assert torch.all(lower >= -0.01)
    assert torch.all(lower <= 0.01)
    assert torch.all(upper >= 0.70)
    assert torch.all(upper <= 0.72)


def test_sin_increasing_interval() -> None:
    """Test sin on interval where it's monotone increasing."""
    # Region: x ∈ [0, π/2]
    # sin([0, π/2]) = [sin(0), sin(π/2)] = [0, 1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([math.pi / 2]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # sin(0) = 0, sin(π/2) = 1
    assert torch.all(lower >= -0.01)
    assert torch.all(lower <= 0.01)
    assert torch.all(upper >= 0.99)
    assert torch.all(upper <= 1.01)


def test_sin_crossing_maximum() -> None:
    """Test sin crossing its maximum at π/2."""
    # Region: x ∈ [π/4, 3π/4]
    # sin(π/2) = 1 (maximum in this interval)
    # sin([π/4, 3π/4]) = [√2/2, 1] ≈ [0.707, 1]
    region = HyperRectangle(lower=torch.tensor([math.pi / 4]), upper=torch.tensor([3 * math.pi / 4]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # Should contain [√2/2, 1] ≈ [0.707, 1]
    assert torch.all(lower <= 0.72)
    assert torch.all(upper >= 0.99)
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_sin_full_period() -> None:
    """Test sin on a full period."""
    # Region: x ∈ [0, 2π]
    # sin([0, 2π]) = [-1, 1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([2 * math.pi]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # Should contain full range [-1, 1]
    assert torch.all(lower >= -1.01)
    assert torch.all(lower <= -0.99)
    assert torch.all(upper >= 0.99)
    assert torch.all(upper <= 1.01)


def test_sin_negative_interval() -> None:
    """Test sin on a negative interval."""
    # Region: x ∈ [-π/2, 0]
    # sin([-π/2, 0]) = [-1, 0]
    region = HyperRectangle(lower=torch.tensor([-math.pi / 2]), upper=torch.tensor([0.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # sin(-π/2) = -1, sin(0) = 0
    assert torch.all(lower >= -1.01)
    assert torch.all(lower <= -0.99)
    assert torch.all(upper >= -0.01)
    assert torch.all(upper <= 0.01)


def test_sin_zero_width() -> None:
    """Test sin on a point interval."""
    # Region: x ∈ [π/6, π/6]
    # sin(π/6) = 0.5
    region = HyperRectangle(lower=torch.tensor([math.pi / 6]), upper=torch.tensor([math.pi / 6]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # For point interval, should be tight
    assert torch.allclose(lower, torch.tensor([0.5]), atol=1e-5)
    assert torch.allclose(upper, torch.tensor([0.5]), atol=1e-5)


def test_sin_zero_width_rising() -> None:
    """Test sin on a point interval in rising region."""
    # Region: x ∈ [π/4, π/4]
    # sin(π/4) = √2/2 ≈ 0.707
    # This is in the rising region [-π/2, π/2] where derivative is positive
    region = HyperRectangle(lower=torch.tensor([math.pi / 4]), upper=torch.tensor([math.pi / 4]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # For point interval, should be tight
    expected = math.sin(math.pi / 4)  # ≈ 0.707
    assert torch.allclose(lower, torch.tensor([expected]), atol=1e-5)
    assert torch.allclose(upper, torch.tensor([expected]), atol=1e-5)


def test_sin_zero_width_falling() -> None:
    """Test sin on a point interval in falling region."""
    # Region: x ∈ [2π/3, 2π/3]
    # sin(2π/3) = √3/2 ≈ 0.866
    # This is in the falling region [π/2, 3π/2] where derivative is negative
    region = HyperRectangle(lower=torch.tensor([2 * math.pi / 3]), upper=torch.tensor([2 * math.pi / 3]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()
    # For point interval, should be tight
    expected = math.sin(2 * math.pi / 3)  # ≈ 0.866
    assert torch.allclose(lower, torch.tensor([expected]), atol=1e-5)
    assert torch.allclose(upper, torch.tensor([expected]), atol=1e-5)


def test_sin_concave_region_first_half() -> None:
    """Test sin on concave region [0, π] - first half where sin is increasing and concave."""
    # Region: x ∈ [π/4, π/2]
    # In this region: sin is increasing (derivative > 0) and concave (second derivative < 0)
    # sin(π/4) ≈ 0.707, sin(π/2) = 1
    region = HyperRectangle(lower=torch.tensor([math.pi / 4]), upper=torch.tensor([math.pi / 2]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()

    # Actual range: [√2/2, 1] ≈ [0.707, 1]
    actual_lower = math.sin(math.pi / 4)  # ≈ 0.707
    actual_upper = math.sin(math.pi / 2)  # = 1

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_sin_concave_region_second_half() -> None:
    """Test sin on concave region [0, π] - second half where sin is decreasing and concave."""
    # Region: x ∈ [π/2, 3π/4]
    # sin is decreasing (derivative < 0 after π/2) and concave
    # sin(π/2) = 1, sin(3π/4) ≈ 0.707
    region = HyperRectangle(lower=torch.tensor([math.pi / 2]), upper=torch.tensor([3 * math.pi / 4]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()

    # Actual range: [√2/2, 1]
    actual_lower = math.sin(3 * math.pi / 4)  # ≈ 0.707
    actual_upper = math.sin(math.pi / 2)  # = 1

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_sin_convex_region_first_half() -> None:
    """Test sin on convex region [π, 2π] - first half where sin is decreasing and convex."""
    # Region: x ∈ [π, 5π/4]
    # In this region: sin is decreasing and convex (second derivative > 0)
    # sin(π) = 0, sin(5π/4) ≈ -0.707
    region = HyperRectangle(lower=torch.tensor([math.pi]), upper=torch.tensor([5 * math.pi / 4]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()

    # Actual range: [-√2/2, 0]
    actual_lower = math.sin(5 * math.pi / 4)  # ≈ -0.707
    actual_upper = math.sin(math.pi)  # = 0

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_sin_convex_region_second_half() -> None:
    """Test sin on convex region [π, 2π] - second half where sin is increasing and convex."""
    # Region: x ∈ [5π/4, 3π/2]
    # sin is increasing and convex
    # sin(5π/4) ≈ -0.707, sin(3π/2) = -1
    region = HyperRectangle(lower=torch.tensor([5 * math.pi / 4]), upper=torch.tensor([3 * math.pi / 2]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()

    # Actual range: [-1, -√2/2]
    actual_lower = math.sin(3 * math.pi / 2)  # = -1
    actual_upper = math.sin(5 * math.pi / 4)  # ≈ -0.707

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_sin_across_maximum_at_half_pi() -> None:
    """Test sin crossing maximum at π/2."""
    # Region: x ∈ [π/4, 3π/4]
    # Crosses maximum at x=π/2 where sin(π/2) = 1
    region = HyperRectangle(lower=torch.tensor([math.pi / 4]), upper=torch.tensor([3 * math.pi / 4]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()

    # Actual range: [sin(π/4), 1] ≈ [0.707, 1]
    actual_lower = math.sin(math.pi / 4)  # ≈ 0.707
    actual_upper = 1.0  # Maximum at π/2

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_sin_across_minimum_at_3half_pi() -> None:
    """Test sin crossing minimum at 3π/2."""
    # Region: x ∈ [5π/4, 7π/4]
    # Crosses minimum at x=3π/2 where sin(3π/2) = -1
    region = HyperRectangle(lower=torch.tensor([5 * math.pi / 4]), upper=torch.tensor([7 * math.pi / 4]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()

    # Actual range: [-1, sin(5π/4)] where sin(5π/4) ≈ -0.707
    actual_lower = -1.0  # Minimum at 3π/2
    actual_upper = math.sin(5 * math.pi / 4)  # ≈ -0.707

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_sin_strictly_increasing_small() -> None:
    """Test sin on a small strictly increasing interval."""
    # Region: x ∈ [0, π/3]
    # sin is strictly increasing in [-π/2, π/2]
    # sin(0) = 0, sin(π/3) = √3/2 ≈ 0.866
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([math.pi / 3]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()

    # Actual range: [sin(0), sin(π/3)] = [0, √3/2]
    actual_lower = math.sin(0.0)  # = 0
    actual_upper = math.sin(math.pi / 3)  # ≈ 0.866

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_sin_strictly_decreasing_small() -> None:
    """Test sin on a small strictly decreasing interval."""
    # Region: x ∈ [2π/3, π]
    # sin is strictly decreasing in [π/2, 3π/2]
    # sin(2π/3) = √3/2 ≈ 0.866, sin(π) = 0
    region = HyperRectangle(lower=torch.tensor([2 * math.pi / 3]), upper=torch.tensor([math.pi]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()

    # Actual range: [sin(π), sin(2π/3)] = [0, √3/2]
    actual_lower = math.sin(math.pi)  # = 0
    actual_upper = math.sin(2 * math.pi / 3)  # ≈ 0.866

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_sin_mixed_convexity_half_wave() -> None:
    """Test sin on interval spanning both concave and convex regions."""
    # Region: x ∈ [π/2, 3π/2]
    # Spans concave region [π/2, π] and convex region [π, 3π/2]
    # Crosses minimum at 3π/2
    region = HyperRectangle(lower=torch.tensor([math.pi / 2]), upper=torch.tensor([3 * math.pi / 2]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()

    # Actual range: [-1, 1] (endpoints at 1, minimum at -1)
    actual_lower = -1.0  # Minimum at 3π/2
    actual_upper = 1.0  # Max at π/2

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_sin_negative_region_increasing() -> None:
    """Test sin on a negative interval where sin is increasing."""
    # Region: x ∈ [-π/2, -π/4]
    # sin is increasing in [-π/2, π/2], and convex in this range (since [π, 2π] shifted)
    # sin(-π/2) = -1, sin(-π/4) ≈ -0.707
    region = HyperRectangle(lower=torch.tensor([-math.pi / 2]), upper=torch.tensor([-math.pi / 4]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()

    # Actual range: [-1, -√2/2]
    actual_lower = math.sin(-math.pi / 2)  # = -1
    actual_upper = math.sin(-math.pi / 4)  # ≈ -0.707

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_sin_large_interval_multiple_periods() -> None:
    """Test sin on interval spanning multiple periods."""
    # Region: x ∈ [0, 3π]
    # Spans 1.5 periods, includes maximum at π/2 and minimum at 3π/2
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([3 * math.pi]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPSinStrategy()
    result = strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]

    lower, upper = result.concretize()

    # Actual range: [-1, 1] (full range of sine)
    actual_lower = -1.0
    actual_upper = 1.0

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"
