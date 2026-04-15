from __future__ import annotations

import math

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.cos import ForwardLBPCos
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


def test_cos_small_positive_interval() -> None:
    """Test cos on a small interval in [0, π/2] where cos is monotone decreasing."""
    # Region: x ∈ [0, π/4]
    # Bounds: identity
    # cos([0, π/4]) = [cos(π/4), cos(0)] = [√2/2, 1] ≈ [0.707, 1]
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([math.pi / 4]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    # Should have linear relaxation
    assert result.linear_lower is not None
    assert result.linear_upper is not None

    lower_x = torch.tensor(0.0)
    upper_x = torch.tensor(math.pi / 4)
    secant_slope = (torch.cos(upper_x) - torch.cos(lower_x)) / (upper_x - lower_x)

    assert torch.allclose(result.linear_lower, secant_slope.reshape(1, 1))
    assert torch.allclose(result.bias_lower, torch.tensor([1.0]))
    assert torch.allclose(result.linear_upper, torch.tensor([[0.0]]), atol=1e-6)
    assert torch.allclose(result.bias_upper, torch.tensor([1.0]), atol=1e-6)

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

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

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

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    # Crossing the cosine minimum yields constant lower -1 and constant upper at the endpoint max.
    assert torch.allclose(result.linear_lower, torch.tensor([[0.0]]), atol=1e-6)
    assert torch.allclose(result.bias_lower, torch.tensor([-1.0]), atol=1e-6)
    assert torch.allclose(result.linear_upper, torch.tensor([[0.0]]), atol=1e-6)
    assert torch.allclose(result.bias_upper, torch.tensor([0.0]), atol=1e-5)

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

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    assert torch.allclose(result.linear_lower, torch.tensor([[0.0]]), atol=1e-6)
    assert torch.allclose(result.bias_lower, torch.tensor([-1.0]), atol=1e-6)
    assert torch.allclose(result.linear_upper, torch.tensor([[0.0]]), atol=1e-6)
    assert torch.allclose(result.bias_upper, torch.tensor([1.0]), atol=1e-6)

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

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

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

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    assert torch.allclose(result.linear_lower, torch.tensor([[0.0]]), atol=1e-6)
    assert torch.allclose(result.bias_lower, torch.tensor([0.5]), atol=1e-5)
    assert torch.allclose(result.linear_upper, torch.tensor([[0.0]]), atol=1e-6)
    assert torch.allclose(result.bias_upper, torch.tensor([0.5]), atol=1e-5)

    lower, upper = result.concretize()
    # For point interval, should be tight
    assert torch.allclose(lower, torch.tensor([0.5]), atol=1e-5)
    assert torch.allclose(upper, torch.tensor([0.5]), atol=1e-5)


def test_cos_zero_width_falling() -> None:
    """Test cos on a point interval in falling region."""
    # Region: x ∈ [π/6, π/6]
    # cos(π/6) = √3/2 ≈ 0.866
    # This is in the falling region [0, π] where derivative is negative
    region = HyperRectangle(lower=torch.tensor([math.pi / 6]), upper=torch.tensor([math.pi / 6]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    # For point interval, should be tight
    expected = math.cos(math.pi / 6)  # ≈ 0.866
    assert torch.allclose(lower, torch.tensor([expected]), atol=1e-5)
    assert torch.allclose(upper, torch.tensor([expected]), atol=1e-5)


def test_cos_zero_width_rising() -> None:
    """Test cos on a point interval in rising region."""
    # Region: x ∈ [3π/2, 3π/2]
    # cos(3π/2) ≈ 0
    # This is in the rising region [π, 2π] where derivative is positive
    region = HyperRectangle(lower=torch.tensor([3 * math.pi / 2]), upper=torch.tensor([3 * math.pi / 2]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    # For point interval, should be tight
    expected = math.cos(3 * math.pi / 2)  # ≈ 0
    assert torch.allclose(lower, torch.tensor([expected]), atol=1e-5)
    assert torch.allclose(upper, torch.tensor([expected]), atol=1e-5)


def test_cos_concave_region_first_half() -> None:
    """Test cos on concave region [0, π] - first half where cos is decreasing and concave."""
    # Region: x ∈ [0, π/2]
    # In this region: cos is decreasing (derivative < 0) and concave (second derivative < 0)
    # cos(0) = 1, cos(π/2) ≈ 0
    # For concave functions:
    # - Secant line is BELOW the curve (lower bound)
    # - Tangent line is ABOVE the curve (upper bound)
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([math.pi / 2]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()

    # Actual range: [0, 1]
    # Relaxation should contain this range
    actual_lower = math.cos(math.pi / 2)  # ≈ 0
    actual_upper = math.cos(0.0)  # = 1

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"

    # Verify bounds are sound: lower <= upper everywhere
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_cos_concave_region_second_half() -> None:
    """Test cos on concave region [π/2, π] - still concave but now increasing."""
    # Region: x ∈ [π/2, π]
    # cos is increasing (derivative > 0 in this range) and concave
    # cos(π/2) ≈ 0, cos(π) = -1
    region = HyperRectangle(lower=torch.tensor([math.pi / 2]), upper=torch.tensor([math.pi]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()

    # Actual range: [-1, 0]
    actual_lower = math.cos(math.pi)  # = -1
    actual_upper = math.cos(math.pi / 2)  # ≈ 0

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_cos_convex_region_first_half() -> None:
    """Test cos on convex region [π, 3π/2] - first half where cos is decreasing and convex."""
    # Region: x ∈ [π, 3π/2]
    # In this region: cos is decreasing and convex (second derivative > 0)
    # cos(π) = -1, cos(3π/2) ≈ 0
    # For convex functions:
    # - Tangent line is BELOW the curve (lower bound)
    # - Secant line is ABOVE the curve (upper bound)
    region = HyperRectangle(lower=torch.tensor([math.pi]), upper=torch.tensor([3 * math.pi / 2]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()

    # Actual range: [-1, 0]
    actual_lower = math.cos(math.pi)  # = -1
    actual_upper = math.cos(3 * math.pi / 2)  # ≈ 0

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_cos_convex_region_second_half() -> None:
    """Test cos on convex region [3π/2, 2π] - second half where cos is increasing and convex."""
    # Region: x ∈ [3π/2, 2π]
    # cos is increasing and convex
    # cos(3π/2) ≈ 0, cos(2π) = 1
    region = HyperRectangle(lower=torch.tensor([3 * math.pi / 2]), upper=torch.tensor([2 * math.pi]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()

    # Actual range: [0, 1]
    actual_lower = math.cos(3 * math.pi / 2)  # ≈ 0
    actual_upper = math.cos(2 * math.pi)  # = 1

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_cos_across_maximum_at_zero() -> None:
    """Test cos crossing maximum at 0."""
    # Region: x ∈ [-π/4, π/4]
    # Crosses maximum at x=0 where cos(0) = 1
    # This is a concave region containing the global maximum
    region = HyperRectangle(lower=torch.tensor([-math.pi / 4]), upper=torch.tensor([math.pi / 4]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()

    # Actual range: [cos(π/4), 1] ≈ [0.707, 1]
    actual_lower = math.cos(math.pi / 4)  # ≈ 0.707 (by symmetry)
    actual_upper = math.cos(0.0)  # = 1 (maximum)

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_cos_across_maximum_at_2pi() -> None:
    """Test cos crossing maximum at 2π."""
    # Region: x ∈ [7π/4, 9π/4]  (which is [2π - π/4, 2π + π/4])
    # Crosses maximum at x=2π where cos(2π) = 1
    region = HyperRectangle(lower=torch.tensor([7 * math.pi / 4]), upper=torch.tensor([9 * math.pi / 4]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()

    # By symmetry around 2π, minimum value is at endpoints
    actual_lower = math.cos(math.pi / 4)  # ≈ 0.707
    actual_upper = 1.0  # Maximum at 2π

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_cos_across_minimum_at_pi() -> None:
    """Test cos crossing minimum at π."""
    # Region: x ∈ [3π/4, 5π/4]  (which is [π - π/4, π + π/4])
    # Crosses minimum at x=π where cos(π) = -1
    region = HyperRectangle(lower=torch.tensor([3 * math.pi / 4]), upper=torch.tensor([5 * math.pi / 4]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()

    # Actual range: [-1, cos(3π/4)] where cos(3π/4) ≈ -0.707
    actual_lower = -1.0  # Minimum at π
    actual_upper = math.cos(3 * math.pi / 4)  # ≈ -0.707

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_cos_strictly_decreasing_small() -> None:
    """Test cos on a small strictly decreasing interval."""
    # Region: x ∈ [π/6, π/3]
    # cos is strictly decreasing in [0, π]
    # cos(π/6) = √3/2 ≈ 0.866, cos(π/3) = 0.5
    region = HyperRectangle(lower=torch.tensor([math.pi / 6]), upper=torch.tensor([math.pi / 3]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()

    # Actual range: [cos(π/3), cos(π/6)] = [0.5, √3/2]
    actual_lower = math.cos(math.pi / 3)  # = 0.5
    actual_upper = math.cos(math.pi / 6)  # ≈ 0.866

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_cos_strictly_increasing_small() -> None:
    """Test cos on a small strictly increasing interval."""
    # Region: x ∈ [4π/3, 5π/3]
    # cos is strictly increasing in [π, 2π]
    # cos(4π/3) = -0.5, cos(5π/3) = 0.5
    region = HyperRectangle(lower=torch.tensor([4 * math.pi / 3]), upper=torch.tensor([5 * math.pi / 3]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()

    # Actual range: [cos(4π/3), cos(5π/3)] = [-0.5, 0.5]
    actual_lower = math.cos(4 * math.pi / 3)  # = -0.5
    actual_upper = math.cos(5 * math.pi / 3)  # = 0.5

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_cos_mixed_convexity_full_wave() -> None:
    """Test cos on interval spanning both concave and convex regions."""
    # Region: x ∈ [π/2, 3π/2]
    # Spans concave region [π/2, π] and convex region [π, 3π/2]
    # Crosses minimum at π
    region = HyperRectangle(lower=torch.tensor([math.pi / 2]), upper=torch.tensor([3 * math.pi / 2]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()

    # Actual range: [-1, 0] (endpoints at 0, minimum at -1)
    actual_lower = -1.0  # Minimum at π
    actual_upper = 0.0  # Max at endpoints (cos(π/2) ≈ 0, cos(3π/2) ≈ 0)

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_cos_negative_region_decreasing() -> None:
    """Test cos on a negative interval where cos is increasing."""
    # Region: x ∈ [-π, -π/2]
    # cos is increasing in [-π, 0], and convex in [-π, 0] (since concave in [0, π] and cos is even about origin)
    # cos(-π) = -1, cos(-π/2) ≈ 0
    region = HyperRectangle(lower=torch.tensor([-math.pi]), upper=torch.tensor([-math.pi / 2]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()

    # Actual range: [-1, 0]
    actual_lower = math.cos(-math.pi)  # = -1
    actual_upper = math.cos(-math.pi / 2)  # ≈ 0

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"


def test_cos_large_interval_multiple_periods() -> None:
    """Test cos on interval spanning multiple periods."""
    # Region: x ∈ [0, 3π]
    # Spans 1.5 periods, includes maximum at 0 (and 2π) and minimum at π
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([3 * math.pi]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPCos()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()

    # Actual range: [-1, 1] (full range of cosine)
    actual_lower = -1.0
    actual_upper = 1.0

    assert torch.all(lower <= actual_lower + 0.01), f"Lower bound {lower} should be <= {actual_lower}"
    assert torch.all(upper >= actual_upper - 0.01), f"Upper bound {upper} should be >= {actual_upper}"
    assert torch.all(lower <= upper + 1e-6), "Lower bound must be <= upper bound"
