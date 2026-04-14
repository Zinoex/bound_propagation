from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.log import ForwardLBPLog
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


def test_log_positive_interval() -> None:
    """Test log on a positive interval."""
    # Region: x ∈ [1, 4]
    # log([1, 4]) = [0, ln(4)] ≈ [0, 1.386]
    # log is concave; this implementation uses secant for the lower bound and
    # a midpoint tangent for the upper bound (valid but looser than endpoint tangents).
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([4.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPLog()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    # log(1) = 0, log(4) ≈ 1.386
    assert torch.all(lower >= -0.1)
    assert torch.all(lower <= 0.1)
    assert torch.all(upper >= 1.35)
    assert torch.all(upper <= 1.55)


def test_log_small_positive_interval() -> None:
    """Test log on a small positive interval."""
    # Region: x ∈ [0.5, 1]
    # log([0.5, 1]) = [ln(0.5), 0] ≈ [-0.693, 0]
    region = HyperRectangle(lower=torch.tensor([0.5]), upper=torch.tensor([1.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPLog()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    # log(0.5) ≈ -0.693, log(1) = 0
    assert torch.all(lower >= -0.7)
    assert torch.all(lower <= -0.68)
    assert torch.all(upper >= -0.1)
    assert torch.all(upper <= 0.1)


def test_log_large_interval() -> None:
    """Test log on a large interval."""
    # Region: x ∈ [2, 10]
    # log([2, 10]) = [ln(2), ln(10)] ≈ [0.693, 2.303]
    region = HyperRectangle(lower=torch.tensor([2.0]), upper=torch.tensor([10.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPLog()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    assert torch.all(lower >= 0.65)
    assert torch.all(lower <= 0.75)
    assert torch.all(upper >= 2.25)
    assert torch.all(upper <= 2.50)


def test_log_point_at_e() -> None:
    """Test log at Euler's number."""
    # Region: x ∈ [e, e] ≈ [2.718, 2.718]
    # log([e, e]) = [1, 1]
    e = torch.tensor([2.71828])
    region = HyperRectangle(lower=e, upper=e)
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPLog()
    result = propagate(strategy, bounds)

    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([1.0]), atol=1e-3)
    assert torch.allclose(upper, torch.tensor([1.0]), atol=1e-3)
