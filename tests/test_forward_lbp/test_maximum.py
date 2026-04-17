from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.forward_lbp.pairwise import (
    ForwardLBPMaximum,
)
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


def _check_soundness(result: LinearBounds, region: HyperRectangle, fn, n_samples: int = 200) -> None:
    """Check that the linear relaxation is a sound over-approximation via sampling."""
    lower, upper = result.concretize()
    dim = region.lower.numel()
    samples = region.lower + (region.upper - region.lower) * torch.rand(n_samples, dim)
    for x in samples:
        true_val = fn(x)
        assert torch.all(lower <= true_val + 1e-6), f"Lower bound {lower} > true value {true_val}"
        assert torch.all(upper >= true_val - 1e-6), f"Upper bound {upper} < true value {true_val}"


def test_maximum_abstract_abstract_preserves_linear() -> None:
    """Test element-wise maximum of two abstract bounds preserves linear structure."""
    region = HyperRectangle(lower=torch.tensor([1.0, 2.0]), upper=torch.tensor([3.0, 4.0]))

    bounds_a = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        bias_lower=torch.tensor([0.0, 0.0]),
        linear_upper=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        bias_upper=torch.tensor([0.0, 0.0]),
    )
    bounds_b = LinearBounds(
        regions=[region],
        linear_lower=torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
        bias_lower=torch.tensor([0.0, 0.0]),
        linear_upper=torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
        bias_upper=torch.tensor([0.0, 0.0]),
    )

    strategy = ForwardLBPMaximum()
    result = propagate(strategy, bounds_a, bounds_b)

    # Should preserve linear structure (not concretize)
    assert len(result.linear_lowers) > 0

    # Soundness check
    lower, upper = result.concretize()
    # First element: max(x0, 0) where x0 ∈ [1, 3] → [1, 3]
    # Second element: max(0, x1) where x1 ∈ [2, 4] → [2, 4]
    assert torch.all(lower <= torch.tensor([1.0, 2.0]) + 1e-6)
    assert torch.all(upper >= torch.tensor([3.0, 4.0]) - 1e-6)


def test_maximum_abstract_constant_positive() -> None:
    """Test element-wise maximum with a positive constant."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([5.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMaximum()
    result = propagate(strategy, bounds, 2.0)

    # Should preserve linear structure
    assert len(result.linear_lowers) > 0

    _check_soundness(result, region, lambda x: torch.maximum(x, torch.tensor(2.0)))


def test_maximum_abstract_constant_below_range() -> None:
    """Test maximum with constant below the input range — identity passthrough."""
    region = HyperRectangle(lower=torch.tensor([3.0]), upper=torch.tensor([7.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMaximum()
    result = propagate(strategy, bounds, 1.0)

    # a dominates: max(x, 1) = x when x >= 3 > 1
    # Should be identity passthrough with linear structure
    assert len(result.linear_lowers) > 0
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([3.0]))
    assert torch.allclose(upper, torch.tensor([7.0]))


def test_maximum_abstract_constant_above_range() -> None:
    """Test maximum with constant above the input range — constant output."""
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([3.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMaximum()
    result = propagate(strategy, bounds, 5.0)

    # b dominates: max(x, 5) = 5 when x <= 3 < 5
    lower, upper = result.concretize()
    assert torch.allclose(lower, torch.tensor([5.0]))
    assert torch.allclose(upper, torch.tensor([5.0]))


def test_maximum_abstract_constant_tensor() -> None:
    """Test maximum with a tensor constant."""
    region = HyperRectangle(lower=torch.tensor([0.0, 1.0]), upper=torch.tensor([4.0, 5.0]))
    bounds = _make_linear_bounds(region)

    constant = torch.tensor([2.0, 3.0])
    strategy = ForwardLBPMaximum()
    result = propagate(strategy, bounds, constant)

    _check_soundness(result, region, lambda x: torch.maximum(x, constant))


def test_maximum_constant_abstract() -> None:
    """Test maximum with constant as first operand (commutative)."""
    region = HyperRectangle(lower=torch.tensor([1.0]), upper=torch.tensor([6.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMaximum()
    result = propagate(strategy, 3.0, bounds)

    _check_soundness(result, region, lambda x: torch.maximum(torch.tensor(3.0), x))


def test_maximum_negative_values() -> None:
    """Test maximum with negative values."""
    region = HyperRectangle(lower=torch.tensor([-5.0]), upper=torch.tensor([-1.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMaximum()
    result = propagate(strategy, bounds, -3.0)

    _check_soundness(result, region, lambda x: torch.maximum(x, torch.tensor(-3.0)))


def test_maximum_crossing_tightness() -> None:
    """Test that the relaxation is reasonably tight in the crossing case."""
    region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([4.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMaximum()
    result = propagate(strategy, bounds, 2.0)

    lower, upper = result.concretize()
    # max(x, 2) for x ∈ [0, 4]: true range is [2, 4]
    # Lower bound should be <= 2, upper should be >= 4
    assert lower.item() <= 2.0 + 1e-6
    assert upper.item() >= 4.0 - 1e-6
    # Gap should be less than naive interval (which would be [0, 4] gap = 4)
    # With linear relaxation gap should be tighter
    gap = (upper - lower).item()
    assert gap < 4.0


def test_maximum_abstract_abstract_same_bounds() -> None:
    """Test maximum of two identical abstract bounds — sound over-approximation of identity."""
    region = HyperRectangle(lower=torch.tensor([1.0, 2.0]), upper=torch.tensor([3.0, 4.0]))
    bounds = _make_linear_bounds(region)

    strategy = ForwardLBPMaximum()
    result = propagate(strategy, bounds, bounds)

    # max(x, x) = x — relaxation treats inputs as independent over the box,
    # so it's sound but not tight for this degenerate case.
    lower, upper = result.concretize()
    assert torch.all(lower <= region.lower + 1e-5)
    assert torch.all(upper >= region.upper - 1e-5)
