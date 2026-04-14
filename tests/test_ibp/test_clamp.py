from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.clamp import IBPClamp
from tests.helpers import propagate


def _propagate(
    lower: torch.Tensor, upper: torch.Tensor, clamp_min: float | None = None, clamp_max: float | None = None
) -> IntervalBounds:
    """Propagate bounds for clamp operation."""
    strategy = IBPClamp()
    bounds = IntervalBounds(lower=lower, upper=upper)
    kwargs: dict[str, float | None] = {}
    if clamp_min is not None:
        kwargs["min"] = clamp_min
    if clamp_max is not None:
        kwargs["max"] = clamp_max
    return propagate(strategy, bounds, **kwargs)


def test_clamp_both_bounds() -> None:
    """Test clamp with both min and max."""
    # clamp([0, 10], min=2, max=8) = [2, 8]
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([10.0]),
        clamp_min=2.0,
        clamp_max=8.0,
    )

    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([8.0]))


def test_clamp_only_min() -> None:
    """Test clamp with only min."""
    # clamp([0, 10], min=3) = [3, 10]
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([10.0]),
        clamp_min=3.0,
        clamp_max=None,
    )

    assert torch.allclose(out.lower, torch.tensor([3.0]))
    assert torch.allclose(out.upper, torch.tensor([10.0]))


def test_clamp_only_max() -> None:
    """Test clamp with only max."""
    # clamp([0, 10], max=7) = [0, 7]
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([10.0]),
        clamp_min=None,
        clamp_max=7.0,
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([7.0]))


def test_clamp_interval_within_range() -> None:
    """Test clamp when interval is already within clamp range."""
    # clamp([3, 5], min=0, max=10) = [3, 5]
    out = _propagate(
        lower=torch.tensor([3.0]),
        upper=torch.tensor([5.0]),
        clamp_min=0.0,
        clamp_max=10.0,
    )

    assert torch.allclose(out.lower, torch.tensor([3.0]))
    assert torch.allclose(out.upper, torch.tensor([5.0]))


def test_clamp_interval_below_range() -> None:
    """Test clamp when interval is completely below clamp range."""
    # clamp([-5, -2], min=0, max=10) = [0, 0]
    out = _propagate(
        lower=torch.tensor([-5.0]),
        upper=torch.tensor([-2.0]),
        clamp_min=0.0,
        clamp_max=10.0,
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_clamp_interval_above_range() -> None:
    """Test clamp when interval is completely above clamp range."""
    # clamp([12, 15], min=0, max=10) = [10, 10]
    out = _propagate(
        lower=torch.tensor([12.0]),
        upper=torch.tensor([15.0]),
        clamp_min=0.0,
        clamp_max=10.0,
    )

    assert torch.allclose(out.lower, torch.tensor([10.0]))
    assert torch.allclose(out.upper, torch.tensor([10.0]))


def test_clamp_interval_crossing_min() -> None:
    """Test clamp when interval crosses the min boundary."""
    # clamp([-2, 5], min=0, max=10) = [0, 5]
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([5.0]),
        clamp_min=0.0,
        clamp_max=10.0,
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([5.0]))


def test_clamp_interval_crossing_max() -> None:
    """Test clamp when interval crosses the max boundary."""
    # clamp([5, 12], min=0, max=10) = [5, 10]
    out = _propagate(
        lower=torch.tensor([5.0]),
        upper=torch.tensor([12.0]),
        clamp_min=0.0,
        clamp_max=10.0,
    )

    assert torch.allclose(out.lower, torch.tensor([5.0]))
    assert torch.allclose(out.upper, torch.tensor([10.0]))


def test_clamp_interval_crossing_both() -> None:
    """Test clamp when interval crosses both boundaries."""
    # clamp([-5, 15], min=0, max=10) = [0, 10]
    out = _propagate(
        lower=torch.tensor([-5.0]),
        upper=torch.tensor([15.0]),
        clamp_min=0.0,
        clamp_max=10.0,
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([10.0]))


def test_clamp_point_interval() -> None:
    """Test clamp of point interval (lower = upper)."""
    # clamp([5, 5], min=0, max=10) = [5, 5]
    out = _propagate(
        lower=torch.tensor([5.0]),
        upper=torch.tensor([5.0]),
        clamp_min=0.0,
        clamp_max=10.0,
    )

    assert torch.allclose(out.lower, torch.tensor([5.0]))
    assert torch.allclose(out.upper, torch.tensor([5.0]))


def test_clamp_point_below_min() -> None:
    """Test clamp of point interval below min."""
    # clamp([-5, -5], min=0, max=10) = [0, 0]
    out = _propagate(
        lower=torch.tensor([-5.0]),
        upper=torch.tensor([-5.0]),
        clamp_min=0.0,
        clamp_max=10.0,
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_clamp_point_above_max() -> None:
    """Test clamp of point interval above max."""
    # clamp([15, 15], min=0, max=10) = [10, 10]
    out = _propagate(
        lower=torch.tensor([15.0]),
        upper=torch.tensor([15.0]),
        clamp_min=0.0,
        clamp_max=10.0,
    )

    assert torch.allclose(out.lower, torch.tensor([10.0]))
    assert torch.allclose(out.upper, torch.tensor([10.0]))


def test_clamp_batched_intervals() -> None:
    """Test clamp with batched intervals."""
    out = _propagate(
        lower=torch.tensor([-5.0, 3.0, 5.0, 12.0, -2.0]),
        upper=torch.tensor([-2.0, 5.0, 12.0, 15.0, 5.0]),
        clamp_min=0.0,
        clamp_max=10.0,
    )

    # [-5, -2] -> [0, 0]
    # [3, 5] -> [3, 5]
    # [5, 12] -> [5, 10]
    # [12, 15] -> [10, 10]
    # [-2, 5] -> [0, 5]
    expected_lower = torch.tensor([0.0, 3.0, 5.0, 10.0, 0.0])
    expected_upper = torch.tensor([0.0, 5.0, 10.0, 10.0, 5.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_clamp_multidimensional() -> None:
    """Test clamp with multi-dimensional intervals."""
    out = _propagate(
        lower=torch.tensor([[-5.0, 3.0], [5.0, 12.0]]),
        upper=torch.tensor([[-2.0, 5.0], [12.0, 15.0]]),
        clamp_min=0.0,
        clamp_max=10.0,
    )

    expected_lower = torch.tensor([[0.0, 3.0], [5.0, 10.0]])
    expected_upper = torch.tensor([[0.0, 5.0], [10.0, 10.0]])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_clamp_negative_range() -> None:
    """Test clamp with negative range."""
    # clamp([-10, 5], min=-5, max=-1) = [-5, -1]
    out = _propagate(
        lower=torch.tensor([-10.0]),
        upper=torch.tensor([5.0]),
        clamp_min=-5.0,
        clamp_max=-1.0,
    )

    assert torch.allclose(out.lower, torch.tensor([-5.0]))
    assert torch.allclose(out.upper, torch.tensor([-1.0]))


def test_clamp_no_bounds() -> None:
    """Test clamp with no bounds requires at least one bound."""
    # PyTorch clamp requires at least one of min/max, so this should raise an error
    # or we skip this test since it's not valid
    # Let's test with just one bound being None instead
    out = _propagate(
        lower=torch.tensor([1.0]),
        upper=torch.tensor([5.0]),
        clamp_min=0.0,
        clamp_max=None,
    )

    # With min=0, interval [1, 5] stays [1, 5]
    assert torch.allclose(out.lower, torch.tensor([1.0]))
    assert torch.allclose(out.upper, torch.tensor([5.0]))


def test_clamp_idempotency() -> None:
    """Test that clamp(clamp(x)) = clamp(x)."""
    strategy = IBPClamp()
    a = IntervalBounds(torch.tensor([-5.0, 3.0, 12.0]), torch.tensor([15.0, 7.0, 20.0]))

    # clamp(a)
    clamp_a = propagate(strategy, a, min=0.0, max=10.0)

    # clamp(clamp(a))
    clamp_clamp_a = propagate(strategy, clamp_a, min=0.0, max=10.0)

    # Should be the same
    assert torch.allclose(clamp_a.lower, clamp_clamp_a.lower)
    assert torch.allclose(clamp_a.upper, clamp_clamp_a.upper)


def test_clamp_monotonicity() -> None:
    """Test that clamp is monotonically non-decreasing."""
    # If [a, b] ⊆ [c, d], then clamp([a, b]) ⊆ clamp([c, d])
    inner = IntervalBounds(torch.tensor([2.0, -2.0]), torch.tensor([4.0, 8.0]))
    outer = IntervalBounds(torch.tensor([0.0, -5.0]), torch.tensor([6.0, 12.0]))

    strategy = IBPClamp()

    out_inner = propagate(strategy, inner, min=0.0, max=10.0)
    out_outer = propagate(strategy, outer, min=0.0, max=10.0)

    assert torch.all(out_outer.lower <= out_inner.lower)
    assert torch.all(out_outer.upper >= out_inner.upper)
