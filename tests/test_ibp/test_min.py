from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.min import IBPMin
from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor, dim: int | None = None) -> IntervalBounds:
    """Propagate bounds for min operation."""
    strategy = IBPMin()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds, dim=dim)


def test_min_positive_intervals() -> None:
    """Test min of positive intervals."""
    # min([5, 10], [3, 8], [7, 12]) = [3, 8]
    out = _propagate(
        lower=torch.tensor([[5.0, 3.0, 7.0]]),
        upper=torch.tensor([[10.0, 8.0, 12.0]]),
        dim=1,
    )

    # Min lower is min of lowers: min(5, 3, 7) = 3
    # Min upper is min of uppers: min(10, 8, 12) = 8
    expected_lower = torch.tensor([3.0])
    expected_upper = torch.tensor([8.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_min_negative_intervals() -> None:
    """Test min of negative intervals."""
    # min([-5, -2], [-8, -3], [-4, -1]) = [-8, -3]
    out = _propagate(
        lower=torch.tensor([[-5.0, -8.0, -4.0]]),
        upper=torch.tensor([[-2.0, -3.0, -1.0]]),
        dim=1,
    )

    # Min lower is min of lowers: min(-5, -8, -4) = -8
    # Min upper is min of uppers: min(-2, -3, -1) = -3
    expected_lower = torch.tensor([-8.0])
    expected_upper = torch.tensor([-3.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_min_mixed_intervals() -> None:
    """Test min of mixed sign intervals."""
    # min([-2, 3], [1, 5], [-1, 2]) = [-2, 2]
    out = _propagate(
        lower=torch.tensor([[-2.0, 1.0, -1.0]]),
        upper=torch.tensor([[3.0, 5.0, 2.0]]),
        dim=1,
    )

    # Min lower is min of lowers: min(-2, 1, -1) = -2
    # Min upper is min of uppers: min(3, 5, 2) = 2
    expected_lower = torch.tensor([-2.0])
    expected_upper = torch.tensor([2.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_min_point_intervals() -> None:
    """Test min of point intervals (lower = upper)."""
    # min([5, 5], [3, 3], [7, 7]) = [3, 3]
    out = _propagate(
        lower=torch.tensor([[5.0, 3.0, 7.0]]),
        upper=torch.tensor([[5.0, 3.0, 7.0]]),
        dim=1,
    )

    expected_min = torch.tensor([3.0])

    assert torch.allclose(out.lower, expected_min)
    assert torch.allclose(out.upper, expected_min)


def test_min_overlapping_intervals() -> None:
    """Test min with overlapping intervals."""
    # min([2, 5], [3, 6], [1, 4]) = [1, 4]
    out = _propagate(
        lower=torch.tensor([[2.0, 3.0, 1.0]]),
        upper=torch.tensor([[5.0, 6.0, 4.0]]),
        dim=1,
    )

    # Min lower: min(2, 3, 1) = 1
    # Min upper: min(5, 6, 4) = 4
    expected_lower = torch.tensor([1.0])
    expected_upper = torch.tensor([4.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_min_multi_batch() -> None:
    """Test min with multiple batches."""
    out = _propagate(
        lower=torch.tensor([[5.0, 3.0], [2.0, 6.0]]),
        upper=torch.tensor([[10.0, 8.0], [4.0, 9.0]]),
        dim=1,
    )

    # Batch 0: min([5, 10], [3, 8]) = [3, 8]
    # Batch 1: min([2, 4], [6, 9]) = [2, 4]
    expected_lower = torch.tensor([3.0, 2.0])
    expected_upper = torch.tensor([8.0, 4.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_min_dim_0() -> None:
    """Test min across dimension 0."""
    out = _propagate(
        lower=torch.tensor([[5.0, 3.0], [2.0, 6.0]]),
        upper=torch.tensor([[10.0, 8.0], [4.0, 9.0]]),
        dim=0,
    )

    # Min along dim 0:
    # Column 0: min([5, 10], [2, 4]) = [2, 4]
    # Column 1: min([3, 8], [6, 9]) = [3, 8]
    expected_lower = torch.tensor([2.0, 3.0])
    expected_upper = torch.tensor([4.0, 8.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_min_entire_tensor() -> None:
    """Test min without specifying dim (min over entire tensor)."""
    out = _propagate(
        lower=torch.tensor([[5.0, 3.0], [2.0, 6.0]]),
        upper=torch.tensor([[10.0, 8.0], [4.0, 9.0]]),
        dim=None,
    )

    # Min of all lowers: min(5, 3, 2, 6) = 2
    # Min of all uppers: min(10, 8, 4, 9) = 4
    expected_lower = torch.tensor(2.0)
    expected_upper = torch.tensor(4.0)

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_min_monotonicity() -> None:
    """Test that min preserves monotonicity: if [a, b] ⊆ [c, d], then min([a, b]) ⊆ min([c, d])."""
    inner = IntervalBounds(torch.tensor([[3.0, 4.0]]), torch.tensor([[5.0, 6.0]]))
    outer = IntervalBounds(torch.tensor([[2.0, 3.0]]), torch.tensor([[6.0, 7.0]]))

    strategy = IBPMin()

    out_inner = propagate(strategy, inner, dim=1)
    out_outer = propagate(strategy, outer, dim=1)

    assert torch.all(out_outer.lower <= out_inner.lower)
    assert torch.all(out_outer.upper >= out_inner.upper)


def test_min_idempotency() -> None:
    """Test that min(min(a)) = min(a) for single-element dimension."""
    # Create interval that when reduced has only one element
    lower = torch.tensor([[[2.0]]])
    upper = torch.tensor([[[5.0]]])

    out1 = _propagate(lower, upper, dim=1)
    out2 = _propagate(out1.lower.unsqueeze(1), out1.upper.unsqueeze(1), dim=1)

    assert torch.allclose(out1.lower, out2.lower)
    assert torch.allclose(out1.upper, out2.upper)


def test_min_comparison_with_individual() -> None:
    """Test that min of intervals is smaller than or equal to each individual interval."""
    lower = torch.tensor([[5.0, 3.0, 7.0]])
    upper = torch.tensor([[10.0, 8.0, 12.0]])

    out = _propagate(lower, upper, dim=1)

    # Min should be <= each individual interval bound
    assert out.lower <= lower.min()
    assert out.upper <= upper.min()


def test_min_dimension_reduction() -> None:
    """Test that min reduces the dimensionality correctly."""
    out = _propagate(
        lower=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        upper=torch.tensor([[[2.0, 3.0], [5.0, 6.0]]]),
        dim=1,
    )

    # Should reduce from (1, 2, 2) to (1, 2)
    assert out.lower.shape == (1, 2)
    assert out.upper.shape == (1, 2)
