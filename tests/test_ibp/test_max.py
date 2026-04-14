from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.max import IBPMax
from bound_propagation.propagation.ibp.min import IBPMin
from bound_propagation.propagation.ibp.neg import IBPNeg
from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor, dim: int | None = None) -> IntervalBounds:
    """Propagate bounds for max operation."""
    strategy = IBPMax()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds, dim=dim)
    shape_list.pop(dim)
    return tuple(shape_list) if shape_list else ()


def test_max_positive_intervals() -> None:
    """Test max of positive intervals."""
    # max([5, 10], [3, 8], [7, 12]) = [7, 12]
    out = _propagate(
        lower=torch.tensor([[5.0, 3.0, 7.0]]),
        upper=torch.tensor([[10.0, 8.0, 12.0]]),
        dim=1,
    )

    # Max lower is max of lowers: max(5, 3, 7) = 7
    # Max upper is max of uppers: max(10, 8, 12) = 12
    expected_lower = torch.tensor([7.0])
    expected_upper = torch.tensor([12.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_max_negative_intervals() -> None:
    """Test max of negative intervals."""
    # max([-5, -2], [-8, -3], [-4, -1]) = [-4, -1]
    out = _propagate(
        lower=torch.tensor([[-5.0, -8.0, -4.0]]),
        upper=torch.tensor([[-2.0, -3.0, -1.0]]),
        dim=1,
    )

    # Max lower is max of lowers: max(-5, -8, -4) = -4
    # Max upper is max of uppers: max(-2, -3, -1) = -1
    expected_lower = torch.tensor([-4.0])
    expected_upper = torch.tensor([-1.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_max_mixed_intervals() -> None:
    """Test max of mixed sign intervals."""
    # max([-2, 3], [1, 5], [-1, 2]) = [1, 5]
    out = _propagate(
        lower=torch.tensor([[-2.0, 1.0, -1.0]]),
        upper=torch.tensor([[3.0, 5.0, 2.0]]),
        dim=1,
    )

    # Max lower is max of lowers: max(-2, 1, -1) = 1
    # Max upper is max of uppers: max(3, 5, 2) = 5
    expected_lower = torch.tensor([1.0])
    expected_upper = torch.tensor([5.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_max_point_intervals() -> None:
    """Test max of point intervals (lower = upper)."""
    # max([5, 5], [3, 3], [7, 7]) = [7, 7]
    out = _propagate(
        lower=torch.tensor([[5.0, 3.0, 7.0]]),
        upper=torch.tensor([[5.0, 3.0, 7.0]]),
        dim=1,
    )

    expected_max = torch.tensor([7.0])

    assert torch.allclose(out.lower, expected_max)
    assert torch.allclose(out.upper, expected_max)


def test_max_overlapping_intervals() -> None:
    """Test max with overlapping intervals."""
    # max([2, 5], [3, 6], [1, 4]) = [3, 6]
    out = _propagate(
        lower=torch.tensor([[2.0, 3.0, 1.0]]),
        upper=torch.tensor([[5.0, 6.0, 4.0]]),
        dim=1,
    )

    # Max lower: max(2, 3, 1) = 3
    # Max upper: max(5, 6, 4) = 6
    expected_lower = torch.tensor([3.0])
    expected_upper = torch.tensor([6.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_max_multi_batch() -> None:
    """Test max with multiple batches."""
    out = _propagate(
        lower=torch.tensor([[5.0, 3.0], [2.0, 6.0]]),
        upper=torch.tensor([[10.0, 8.0], [4.0, 9.0]]),
        dim=1,
    )

    # Batch 0: max([5, 10], [3, 8]) = [5, 10]
    # Batch 1: max([2, 4], [6, 9]) = [6, 9]
    expected_lower = torch.tensor([5.0, 6.0])
    expected_upper = torch.tensor([10.0, 9.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_max_dim_0() -> None:
    """Test max across dimension 0."""
    out = _propagate(
        lower=torch.tensor([[5.0, 3.0], [2.0, 6.0]]),
        upper=torch.tensor([[10.0, 8.0], [4.0, 9.0]]),
        dim=0,
    )

    # Max along dim 0:
    # Column 0: max([5, 10], [2, 4]) = [5, 10]
    # Column 1: max([3, 8], [6, 9]) = [6, 9]
    expected_lower = torch.tensor([5.0, 6.0])
    expected_upper = torch.tensor([10.0, 9.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_max_entire_tensor() -> None:
    """Test max without specifying dim (max over entire tensor)."""
    out = _propagate(
        lower=torch.tensor([[5.0, 3.0], [2.0, 6.0]]),
        upper=torch.tensor([[10.0, 8.0], [4.0, 9.0]]),
        dim=None,
    )

    # Max of all lowers: max(5, 3, 2, 6) = 6
    # Max of all uppers: max(10, 8, 4, 9) = 10
    expected_lower = torch.tensor(6.0)
    expected_upper = torch.tensor(10.0)

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_max_monotonicity() -> None:
    """Test that max preserves monotonicity: if [a, b] ⊆ [c, d], then max([a, b]) ⊆ max([c, d])."""
    inner = IntervalBounds(torch.tensor([[3.0, 4.0]]), torch.tensor([[5.0, 6.0]]))
    outer = IntervalBounds(torch.tensor([[2.0, 3.0]]), torch.tensor([[6.0, 7.0]]))

    strategy = IBPMax()

    out_inner = propagate(strategy, inner, dim=1)
    out_outer = propagate(strategy, outer, dim=1)

    assert torch.all(out_outer.lower <= out_inner.lower)
    assert torch.all(out_outer.upper >= out_inner.upper)


def test_max_idempotency() -> None:
    """Test that max(max(a)) = max(a) for single-element dimension."""
    # Create interval that when reduced has only one element
    lower = torch.tensor([[[2.0]]])
    upper = torch.tensor([[[5.0]]])

    out1 = _propagate(lower, upper, dim=1)
    out2 = _propagate(out1.lower.unsqueeze(1), out1.upper.unsqueeze(1), dim=1)

    assert torch.allclose(out1.lower, out2.lower)
    assert torch.allclose(out1.upper, out2.upper)


def test_max_comparison_with_individual() -> None:
    """Test that max of intervals is greater than or equal to each individual interval."""
    lower = torch.tensor([[5.0, 3.0, 7.0]])
    upper = torch.tensor([[10.0, 8.0, 12.0]])

    out = _propagate(lower, upper, dim=1)

    # Max should be >= each individual interval bound
    assert out.lower >= lower.max()
    assert out.upper >= upper.max()


def test_max_dimension_reduction() -> None:
    """Test that max reduces the dimensionality correctly."""
    out = _propagate(
        lower=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        upper=torch.tensor([[[2.0, 3.0], [5.0, 6.0]]]),
        dim=1,
    )

    # Should reduce from (1, 2, 2) to (1, 2)
    assert out.lower.shape == (1, 2)
    assert out.upper.shape == (1, 2)


def test_max_min_duality() -> None:
    """Test that max(-x) = -min(x)."""
    a = IntervalBounds(torch.tensor([[2.0, 3.0, 1.0]]), torch.tensor([[5.0, 6.0, 4.0]]))

    # max(a)
    max_strategy = IBPMax()
    max_a = propagate(max_strategy, a, dim=1)

    # -a
    neg_strategy = IBPNeg()
    neg_a = propagate(neg_strategy, a)

    # min(-a)
    min_strategy = IBPMin()
    min_neg_a = propagate(min_strategy, neg_a, dim=1)

    # -min(-a)
    neg_min_neg_a = propagate(neg_strategy, min_neg_a)

    # max(a) should equal -min(-a)
    assert torch.allclose(max_a.lower, neg_min_neg_a.lower)
    assert torch.allclose(max_a.upper, neg_min_neg_a.upper)
