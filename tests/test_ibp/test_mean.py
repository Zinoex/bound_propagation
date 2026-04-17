from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.reduction import IBPMean, IBPSum
from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor, dim: int = 0) -> IntervalBounds:
    """Propagate bounds for mean operation."""
    strategy = IBPMean()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds, dim=dim)


def test_mean_positive_interval() -> None:
    """Test mean of positive intervals."""
    # mean([1, 3], [2, 4], [3, 5]) along dim=1 = [(1+2+3)/3, (3+4+5)/3] = [2, 4]
    out = _propagate(
        lower=torch.tensor([[1.0, 2.0, 3.0]]),
        upper=torch.tensor([[3.0, 4.0, 5.0]]),
        dim=1,
    )

    expected_lower = (1.0 + 2.0 + 3.0) / 3  # 2.0
    expected_upper = (3.0 + 4.0 + 5.0) / 3  # 4.0

    assert torch.allclose(out.lower, torch.tensor([expected_lower]))
    assert torch.allclose(out.upper, torch.tensor([expected_upper]))


def test_mean_negative_interval() -> None:
    """Test mean of negative intervals."""
    # mean([-3, -1], [-6, -2], [-9, -3]) = [(-3-6-9)/3, (-1-2-3)/3] = [-6, -2]
    out = _propagate(
        lower=torch.tensor([[-3.0, -6.0, -9.0]]),
        upper=torch.tensor([[-1.0, -2.0, -3.0]]),
        dim=1,
    )

    expected_lower = (-3.0 + -6.0 + -9.0) / 3  # -6.0
    expected_upper = (-1.0 + -2.0 + -3.0) / 3  # -2.0

    assert torch.allclose(out.lower, torch.tensor([expected_lower]))
    assert torch.allclose(out.upper, torch.tensor([expected_upper]))


def test_mean_mixed_intervals() -> None:
    """Test mean of mixed sign intervals."""
    # mean([-2, 1], [0, 3], [-1, 2]) = [(-2+0-1)/3, (1+3+2)/3] = [-1, 2]
    out = _propagate(
        lower=torch.tensor([[-2.0, 0.0, -1.0]]),
        upper=torch.tensor([[1.0, 3.0, 2.0]]),
        dim=1,
    )

    expected_lower = (-2.0 + 0.0 + -1.0) / 3  # -1.0
    expected_upper = (1.0 + 3.0 + 2.0) / 3  # 2.0

    assert torch.allclose(out.lower, torch.tensor([expected_lower]))
    assert torch.allclose(out.upper, torch.tensor([expected_upper]))


def test_mean_point_intervals() -> None:
    """Test mean of point intervals (lower = upper)."""
    # mean([2, 2], [3, 3], [4, 4]) = [3, 3]
    out = _propagate(
        lower=torch.tensor([[2.0, 3.0, 4.0]]),
        upper=torch.tensor([[2.0, 3.0, 4.0]]),
        dim=1,
    )

    expected_mean = (2.0 + 3.0 + 4.0) / 3  # 3.0

    assert torch.allclose(out.lower, torch.tensor([expected_mean]))
    assert torch.allclose(out.upper, torch.tensor([expected_mean]))


def test_mean_multi_batch() -> None:
    """Test mean with multiple batches."""
    out = _propagate(
        lower=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        upper=torch.tensor([[2.0, 3.0], [5.0, 6.0]]),
        dim=1,
    )

    # Batch 0: [(1+2)/2, (2+3)/2] = [1.5, 2.5]
    # Batch 1: [(3+4)/2, (5+6)/2] = [3.5, 5.5]
    expected_lower = torch.tensor([1.5, 3.5])
    expected_upper = torch.tensor([2.5, 5.5])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_mean_dim_0() -> None:
    """Test mean across dimension 0."""
    out = _propagate(
        lower=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        upper=torch.tensor([[2.0, 3.0], [5.0, 6.0]]),
        dim=0,
    )

    # Mean along dim 0: [(1+3)/2, (2+4)/2], [(2+5)/2, (3+6)/2] = [2, 3], [3.5, 4.5]
    expected_lower = torch.tensor([2.0, 3.0])
    expected_upper = torch.tensor([3.5, 4.5])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_mean_single_element() -> None:
    """Test mean of single element (identity operation)."""
    out = _propagate(
        lower=torch.tensor([[2.0]]),
        upper=torch.tensor([[5.0]]),
        dim=1,
    )

    # Mean of single element is the element itself
    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([5.0]))


def test_mean_monotonicity() -> None:
    """Test that mean preserves monotonicity: if [a, b] ⊆ [c, d], then mean([a, b]) ⊆ mean([c, d])."""
    inner = IntervalBounds(torch.tensor([[2.0, 3.0]]), torch.tensor([[4.0, 5.0]]))
    outer = IntervalBounds(torch.tensor([[1.0, 2.0]]), torch.tensor([[5.0, 6.0]]))

    strategy = IBPMean()

    out_inner = propagate(strategy, inner, dim=1)
    out_outer = propagate(strategy, outer, dim=1)

    assert torch.all(out_outer.lower <= out_inner.lower)
    assert torch.all(out_outer.upper >= out_inner.upper)


def test_mean_narrowing() -> None:
    """Test that mean may narrow intervals (reduces variance)."""
    # If all intervals are [a, b], then mean is also [a, b]
    lower = torch.tensor([[1.0, 1.0, 1.0]])
    upper = torch.tensor([[2.0, 2.0, 2.0]])

    out = _propagate(lower, upper, dim=1)

    # mean([1, 2], [1, 2], [1, 2]) = [1, 2]
    assert torch.allclose(out.lower, torch.tensor([1.0]))
    assert torch.allclose(out.upper, torch.tensor([2.0]))


def test_mean_dimension_reduction() -> None:
    """Test that mean reduces the dimensionality correctly."""
    out = _propagate(
        lower=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        upper=torch.tensor([[[2.0, 3.0], [5.0, 6.0]]]),
        dim=1,
    )

    # Should reduce from (1, 2, 2) to (1, 2)
    assert out.lower.shape == (1, 2)
    assert out.upper.shape == (1, 2)


def test_mean_relation_to_sum() -> None:
    """Test that mean([a, b]) * n = sum([a, b]) where n is the size of the dimension."""
    lower = torch.tensor([[1.0, 2.0, 3.0]])
    upper = torch.tensor([[2.0, 3.0, 4.0]])

    # Compute mean
    mean_out = _propagate(lower, upper, dim=1)

    # Compute sum
    sum_strategy = IBPSum()
    bounds = IntervalBounds(lower, upper)
    sum_out = propagate(sum_strategy, bounds, dim=1, keepdim=False)

    # mean * n should equal sum
    n = lower.shape[1]  # 3
    assert torch.allclose(mean_out.lower * n, sum_out.lower)
    assert torch.allclose(mean_out.upper * n, sum_out.upper)
