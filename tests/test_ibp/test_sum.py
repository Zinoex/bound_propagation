from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.reduction import IBPSum
from tests.helpers import propagate


def _propagate(
    lower: torch.Tensor, upper: torch.Tensor, dim: int | None = None, keepdim: bool = False
) -> IntervalBounds:
    """Propagate bounds for sum operation."""
    strategy = IBPSum()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds, dim=dim, keepdim=keepdim)


def test_sum_positive_interval() -> None:
    """Test sum of positive intervals."""
    # sum([1, 3], [2, 4], [3, 5]) = [6, 12]
    out = _propagate(
        lower=torch.tensor([[1.0, 2.0, 3.0]]),
        upper=torch.tensor([[3.0, 4.0, 5.0]]),
        dim=1,
    )

    expected_lower = 1.0 + 2.0 + 3.0  # 6.0
    expected_upper = 3.0 + 4.0 + 5.0  # 12.0

    assert torch.allclose(out.lower, torch.tensor([expected_lower]))
    assert torch.allclose(out.upper, torch.tensor([expected_upper]))


def test_sum_negative_interval() -> None:
    """Test sum of negative intervals."""
    # sum([-3, -1], [-4, -2], [-5, -3]) = [-12, -6]
    out = _propagate(
        lower=torch.tensor([[-3.0, -4.0, -5.0]]),
        upper=torch.tensor([[-1.0, -2.0, -3.0]]),
        dim=1,
    )

    expected_lower = -3.0 + -4.0 + -5.0  # -12.0
    expected_upper = -1.0 + -2.0 + -3.0  # -6.0

    assert torch.allclose(out.lower, torch.tensor([expected_lower]))
    assert torch.allclose(out.upper, torch.tensor([expected_upper]))


def test_sum_mixed_intervals() -> None:
    """Test sum of mixed sign intervals."""
    # sum([-2, 1], [0, 3], [-1, 2]) = [-3, 6]
    out = _propagate(
        lower=torch.tensor([[-2.0, 0.0, -1.0]]),
        upper=torch.tensor([[1.0, 3.0, 2.0]]),
        dim=1,
    )

    expected_lower = -2.0 + 0.0 + -1.0  # -3.0
    expected_upper = 1.0 + 3.0 + 2.0  # 6.0

    assert torch.allclose(out.lower, torch.tensor([expected_lower]))
    assert torch.allclose(out.upper, torch.tensor([expected_upper]))


def test_sum_point_intervals() -> None:
    """Test sum of point intervals (lower = upper)."""
    # sum([2, 2], [3, 3], [4, 4]) = [9, 9]
    out = _propagate(
        lower=torch.tensor([[2.0, 3.0, 4.0]]),
        upper=torch.tensor([[2.0, 3.0, 4.0]]),
        dim=1,
    )

    expected_sum = 2.0 + 3.0 + 4.0  # 9.0

    assert torch.allclose(out.lower, torch.tensor([expected_sum]))
    assert torch.allclose(out.upper, torch.tensor([expected_sum]))


def test_sum_multi_batch() -> None:
    """Test sum with multiple batches."""
    out = _propagate(
        lower=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        upper=torch.tensor([[2.0, 3.0], [5.0, 6.0]]),
        dim=1,
    )

    # Batch 0: [1+2, 2+3] = [3, 5]
    # Batch 1: [3+4, 5+6] = [7, 11]
    expected_lower = torch.tensor([3.0, 7.0])
    expected_upper = torch.tensor([5.0, 11.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_sum_dim_0() -> None:
    """Test sum across dimension 0."""
    out = _propagate(
        lower=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        upper=torch.tensor([[2.0, 3.0], [5.0, 6.0]]),
        dim=0,
    )

    # Sum along dim 0: [[1+3, 2+4], [2+5, 3+6]] = [[4, 6], [7, 9]]
    expected_lower = torch.tensor([4.0, 6.0])
    expected_upper = torch.tensor([7.0, 9.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_sum_keepdim() -> None:
    """Test sum with keepdim=True."""
    out = _propagate(
        lower=torch.tensor([[1.0, 2.0, 3.0]]),
        upper=torch.tensor([[2.0, 3.0, 4.0]]),
        dim=1,
        keepdim=True,
    )

    # Sum along dim 1 with keepdim: shape becomes (1, 1)
    expected_lower = torch.tensor([[6.0]])
    expected_upper = torch.tensor([[9.0]])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)
    assert out.lower.shape == (1, 1)


def test_sum_entire_tensor() -> None:
    """Test sum without specifying dim (sums entire tensor)."""
    out = _propagate(
        lower=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        upper=torch.tensor([[2.0, 3.0], [5.0, 6.0]]),
        dim=None,
    )

    # Sum all elements: [1+2+3+4, 2+3+5+6] = [10, 16]
    expected_lower = torch.tensor(10.0)
    expected_upper = torch.tensor(16.0)

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_sum_monotonicity() -> None:
    """Test that sum preserves monotonicity: if [a, b] ⊆ [c, d], then sum([a, b]) ⊆ sum([c, d])."""
    inner = IntervalBounds(torch.tensor([[2.0, 3.0]]), torch.tensor([[4.0, 5.0]]))
    outer = IntervalBounds(torch.tensor([[1.0, 2.0]]), torch.tensor([[5.0, 6.0]]))

    strategy = IBPSum()

    out_inner = propagate(strategy, inner, dim=1, keepdim=False)
    out_outer = propagate(strategy, outer, dim=1, keepdim=False)

    assert torch.all(out_outer.lower <= out_inner.lower)
    assert torch.all(out_outer.upper >= out_inner.upper)


def test_sum_zero_widening() -> None:
    """Test that sum may widen intervals."""
    # Individual intervals have width 1, but sum has larger width
    lower = torch.tensor([[1.0, 2.0, 3.0]])
    upper = torch.tensor([[2.0, 3.0, 4.0]])

    out = _propagate(lower, upper, dim=1)

    input_widths = upper - lower  # [1, 1, 1]
    input_total_width = input_widths.sum()  # 3

    output_width = out.upper - out.lower  # Should also be 3

    assert torch.allclose(output_width, input_total_width)


def test_sum_dimension_reduction() -> None:
    """Test that sum reduces the dimensionality correctly."""
    out = _propagate(
        lower=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        upper=torch.tensor([[[2.0, 3.0], [5.0, 6.0]]]),
        dim=1,
        keepdim=False,
    )

    # Should reduce from (1, 2, 2) to (1, 2)
    assert out.lower.shape == (1, 2)
    assert out.upper.shape == (1, 2)
