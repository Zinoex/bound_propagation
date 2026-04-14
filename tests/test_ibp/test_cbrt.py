from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.cbrt import IBPCbrt
from bound_propagation.propagation.ibp.neg import IBPNeg

from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
    """Propagate bounds for cbrt (cube root) operation."""
    strategy = IBPCbrt()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds)


def test_cbrt_positive_interval() -> None:
    """Test cbrt of positive interval."""
    # cbrt([8, 27]) = [2, 3]
    out = _propagate(
        lower=torch.tensor([8.0]),
        upper=torch.tensor([27.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([3.0]))


def test_cbrt_negative_interval() -> None:
    """Test cbrt of negative interval."""
    # cbrt([-27, -8]) = [-3, -2]
    out = _propagate(
        lower=torch.tensor([-27.0]),
        upper=torch.tensor([-8.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-3.0]))
    assert torch.allclose(out.upper, torch.tensor([-2.0]))


def test_cbrt_mixed_sign_interval() -> None:
    """Test cbrt of interval with mixed signs."""
    # cbrt([-8, 27]) = [-2, 3]
    out = _propagate(
        lower=torch.tensor([-8.0]),
        upper=torch.tensor([27.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-2.0]))
    assert torch.allclose(out.upper, torch.tensor([3.0]))


def test_cbrt_zero_interval() -> None:
    """Test cbrt of zero interval."""
    # cbrt([0, 0]) = [0, 0]
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([0.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_cbrt_zero_lower_bound() -> None:
    """Test cbrt with zero as lower bound."""
    # cbrt([0, 64]) = [0, 4]
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([64.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([4.0]))


def test_cbrt_zero_upper_bound() -> None:
    """Test cbrt with zero as upper bound."""
    # cbrt([-64, 0]) = [-4, 0]
    out = _propagate(
        lower=torch.tensor([-64.0]),
        upper=torch.tensor([0.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-4.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_cbrt_point_interval() -> None:
    """Test cbrt of point interval (lower = upper)."""
    # cbrt([64, 64]) = [4, 4]
    out = _propagate(
        lower=torch.tensor([64.0]),
        upper=torch.tensor([64.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([4.0]))
    assert torch.allclose(out.upper, torch.tensor([4.0]))


def test_cbrt_small_positive_interval() -> None:
    """Test cbrt of small positive interval."""
    # cbrt([0.125, 0.512]) ≈ [0.5, 0.8]
    out = _propagate(
        lower=torch.tensor([0.125]),
        upper=torch.tensor([0.512]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.5]), atol=1e-6)
    assert torch.allclose(out.upper, torch.tensor([0.8]), atol=1e-6)


def test_cbrt_small_negative_interval() -> None:
    """Test cbrt of small negative interval."""
    # cbrt([-0.512, -0.125]) ≈ [-0.8, -0.5]
    out = _propagate(
        lower=torch.tensor([-0.512]),
        upper=torch.tensor([-0.125]),
    )

    assert torch.allclose(out.lower, torch.tensor([-0.8]), atol=1e-6)
    assert torch.allclose(out.upper, torch.tensor([-0.5]), atol=1e-6)


def test_cbrt_batched_intervals() -> None:
    """Test cbrt with batched intervals."""
    out = _propagate(
        lower=torch.tensor([0.0, 8.0, -27.0, -8.0, 1.0]),
        upper=torch.tensor([8.0, 64.0, -1.0, 27.0, 8.0]),
    )

    expected_lower = torch.tensor([0.0, 2.0, -3.0, -2.0, 1.0])
    expected_upper = torch.tensor([2.0, 4.0, -1.0, 3.0, 2.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_cbrt_multidimensional() -> None:
    """Test cbrt with multi-dimensional intervals."""
    out = _propagate(
        lower=torch.tensor([[0.0, -8.0], [1.0, -27.0]]),
        upper=torch.tensor([[27.0, 8.0], [64.0, -1.0]]),
    )

    expected_lower = torch.tensor([[0.0, -2.0], [1.0, -3.0]])
    expected_upper = torch.tensor([[3.0, 2.0], [4.0, -1.0]])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_cbrt_monotonicity() -> None:
    """Test that cbrt is monotonically increasing."""
    # If [a, b] ⊆ [c, d], then cbrt([a, b]) ⊆ cbrt([c, d])
    inner = IntervalBounds(torch.tensor([8.0]), torch.tensor([27.0]))
    outer = IntervalBounds(torch.tensor([1.0]), torch.tensor([64.0]))

    strategy = IBPCbrt()

    out_inner = propagate(strategy, inner)
    out_outer = propagate(strategy, outer)

    # cbrt([8, 27]) = [2, 3] should be contained in cbrt([1, 64]) = [1, 4]
    assert out_outer.lower <= out_inner.lower
    assert out_outer.upper >= out_inner.upper


def test_cbrt_odd_function_property() -> None:
    """Test that cbrt is an odd function: cbrt(-x) = -cbrt(x)."""
    a = IntervalBounds(torch.tensor([8.0, 1.0]), torch.tensor([27.0, 64.0]))

    strategy = IBPCbrt()

    neg_strategy = IBPNeg()

    # cbrt(a)
    cbrt_a = propagate(strategy, a)

    # -a
    neg_a = propagate(neg_strategy, a)

    # cbrt(-a)
    cbrt_neg_a = propagate(strategy, neg_a)

    # -cbrt(a)
    neg_cbrt_a = propagate(neg_strategy, cbrt_a)

    # cbrt(-a) should equal -cbrt(a)
    assert torch.allclose(cbrt_neg_a.lower, neg_cbrt_a.lower)
    assert torch.allclose(cbrt_neg_a.upper, neg_cbrt_a.upper)


def test_cbrt_composition_with_cube() -> None:
    """Test that cbrt(x^3) = x."""
    # Test with both positive and negative values
    a_lower = torch.tensor([2.0, -3.0])
    a_upper = torch.tensor([4.0, -1.0])

    # x^3 (manually computed)
    a_cubed_lower = a_lower**3
    a_cubed_upper = a_upper**3

    # cbrt(x^3)
    cbrt_strategy = IBPCbrt()
    a_cubed = IntervalBounds(a_cubed_lower, a_cubed_upper)
    result = propagate(cbrt_strategy, a_cubed)

    # Should recover the original interval (or be very close due to numerical precision)
    assert torch.allclose(result.lower, a_lower, rtol=1e-5)
    assert torch.allclose(result.upper, a_upper, rtol=1e-5)


def test_cbrt_symmetric_interval() -> None:
    """Test cbrt of symmetric interval around zero."""
    # cbrt([-27, 27]) = [-3, 3]
    out = _propagate(
        lower=torch.tensor([-27.0]),
        upper=torch.tensor([27.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-3.0]))
    assert torch.allclose(out.upper, torch.tensor([3.0]))


def test_cbrt_narrowing_widening_property() -> None:
    """Test that cbrt narrows intervals for |x| > 1 and widens for |x| < 1."""
    # For [a, b] where |a|, |b| > 1, cbrt([a, b]) has smaller width
    large = _propagate(torch.tensor([8.0]), torch.tensor([64.0]))
    large_width_in = 64.0 - 8.0
    large_width_out = large.upper - large.lower
    assert large_width_out < large_width_in  # 4 - 2 = 2 < 56

    # For [a, b] where 0 < |a|, |b| < 1, cbrt([a, b]) has larger width
    small = _propagate(torch.tensor([0.008]), torch.tensor([0.125]))
    small_width_in = 0.125 - 0.008
    small_width_out = small.upper - small.lower
    assert small_width_out > small_width_in  # 0.5 - 0.2 = 0.3 > 0.117
