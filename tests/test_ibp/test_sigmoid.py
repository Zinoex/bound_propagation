from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.elementwise import IBPSigmoid
from bound_propagation.propagation.ibp.linear import IBPNeg
from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
    """Propagate bounds for sigmoid operation."""
    strategy = IBPSigmoid()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds)


def test_sigmoid_positive_interval() -> None:
    """Test sigmoid of positive interval."""
    # sigmoid([1, 2]) ≈ [0.731, 0.881]
    out = _propagate(
        lower=torch.tensor([1.0]),
        upper=torch.tensor([2.0]),
    )

    expected_lower = torch.sigmoid(torch.tensor([1.0]))
    expected_upper = torch.sigmoid(torch.tensor([2.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_sigmoid_negative_interval() -> None:
    """Test sigmoid of negative interval."""
    # sigmoid([-2, -1]) ≈ [0.119, 0.269]
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([-1.0]),
    )

    expected_lower = torch.sigmoid(torch.tensor([-2.0]))
    expected_upper = torch.sigmoid(torch.tensor([-1.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_sigmoid_mixed_sign_interval() -> None:
    """Test sigmoid of interval with mixed signs."""
    # sigmoid([-1, 1]) ≈ [0.269, 0.731]
    out = _propagate(
        lower=torch.tensor([-1.0]),
        upper=torch.tensor([1.0]),
    )

    expected_lower = torch.sigmoid(torch.tensor([-1.0]))
    expected_upper = torch.sigmoid(torch.tensor([1.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_sigmoid_zero() -> None:
    """Test sigmoid(0) = 0.5."""
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([0.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.5]))
    assert torch.allclose(out.upper, torch.tensor([0.5]))


def test_sigmoid_interval_containing_zero() -> None:
    """Test sigmoid of interval containing zero."""
    # sigmoid([-2, 2]) should contain 0.5
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([2.0]),
    )

    # Should contain 0.5
    assert out.lower < 0.5
    assert out.upper > 0.5


def test_sigmoid_point_interval() -> None:
    """Test sigmoid of point interval (lower = upper)."""
    out = _propagate(
        lower=torch.tensor([1.5]),
        upper=torch.tensor([1.5]),
    )

    expected = torch.sigmoid(torch.tensor([1.5]))
    assert torch.allclose(out.lower, expected)
    assert torch.allclose(out.upper, expected)


def test_sigmoid_batched_intervals() -> None:
    """Test sigmoid with batched intervals."""
    lower = torch.tensor([0.0, -1.0, 1.0, -2.0])
    upper = torch.tensor([1.0, 0.0, 2.0, -1.0])

    out = _propagate(lower, upper)

    expected_lower = torch.sigmoid(lower)
    expected_upper = torch.sigmoid(upper)

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_sigmoid_multidimensional() -> None:
    """Test sigmoid with multi-dimensional intervals."""
    lower = torch.tensor([[-1.0, 0.0], [1.0, -2.0]])
    upper = torch.tensor([[1.0, 2.0], [3.0, 0.0]])

    out = _propagate(lower, upper)

    expected_lower = torch.sigmoid(lower)
    expected_upper = torch.sigmoid(upper)

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_sigmoid_large_positive_values() -> None:
    """Test sigmoid with large positive values (approaches 1)."""
    # sigmoid([5, 10]) should be very close to [1, 1]
    out = _propagate(
        lower=torch.tensor([5.0]),
        upper=torch.tensor([10.0]),
    )

    expected_lower = torch.sigmoid(torch.tensor([5.0]))
    expected_upper = torch.sigmoid(torch.tensor([10.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)
    # Should be close to 1
    assert out.lower > 0.99
    assert out.upper > 0.9999


def test_sigmoid_large_negative_values() -> None:
    """Test sigmoid with large negative values (approaches 0)."""
    # sigmoid([-10, -5]) should be very close to [0, 0]
    out = _propagate(
        lower=torch.tensor([-10.0]),
        upper=torch.tensor([-5.0]),
    )

    expected_lower = torch.sigmoid(torch.tensor([-10.0]))
    expected_upper = torch.sigmoid(torch.tensor([-5.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)
    # Should be close to 0
    assert out.lower < 0.0001
    assert out.upper < 0.01


def test_sigmoid_bounded_by_unit_interval() -> None:
    """Test that sigmoid always returns bounds in [0, 1]."""
    # Test various intervals
    test_cases = [
        (torch.tensor([-100.0]), torch.tensor([100.0])),
        (torch.tensor([-10.0]), torch.tensor([10.0])),
        (torch.tensor([0.0]), torch.tensor([5.0])),
    ]

    for lower, upper in test_cases:
        out = _propagate(lower, upper)
        # Sigmoid approaches but can equal 0 and 1 in the limits
        assert torch.all(out.lower >= 0.0)
        assert torch.all(out.upper <= 1.0)


def test_sigmoid_monotonicity() -> None:
    """Test that sigmoid is monotonically increasing."""
    # If [a, b] ⊆ [c, d], then sigmoid([a, b]) ⊆ sigmoid([c, d])
    inner = IntervalBounds(torch.tensor([0.5]), torch.tensor([1.5]))
    outer = IntervalBounds(torch.tensor([0.0]), torch.tensor([2.0]))

    strategy = IBPSigmoid()

    out_inner = propagate(strategy, inner)
    out_outer = propagate(strategy, outer)

    assert out_outer.lower <= out_inner.lower
    assert out_outer.upper >= out_inner.upper


def test_sigmoid_symmetry_property() -> None:
    """Test that sigmoid(-x) = 1 - sigmoid(x)."""
    a = IntervalBounds(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 3.0]))

    strategy = IBPSigmoid()

    neg_strategy = IBPNeg()

    # sigmoid(a)
    sig_a = propagate(strategy, a)

    # -a
    neg_a_val = propagate(neg_strategy, a)

    # sigmoid(-a)
    sig_neg_a = propagate(strategy, neg_a_val)

    # sigmoid(-x) + sigmoid(x) = 1
    sum_lower = sig_a.lower + sig_neg_a.upper  # Note: reversed for interval arithmetic
    sum_upper = sig_a.upper + sig_neg_a.lower

    # Should be close to 1
    assert torch.allclose(sum_lower, torch.tensor([1.0]), atol=1e-5)
    assert torch.allclose(sum_upper, torch.tensor([1.0]), atol=1e-5)
