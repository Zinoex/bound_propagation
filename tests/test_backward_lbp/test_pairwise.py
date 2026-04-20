"""Systematic tests for pairwise (binary) backward LBP strategies.

Covers mul, div, maximum, minimum with:
- Both operands abstract
- One operand constant (left or right)
- Same-node case (x * x)
- Soundness sampling
"""

from __future__ import annotations

import torch

from .conftest import assert_exact, assert_sound, region

# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------


class TestBackwardLBPMul:
    def test_constant_right_exact(self):
        """x * 2 is linear: bounds must be exact."""
        r = region([1.0, -1.0], [3.0, 1.0])
        assert_exact(
            lambda x: x * 2.0,
            r,
            torch.tensor([2.0, -2.0]),
            torch.tensor([6.0, 2.0]),
        )

    def test_constant_left_exact(self):
        """3 * x is linear: bounds must be exact."""
        r = region([1.0, -1.0], [3.0, 1.0])
        assert_exact(
            lambda x: 3.0 * x,
            r,
            torch.tensor([3.0, -3.0]),
            torch.tensor([9.0, 3.0]),
        )

    def test_negative_constant_exact(self):
        """x * (-2): flips bounds."""
        r = region([1.0, 2.0], [3.0, 4.0])
        assert_exact(
            lambda x: x * (-2.0),
            r,
            torch.tensor([-6.0, -8.0]),
            torch.tensor([-2.0, -4.0]),
        )

    def test_abstract_times_abstract_sound(self):
        """x0 * x1 with both abstract: McCormick relaxation."""

        def mul_fn(x):
            return x[0:1] * x[1:2]

        assert_sound(mul_fn, region([1.0, 2.0], [3.0, 4.0]))

    def test_abstract_times_abstract_crossing_sound(self):
        """x0 * x1 with crossing zero."""

        def mul_fn(x):
            return x[0:1] * x[1:2]

        assert_sound(mul_fn, region([-1.0, -2.0], [3.0, 4.0]))

    def test_x_squared_sound(self):
        """x * x (same node): accumulate_a_terms must handle correctly."""

        def sq_fn(x):
            return x * x

        assert_sound(sq_fn, region([1.0, -1.0], [3.0, 2.0]))

    def test_x_squared_crossing_zero(self):
        def sq_fn(x):
            return x * x

        assert_sound(sq_fn, region([-2.0], [2.0]))


# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------


class TestBackwardLBPDiv:
    def test_constant_divisor_exact(self):
        """x / 2 is linear: bounds must be exact."""
        r = region([2.0, -4.0], [6.0, 4.0])
        assert_exact(
            lambda x: x / 2.0,
            r,
            torch.tensor([1.0, -2.0]),
            torch.tensor([3.0, 2.0]),
        )

    def test_constant_dividend_sound(self):
        """1.0 / x with x in [1, 3]: nonlinear, check soundness."""

        def div_fn(x):
            return 1.0 / x

        assert_sound(div_fn, region([1.0, 2.0], [3.0, 5.0]))

    def test_abstract_div_abstract_sound(self):
        """x0 / x1 with both abstract."""

        def div_fn(x):
            return x[0:1] / x[1:2]

        assert_sound(div_fn, region([1.0, 1.0], [3.0, 4.0]))

    def test_negative_divisor_sound(self):
        """x / (-2): flips bounds."""
        r = region([2.0, 4.0], [6.0, 8.0])
        assert_exact(
            lambda x: x / (-2.0),
            r,
            torch.tensor([-3.0, -4.0]),
            torch.tensor([-1.0, -2.0]),
        )


# ---------------------------------------------------------------------------
# Maximum
# ---------------------------------------------------------------------------


class TestBackwardLBPMaximum:
    def test_both_abstract_sound(self):
        def max_fn(x):
            return torch.maximum(x[0:1], x[1:2])

        assert_sound(max_fn, region([1.0, 2.0], [4.0, 5.0]))

    def test_both_abstract_disjoint_sound(self):
        """When ranges don't overlap: max is exact."""

        def max_fn(x):
            return torch.maximum(x[0:1], x[1:2])

        # x0 in [5,6], x1 in [1,2] -> max is always x0
        lower, upper = assert_sound(max_fn, region([5.0, 1.0], [6.0, 2.0]))
        assert torch.allclose(lower, torch.tensor([5.0]), atol=1e-5)
        assert torch.allclose(upper, torch.tensor([6.0]), atol=1e-5)

    def test_crossing_ranges_sound(self):
        def max_fn(x):
            return torch.maximum(x[0:1], x[1:2])

        assert_sound(max_fn, region([-1.0, -2.0], [2.0, 3.0]))

    def test_same_node_sound(self):
        """max(x, x) = x: relaxation may over-approximate since it doesn't
        know both operands are the same node."""

        def max_self_fn(x):
            return torch.maximum(x, x)

        r = region([1.0, -1.0], [3.0, 2.0])
        assert_sound(max_self_fn, r)

    def test_constant_right_sound(self):
        """max(x, 0) is like ReLU."""
        assert_sound(lambda x: torch.maximum(x, torch.tensor(0.0)), region([-2.0, -1.0], [3.0, 4.0]))


# ---------------------------------------------------------------------------
# Minimum
# ---------------------------------------------------------------------------


class TestBackwardLBPMinimum:
    def test_both_abstract_sound(self):
        def min_fn(x):
            return torch.minimum(x[0:1], x[1:2])

        assert_sound(min_fn, region([1.0, 2.0], [4.0, 5.0]))

    def test_disjoint_exact(self):
        """When ranges don't overlap: min is exact."""

        def min_fn(x):
            return torch.minimum(x[0:1], x[1:2])

        # x0 in [1,2], x1 in [5,6] -> min is always x0
        lower, upper = assert_sound(min_fn, region([1.0, 5.0], [2.0, 6.0]))
        assert torch.allclose(lower, torch.tensor([1.0]), atol=1e-5)
        assert torch.allclose(upper, torch.tensor([2.0]), atol=1e-5)

    def test_constant_right_sound(self):
        """min(x, 1) clamps from above."""
        assert_sound(lambda x: torch.minimum(x, torch.tensor(1.0)), region([-1.0, 0.0], [3.0, 2.0]))
