"""Systematic tests for reduction backward LBP strategies.

Sum and mean are linear: bounds must be exact.
Amax and amin use IntervalLeafRelaxation (chain-breaking): soundness only.
"""

from __future__ import annotations

import torch

from .conftest import assert_exact, assert_sound, propagate_bound, region

# ---------------------------------------------------------------------------
# Sum
# ---------------------------------------------------------------------------


class TestBackwardLBPSum:
    def test_full_sum_exact(self):
        """sum(x) over all dims: single scalar output."""
        r = region([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert_exact(
            lambda x: torch.sum(x),
            r,
            torch.tensor(6.0),  # 1 + 2 + 3
            torch.tensor(15.0),  # 4 + 5 + 6
        )

    def test_partial_sum_exact(self):
        """sum along one dimension, no keepdim."""
        r = region([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

        def sum_fn(x):
            return x.reshape(2, 3).sum(dim=0)

        bounds = propagate_bound(sum_fn, r)
        lower, upper = bounds.concretize()
        assert lower.shape == (3,)
        expected_lower = r.lower.reshape(2, 3).sum(dim=0)
        expected_upper = r.upper.reshape(2, 3).sum(dim=0)
        assert torch.allclose(lower, expected_lower)
        assert torch.allclose(upper, expected_upper)

    def test_sum_keepdim_exact(self):
        """sum with keepdim=True."""
        r = region([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

        def sum_fn(x):
            return x.reshape(2, 3).sum(dim=1, keepdim=True)

        bounds = propagate_bound(sum_fn, r)
        lower, upper = bounds.concretize()
        assert lower.shape == (2, 1)
        expected_lower = r.lower.reshape(2, 3).sum(dim=1, keepdim=True)
        expected_upper = r.upper.reshape(2, 3).sum(dim=1, keepdim=True)
        assert torch.allclose(lower, expected_lower)
        assert torch.allclose(upper, expected_upper)

    def test_sum_negative_dim(self):
        """sum with negative dimension."""
        r = region([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])

        def sum_fn(x):
            return x.reshape(2, 2).sum(dim=-1)

        bounds = propagate_bound(sum_fn, r)
        lower, upper = bounds.concretize()
        assert lower.shape == (2,)
        expected_lower = r.lower.reshape(2, 2).sum(dim=-1)
        assert torch.allclose(lower, expected_lower)


# ---------------------------------------------------------------------------
# Mean
# ---------------------------------------------------------------------------


class TestBackwardLBPMean:
    def test_full_mean_exact(self):
        """mean(x) over all dims."""
        r = region([2.0, 4.0, 6.0], [4.0, 6.0, 8.0])
        assert_exact(
            lambda x: torch.mean(x),
            r,
            torch.tensor(4.0),  # (2 + 4 + 6) / 3
            torch.tensor(6.0),  # (4 + 6 + 8) / 3
        )

    def test_partial_mean_exact(self):
        """mean along one dimension."""
        r = region([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])

        def mean_fn(x):
            return x.reshape(2, 2).mean(dim=0)

        bounds = propagate_bound(mean_fn, r)
        lower, upper = bounds.concretize()
        assert lower.shape == (2,)
        expected_lower = r.lower.reshape(2, 2).mean(dim=0)
        expected_upper = r.upper.reshape(2, 2).mean(dim=0)
        assert torch.allclose(lower, expected_lower, atol=1e-5)
        assert torch.allclose(upper, expected_upper, atol=1e-5)


# ---------------------------------------------------------------------------
# Amax (chain-breaking)
# ---------------------------------------------------------------------------


class TestBackwardLBPAmax:
    def test_full_amax_sound(self):
        """amax over all dims: scalar output."""
        assert_sound(
            lambda x: torch.amax(x),
            region([-1.0, 0.5, 2.0], [1.0, 2.0, 3.0]),
        )

    def test_partial_amax_sound(self):
        """amax along one dimension."""

        def amax_fn(x):
            return x.reshape(2, 2).amax(dim=0)

        assert_sound(amax_fn, region([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]))

    def test_amax_after_relu_sound(self):
        """amax(relu(x)): chain-breaking after nonlinear."""

        def fn(x):
            return torch.amax(torch.relu(x))

        assert_sound(fn, region([-2.0, 0.5], [1.0, 3.0]))

    def test_amax_single_element(self):
        """amax of 1-element tensor: should be identity-like."""
        lower, upper = assert_sound(lambda x: torch.amax(x), region([2.0], [5.0]))
        assert torch.allclose(lower, torch.tensor(2.0), atol=1e-5)
        assert torch.allclose(upper, torch.tensor(5.0), atol=1e-5)


# ---------------------------------------------------------------------------
# Amin (chain-breaking)
# ---------------------------------------------------------------------------


class TestBackwardLBPAmin:
    def test_full_amin_sound(self):
        assert_sound(
            lambda x: torch.amin(x),
            region([-1.0, 0.5, 2.0], [1.0, 2.0, 3.0]),
        )

    def test_partial_amin_sound(self):
        def amin_fn(x):
            return x.reshape(2, 2).amin(dim=1)

        assert_sound(amin_fn, region([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]))

    def test_amin_single_element(self):
        lower, upper = assert_sound(lambda x: torch.amin(x), region([2.0], [5.0]))
        assert torch.allclose(lower, torch.tensor(2.0), atol=1e-5)
        assert torch.allclose(upper, torch.tensor(5.0), atol=1e-5)


# ---------------------------------------------------------------------------
# Reduction + arithmetic compositions
# ---------------------------------------------------------------------------


class TestReductionComposition:
    def test_sum_then_scale_exact(self):
        """sum(x) * 2: linear, must be exact."""
        r = region([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert_exact(
            lambda x: torch.sum(x) * 2.0,
            r,
            torch.tensor(12.0),
            torch.tensor(30.0),
        )

    def test_mean_then_relu_sound(self):
        """mean(x, dim=0) then relu: nonlinear chain."""

        def fn(x):
            return torch.relu(x.reshape(2, 2).mean(dim=0))

        assert_sound(fn, region([-2.0, -1.0, 1.0, 2.0], [0.0, 1.0, 3.0, 4.0]))
