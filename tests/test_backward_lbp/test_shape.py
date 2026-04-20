"""Systematic tests for shape-manipulation backward LBP strategies.

All shape operations are linear (pure dimension rearrangement),
so bounds must be exact when composed with identity-like inputs.
"""

from __future__ import annotations

import torch

from .conftest import assert_exact, assert_sound, propagate_bound, region

# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------


class TestBackwardLBPReshape:
    def test_flatten_exact(self):
        """reshape([2,3] -> [6]) preserves bounds exactly."""
        r = region([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        def reshape_fn(x):
            return x.reshape(2, 3)

        bounds = propagate_bound(reshape_fn, r)
        lower, upper = bounds.concretize()
        assert lower.shape == (2, 3)
        assert torch.allclose(lower.flatten(), r.lower)
        assert torch.allclose(upper.flatten(), r.upper)

    def test_reshape_roundtrip_exact(self):
        """reshape -> reshape back should be exact identity."""
        r = region([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])

        def roundtrip_fn(x):
            return x.reshape(2, 2).reshape(4)

        assert_exact(roundtrip_fn, r, r.lower, r.upper)


# ---------------------------------------------------------------------------
# Unsqueeze
# ---------------------------------------------------------------------------


class TestBackwardLBPUnsqueeze:
    def test_unsqueeze_0_exact(self):
        r = region([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

        def unsqueeze_fn(x):
            return x.unsqueeze(0)

        bounds = propagate_bound(unsqueeze_fn, r)
        lower, upper = bounds.concretize()
        assert lower.shape == (1, 3)
        assert torch.allclose(lower.squeeze(0), r.lower)
        assert torch.allclose(upper.squeeze(0), r.upper)

    def test_unsqueeze_last_exact(self):
        r = region([1.0, 2.0], [3.0, 4.0])

        def unsqueeze_fn(x):
            return x.unsqueeze(-1)

        bounds = propagate_bound(unsqueeze_fn, r)
        lower, upper = bounds.concretize()
        assert lower.shape == (2, 1)
        assert torch.allclose(lower.squeeze(-1), r.lower)


# ---------------------------------------------------------------------------
# Squeeze
# ---------------------------------------------------------------------------


class TestBackwardLBPSqueeze:
    def test_squeeze_exact(self):
        """squeeze undoes unsqueeze."""
        r = region([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

        def squeeze_fn(x):
            return x.unsqueeze(0).squeeze(0)

        assert_exact(squeeze_fn, r, r.lower, r.upper)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


class TestBackwardLBPTranspose:
    def test_transpose_exact(self):
        r = region([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

        def transpose_fn(x):
            return x.reshape(2, 3).transpose(0, 1)

        bounds = propagate_bound(transpose_fn, r)
        lower, upper = bounds.concretize()
        assert lower.shape == (3, 2)

        expected_lower = r.lower.reshape(2, 3).transpose(0, 1)
        assert torch.allclose(lower, expected_lower)

    def test_double_transpose_identity(self):
        """transpose(0,1) twice is identity."""
        r = region([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])

        def double_transpose_fn(x):
            t = x.reshape(2, 2)
            return t.transpose(0, 1).transpose(0, 1).reshape(4)

        assert_exact(double_transpose_fn, r, r.lower, r.upper)


# ---------------------------------------------------------------------------
# Permute
# ---------------------------------------------------------------------------


class TestBackwardLBPPermute:
    def test_permute_exact(self):
        r = region(
            [float(i) for i in range(24)],
            [float(i + 1) for i in range(24)],
        )

        def permute_fn(x):
            return x.reshape(2, 3, 4).permute(2, 0, 1)

        bounds = propagate_bound(permute_fn, r)
        lower, upper = bounds.concretize()
        assert lower.shape == (4, 2, 3)

        expected_lower = r.lower.reshape(2, 3, 4).permute(2, 0, 1)
        assert torch.allclose(lower, expected_lower)

    def test_inverse_permute_identity(self):
        """Applying permute then inverse permute is identity."""
        r = region([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

        def inv_permute_fn(x):
            t = x.reshape(2, 3)
            return t.permute(1, 0).permute(1, 0).reshape(6)

        assert_exact(inv_permute_fn, r, r.lower, r.upper)


# ---------------------------------------------------------------------------
# Select
# ---------------------------------------------------------------------------


class TestBackwardLBPSelect:
    def test_select_first(self):
        r = region([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

        def select_fn(x):
            return x.reshape(3, 1).select(0, 0)

        bounds = propagate_bound(select_fn, r)
        lower, upper = bounds.concretize()
        assert lower.shape == (1,)
        assert torch.allclose(lower, torch.tensor([1.0]))
        assert torch.allclose(upper, torch.tensor([4.0]))

    def test_select_last(self):
        r = region([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

        def select_fn(x):
            return x.reshape(3, 1).select(0, 2)

        bounds = propagate_bound(select_fn, r)
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, torch.tensor([3.0]))
        assert torch.allclose(upper, torch.tensor([6.0]))


# ---------------------------------------------------------------------------
# GetItem
# ---------------------------------------------------------------------------


class TestBackwardLBPGetItem:
    def test_getitem_slice(self):
        r = region([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])

        def getitem_fn(x):
            return x[1:3]

        bounds = propagate_bound(getitem_fn, r)
        lower, upper = bounds.concretize()
        assert lower.shape == (2,)
        assert torch.allclose(lower, torch.tensor([2.0, 3.0]))
        assert torch.allclose(upper, torch.tensor([6.0, 7.0]))

    def test_getitem_single(self):
        r = region([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

        def getitem_fn(x):
            return x[0:1]

        bounds = propagate_bound(getitem_fn, r)
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, torch.tensor([1.0]))
        assert torch.allclose(upper, torch.tensor([4.0]))


# ---------------------------------------------------------------------------
# Cat
# ---------------------------------------------------------------------------


class TestBackwardLBPCat:
    def test_cat_exact(self):
        """cat([x[:2], x[2:]]) should reconstruct x exactly."""
        r = region([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])

        def cat_fn(x):
            return torch.cat([x[:2], x[2:]], dim=0)

        assert_exact(cat_fn, r, r.lower, r.upper)

    def test_cat_reverse(self):
        """cat with reversed order."""
        r = region([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])

        def cat_rev_fn(x):
            return torch.cat([x[2:], x[:2]], dim=0)

        bounds = propagate_bound(cat_rev_fn, r)
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, torch.tensor([3.0, 4.0, 1.0, 2.0]))
        assert torch.allclose(upper, torch.tensor([7.0, 8.0, 5.0, 6.0]))

    def test_cat_duplicate_input(self):
        """cat([x, x]) should double the bounds."""
        r = region([1.0, 2.0], [3.0, 4.0])

        def cat_dup_fn(x):
            return torch.cat([x, x], dim=0)

        bounds = propagate_bound(cat_dup_fn, r)
        lower, upper = bounds.concretize()
        assert lower.shape == (4,)
        assert torch.allclose(lower, torch.tensor([1.0, 2.0, 1.0, 2.0]))
        assert torch.allclose(upper, torch.tensor([3.0, 4.0, 3.0, 4.0]))


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------


class TestBackwardLBPStack:
    def test_stack_exact(self):
        r = region([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])

        def stack_fn(x):
            return torch.stack([x[:2], x[2:]], dim=0)

        bounds = propagate_bound(stack_fn, r)
        lower, upper = bounds.concretize()
        assert lower.shape == (2, 2)
        assert torch.allclose(lower, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert torch.allclose(upper, torch.tensor([[5.0, 6.0], [7.0, 8.0]]))

    def test_stack_duplicate_input(self):
        """stack([x, x]) repeats bounds along new dim."""
        r = region([1.0, 2.0], [3.0, 4.0])

        def stack_dup_fn(x):
            return torch.stack([x, x], dim=0)

        bounds = propagate_bound(stack_dup_fn, r)
        lower, upper = bounds.concretize()
        assert lower.shape == (2, 2)
        assert torch.allclose(lower, torch.tensor([[1.0, 2.0], [1.0, 2.0]]))


# ---------------------------------------------------------------------------
# Shape + arithmetic compositions
# ---------------------------------------------------------------------------


class TestShapeArithmeticComposition:
    def test_reshape_add_exact(self):
        """Reshape then add constant: still exact."""
        r = region([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
        c = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

        def fn(x):
            return x.reshape(2, 2) + c

        bounds = propagate_bound(fn, r)
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, r.lower.reshape(2, 2) + c)
        assert torch.allclose(upper, r.upper.reshape(2, 2) + c)

    def test_getitem_relu_sound(self):
        """Slice then ReLU: soundness."""

        def fn(x):
            return torch.relu(x[1:3])

        assert_sound(fn, region([-2.0, -1.0, 0.5, 2.0], [1.0, 2.0, 3.0, 4.0]))
