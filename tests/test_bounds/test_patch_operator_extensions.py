"""Tests for the three patch-mode extensions:

1. ``Conv2dPatchOperator.apply_transpose`` now fully structural (via ``F.fold``).
2. ``Conv2dPatchOperator`` accepts arbitrary stride / dilation at the operator
   level (operator-level operations work; the compose helpers still require
   ``stride=1, dilation=1``).
3. ``_compose_conv_with_patch`` composes a conv after a
   ``Conv2dPatchOperator`` into a larger ``Conv2dPatchOperator``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from bound_propagation.linear_operators import (
    Conv2dPatchOperator,
    ScaledConv2dOperator,
    _compose_conv_with_patch,
    _compose_conv_with_scaled,
)
from bound_propagation.regions import HyperRectangle


def _make_region(shape: tuple[int, ...], seed: int = 0) -> HyperRectangle:
    torch.manual_seed(seed)
    lower = torch.randn(*shape)
    upper = lower + torch.rand(*shape) + 0.1
    return HyperRectangle(lower=lower, upper=upper)


def _make_patch_via_compose(
    seed: int = 0, c_z: int = 4, padding1: tuple[int, int] = (1, 1), padding2: tuple[int, int] = (1, 1)
) -> Conv2dPatchOperator:
    torch.manual_seed(seed)
    W1 = torch.randn(3, 2, 3, 3)
    alpha = torch.randn(3, 4, 4)
    W2 = torch.randn(c_z, 3, 3, 3)
    scaled = ScaledConv2dOperator(
        weight=W1,
        stride=(1, 1),
        padding=padding1,
        dilation=(1, 1),
        groups=1,
        alpha=alpha,
        input_shape=(2, 4, 4),
        output_shape=(3, 4, 4),
    )
    return _compose_conv_with_scaled(
        scaled_op=scaled,
        weight2=W2,
        stride2=(1, 1),
        padding2=padding2,
        dilation2=(1, 1),
        groups2=1,
        output_shape=(c_z, 4, 4),
    )


# ----------------------------------------------------------------------
# 1. Structural apply_transpose
# ----------------------------------------------------------------------


class TestApplyTransposeStructural:
    def test_matches_dense_adjoint(self) -> None:
        op = _make_patch_via_compose(seed=0)
        y = torch.randn(*op.output_shape)
        via_struct = op.apply_transpose(y)
        via_dense = op.to_dense().apply_transpose(y)
        assert torch.allclose(via_struct, via_dense, atol=1e-5)

    def test_adjoint_inner_product_identity(self) -> None:
        """<y, op(x)> == <op^T(y), x> for random x, y."""
        op = _make_patch_via_compose(seed=1)
        for _ in range(3):
            x = torch.randn(*op.input_shape)
            y = torch.randn(*op.output_shape)
            lhs = (y * op.apply(x)).sum()
            rhs = (op.apply_transpose(y) * x).sum()
            assert torch.allclose(lhs, rhs, atol=1e-5)

    def test_apply_transpose_with_leading_batch(self) -> None:
        """apply_transpose with extra leading dims must reduce output axes and
        preserve leading dims."""
        op = _make_patch_via_compose(seed=2)
        y = torch.randn(3, *op.output_shape)
        dx = op.apply_transpose(y)
        expected = op.to_dense().apply_transpose(y)
        assert dx.shape == expected.shape
        assert torch.allclose(dx, expected, atol=1e-5)


# ----------------------------------------------------------------------
# 2. Operator-level arbitrary stride / dilation
# ----------------------------------------------------------------------


class TestOperatorLevelStrideDilation:
    def test_strided_operator_apply_matches_manual(self) -> None:
        """A manually-constructed strided patch op must agree with the
        equivalent explicit conv when the patch has identical kernel everywhere."""
        torch.manual_seed(10)
        c_in, c_out = 2, 3
        h_in = 7
        w_in = 7
        stride = (2, 2)
        padding = (1, 1)
        k_h = 3
        k_w = 3
        # H_out = floor((H_in + 2p - k) / s) + 1 = floor((7 + 2 - 3)/2) + 1 = 4
        h_out = (h_in + 2 * padding[0] - k_h) // stride[0] + 1
        w_out = (w_in + 2 * padding[1] - k_w) // stride[1] + 1

        conv_weight = torch.randn(c_out, c_in, k_h, k_w)
        # Build patches where every (h_out, w_out) has the same kernel = conv_weight.
        patches = conv_weight.reshape(1, 1, 1, c_out, c_in, k_h, k_w)
        patches = patches.expand(1, h_out, w_out, c_out, c_in, k_h, k_w).contiguous()
        # Expected shape: (*output_shape, C_in, k_h, k_w) with output_shape = (c_out, h_out, w_out).
        # Reshape: (c_out, h_out, w_out, c_in, k_h, k_w).
        patches = patches.squeeze(0).permute(2, 0, 1, 3, 4, 5).contiguous()

        op = Conv2dPatchOperator(
            patches=patches,
            stride=stride,
            padding=padding,
            dilation=(1, 1),
            groups=1,
            input_shape=(c_in, h_in, w_in),
            output_shape=(c_out, h_out, w_out),
        )

        x = torch.randn(c_in, h_in, w_in)
        got = op.apply(x)
        expected = F.conv2d(x.unsqueeze(0), conv_weight, stride=stride, padding=padding).squeeze(0)
        assert torch.allclose(got, expected, atol=1e-5)

    def test_strided_concretize_matches_dense(self) -> None:
        torch.manual_seed(11)
        c_in, c_out = 2, 3
        h_in, w_in = 7, 7
        stride = (2, 2)
        padding = (1, 1)
        h_out = (h_in + 2 * padding[0] - 3) // stride[0] + 1
        w_out = (w_in + 2 * padding[1] - 3) // stride[1] + 1
        patches = torch.randn(c_out, h_out, w_out, c_in, 3, 3)
        op = Conv2dPatchOperator(
            patches=patches,
            stride=stride,
            padding=padding,
            dilation=(1, 1),
            groups=1,
            input_shape=(c_in, h_in, w_in),
            output_shape=(c_out, h_out, w_out),
        )
        region = _make_region((c_in, h_in, w_in), seed=11)
        struct_min = op.concretize_min(region)
        dense_min = op.to_dense().concretize_min(region)
        assert torch.allclose(struct_min, dense_min, atol=1e-5)

    def test_dilated_operator_apply_matches_to_dense(self) -> None:
        """Dilated patch op: apply and to_dense must agree."""
        torch.manual_seed(12)
        c_in, c_out = 2, 3
        h_in, w_in = 7, 7
        dilation = (2, 2)
        padding = (2, 2)
        k_h, k_w = 3, 3
        # H_out = H_in + 2p - dilation*(k-1) = 7 + 4 - 4 = 7
        h_out, w_out = 7, 7
        patches = torch.randn(c_out, h_out, w_out, c_in, k_h, k_w)
        op = Conv2dPatchOperator(
            patches=patches,
            stride=(1, 1),
            padding=padding,
            dilation=dilation,
            groups=1,
            input_shape=(c_in, h_in, w_in),
            output_shape=(c_out, h_out, w_out),
        )
        x = torch.randn(c_in, h_in, w_in)
        got = op.apply(x)
        via_dense = op.to_dense().apply(x)
        assert torch.allclose(got, via_dense, atol=1e-5)


# ----------------------------------------------------------------------
# 3. conv ∘ Conv2dPatchOperator composition
# ----------------------------------------------------------------------


class TestComposeConvWithPatch:
    def test_composition_matches_sequential(self) -> None:
        """For a patch op produced by scaled→conv compose, applying conv₃
        via _compose_conv_with_patch must match the full sequential
        conv₁→α→conv₂→conv₃ numerically."""
        torch.manual_seed(100)
        W1 = torch.randn(3, 2, 3, 3)
        alpha = torch.randn(3, 4, 4)
        W2 = torch.randn(4, 3, 3, 3)
        W3 = torch.randn(5, 4, 3, 3)

        scaled = ScaledConv2dOperator(
            weight=W1,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            alpha=alpha,
            input_shape=(2, 4, 4),
            output_shape=(3, 4, 4),
        )
        patch2 = _compose_conv_with_scaled(
            scaled_op=scaled,
            weight2=W2,
            stride2=(1, 1),
            padding2=(1, 1),
            dilation2=(1, 1),
            groups2=1,
            output_shape=(4, 4, 4),
        )
        patch3 = _compose_conv_with_patch(
            patch_op=patch2,
            weight3=W3,
            stride3=(1, 1),
            padding3=(1, 1),
            dilation3=(1, 1),
            groups3=1,
            output_shape=(5, 4, 4),
        )
        # Kernel grows: (3+3-1) + (3-1) = 7.
        assert patch3.kernel_size == (7, 7)
        # Padding grows: (1+1) + 1 = 3.
        assert patch3.padding == (3, 3)

        for _ in range(3):
            x = torch.randn(*scaled.input_shape)
            z1 = scaled.apply(x)
            z2 = F.conv2d(z1.unsqueeze(0), W2, padding=1).squeeze(0)
            z3 = F.conv2d(z2.unsqueeze(0), W3, padding=1).squeeze(0)
            got = patch3.apply(x)
            assert torch.allclose(got, z3, atol=1e-4)

    def test_concretize_matches_dense(self) -> None:
        torch.manual_seed(101)
        W3 = torch.randn(5, 4, 3, 3)
        patch2 = _make_patch_via_compose(seed=101)
        patch3 = _compose_conv_with_patch(
            patch_op=patch2,
            weight3=W3,
            stride3=(1, 1),
            padding3=(1, 1),
            dilation3=(1, 1),
            groups3=1,
            output_shape=(5, 4, 4),
        )
        region = _make_region(patch3.input_shape, seed=101)
        assert torch.allclose(patch3.concretize_min(region), patch3.to_dense().concretize_min(region), atol=1e-4)
        assert torch.allclose(patch3.concretize_max(region), patch3.to_dense().concretize_max(region), atol=1e-4)

    def test_rejects_strided_conv(self) -> None:
        patch2 = _make_patch_via_compose(seed=0)
        W3 = torch.randn(5, patch2.output_shape[-3], 3, 3)
        with pytest.raises(NotImplementedError, match="stride"):
            _compose_conv_with_patch(
                patch_op=patch2,
                weight3=W3,
                stride3=(2, 2),
                padding3=(1, 1),
                dilation3=(1, 1),
                groups3=1,
                output_shape=(5, 2, 2),
            )

    def test_rejects_grouped_conv(self) -> None:
        patch2 = _make_patch_via_compose(seed=0)
        W3 = torch.randn(4, 2, 3, 3)
        with pytest.raises(NotImplementedError, match="groups"):
            _compose_conv_with_patch(
                patch_op=patch2,
                weight3=W3,
                stride3=(1, 1),
                padding3=(1, 1),
                dilation3=(1, 1),
                groups3=2,
                output_shape=(4, 4, 4),
            )
