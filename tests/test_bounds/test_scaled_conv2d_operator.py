"""Unit tests for :class:`ScaledConv2dOperator`."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from bound_propagation.linear_operators import Conv2dOperator, DenseOperator, ScaledConv2dOperator
from bound_propagation.regions import HyperRectangle


def _make_region(shape: tuple[int, ...], seed: int = 0) -> HyperRectangle:
    torch.manual_seed(seed)
    lower = torch.randn(*shape)
    upper = lower + torch.rand(*shape) + 0.1
    return HyperRectangle(lower=lower, upper=upper)


def _make_scaled(
    *,
    c_in: int = 2,
    c_out: int = 3,
    h_in: int = 4,
    w_in: int = 4,
    h_out: int | None = None,
    w_out: int | None = None,
    padding: tuple[int, int] = (1, 1),
    stride: tuple[int, int] = (1, 1),
    alpha_seed: int = 0,
    weight_seed: int = 0,
) -> ScaledConv2dOperator:
    h_out = h_in if h_out is None else h_out
    w_out = w_in if w_out is None else w_out
    torch.manual_seed(weight_seed)
    w = torch.randn(c_out, c_in, 3, 3)
    torch.manual_seed(alpha_seed)
    alpha = torch.randn(c_out, h_out, w_out)
    return ScaledConv2dOperator(
        weight=w,
        stride=stride,
        padding=padding,
        dilation=(1, 1),
        groups=1,
        alpha=alpha,
        input_shape=(c_in, h_in, w_in),
        output_shape=(c_out, h_out, w_out),
    )


class TestScaledConv2dOperatorConstruction:
    def test_basic(self) -> None:
        op = _make_scaled()
        assert op.output_shape == torch.Size((3, 4, 4))
        assert op.input_shape == torch.Size((2, 4, 4))

    def test_rejects_alpha_shape_mismatch(self) -> None:
        w = torch.randn(3, 2, 3, 3)
        with pytest.raises(ValueError, match="alpha.shape"):
            ScaledConv2dOperator(
                weight=w,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=1,
                alpha=torch.randn(3, 5, 5),  # mismatches output_shape
                input_shape=(2, 4, 4),
                output_shape=(3, 4, 4),
            )


class TestScaledConv2dOperatorApply:
    def test_apply_matches_alpha_times_conv(self) -> None:
        torch.manual_seed(1)
        op = _make_scaled()
        x = torch.randn(*op.input_shape)
        got = op.apply(x)
        expected = op.alpha * F.conv2d(x.unsqueeze(0), op.weight, padding=1).squeeze(0)
        assert torch.allclose(got, expected)

    def test_apply_with_batched_input(self) -> None:
        torch.manual_seed(2)
        op = _make_scaled(stride=(2, 2), padding=(0, 0), h_in=6, w_in=6, h_out=2, w_out=2)
        x = torch.randn(3, *op.input_shape)
        got = op.apply(x)
        expected = op.alpha * F.conv2d(x, op.weight, stride=2)
        assert torch.allclose(got, expected)


class TestScaledConv2dOperatorApplyTranspose:
    def test_matches_dense(self) -> None:
        op = _make_scaled(alpha_seed=3)
        dense = op.to_dense()
        y = torch.randn(*op.output_shape)
        via_struct = op.apply_transpose(y)
        via_dense = dense.apply_transpose(y)
        assert torch.allclose(via_struct, via_dense, atol=1e-5)

    def test_adjoint_identity(self) -> None:
        """<y, (alpha*conv)(x)> == <(alpha*conv)^T(y), x>."""
        op = _make_scaled(alpha_seed=4)
        x = torch.randn(*op.input_shape)
        y = torch.randn(*op.output_shape)
        lhs = (y * op.apply(x)).sum()
        rhs = (op.apply_transpose(y) * x).sum()
        assert torch.allclose(lhs, rhs, atol=1e-5)


class TestScaledConv2dOperatorConcretize:
    def test_matches_dense(self) -> None:
        op = _make_scaled(alpha_seed=5)
        region = _make_region(op.input_shape, seed=5)
        dense = op.to_dense()
        assert torch.allclose(op.concretize_min(region), dense.concretize_min(region), atol=1e-5)
        assert torch.allclose(op.concretize_max(region), dense.concretize_max(region), atol=1e-5)

    def test_min_le_max(self) -> None:
        op = _make_scaled(alpha_seed=6)
        region = _make_region(op.input_shape, seed=6)
        assert torch.all(op.concretize_min(region) <= op.concretize_max(region) + 1e-6)

    def test_sign_decomposition_formula(self) -> None:
        """Explicit formula: min = alpha_pos * conv_min + alpha_neg * conv_max."""
        op = _make_scaled(alpha_seed=7)
        region = _make_region(op.input_shape, seed=7)

        w_pos = op.weight.clamp(min=0)
        w_neg = op.weight.clamp(max=0)
        conv_min = F.conv2d(region.lower.unsqueeze(0), w_pos, padding=1).squeeze(0) + F.conv2d(
            region.upper.unsqueeze(0), w_neg, padding=1
        ).squeeze(0)
        conv_max = F.conv2d(region.upper.unsqueeze(0), w_pos, padding=1).squeeze(0) + F.conv2d(
            region.lower.unsqueeze(0), w_neg, padding=1
        ).squeeze(0)
        alpha_pos = op.alpha.clamp(min=0)
        alpha_neg = op.alpha.clamp(max=0)
        expected_min = alpha_pos * conv_min + alpha_neg * conv_max
        assert torch.allclose(op.concretize_min(region), expected_min, atol=1e-5)


class TestScaledConv2dOperatorToDense:
    def test_apply_round_trip(self) -> None:
        op = _make_scaled(alpha_seed=8)
        dense = op.to_dense()
        x = torch.randn(*op.input_shape)
        assert torch.allclose(op.apply(x), dense.apply(x), atol=1e-5)


class TestScaledConv2dOperatorAlgebra:
    def test_neg(self) -> None:
        op = _make_scaled(alpha_seed=9)
        neg = op.neg()
        assert isinstance(neg, ScaledConv2dOperator)
        x = torch.randn(*op.input_shape)
        assert torch.allclose(neg.apply(x), -op.apply(x), atol=1e-5)

    def test_scale_composes(self) -> None:
        op = _make_scaled(alpha_seed=10)
        factor = torch.randn(*op.output_shape)
        scaled = op.scale(factor)
        assert isinstance(scaled, ScaledConv2dOperator)
        assert torch.allclose(scaled.alpha, op.alpha * factor)

    def test_scale_by_scalar(self) -> None:
        op = _make_scaled(alpha_seed=11)
        factor = torch.tensor(2.5)
        scaled = op.scale(factor)
        assert isinstance(scaled, ScaledConv2dOperator)
        assert torch.allclose(scaled.alpha, op.alpha * 2.5)

    def test_add_same_conv_stays_structured(self) -> None:
        torch.manual_seed(12)
        w = torch.randn(3, 2, 3, 3)
        alpha_a = torch.randn(3, 4, 4)
        alpha_b = torch.randn(3, 4, 4)
        common: dict = {
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "input_shape": (2, 4, 4),
            "output_shape": (3, 4, 4),
        }
        a = ScaledConv2dOperator(weight=w, alpha=alpha_a, **common)
        b = ScaledConv2dOperator(weight=w, alpha=alpha_b, **common)
        summed = a.add(b)
        assert isinstance(summed, ScaledConv2dOperator)
        assert torch.allclose(summed.alpha, alpha_a + alpha_b)
        assert summed.weight is w  # same weight object

    def test_add_with_plain_conv_same_weight(self) -> None:
        """ScaledConv + Conv on the same weight should collapse: alpha' = alpha + 1."""
        torch.manual_seed(13)
        w = torch.randn(3, 2, 3, 3)
        alpha = torch.randn(3, 4, 4)
        common: dict = {
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "input_shape": (2, 4, 4),
            "output_shape": (3, 4, 4),
        }
        scaled = ScaledConv2dOperator(weight=w, alpha=alpha, **common)
        plain = Conv2dOperator(weight=w, **common)
        summed = scaled.add(plain)
        assert isinstance(summed, ScaledConv2dOperator)
        assert torch.allclose(summed.alpha, alpha + 1)

    def test_add_different_weight_falls_back_to_dense(self) -> None:
        a = _make_scaled(weight_seed=0, alpha_seed=0)
        b = _make_scaled(weight_seed=1, alpha_seed=0)
        summed = a.add(b)
        assert isinstance(summed, DenseOperator)

    def test_add_mismatched_hyperparams_falls_back(self) -> None:
        """Same output shape via different hyperparams: add must fall back to dense.

        Use two kernels of size 3×3 with different stride/padding combos that
        still produce the same output shape (here, padding=(1,1) stride=1
        vs padding=(2,2) stride=1 on a 3×3 kernel — both preserve the 4×4
        spatial output).
        """
        a = _make_scaled(padding=(1, 1), stride=(1, 1))
        # b: same output shape but different padding via a different kernel.
        torch.manual_seed(99)
        w_b = torch.randn(3, 2, 5, 5)  # 5x5 kernel
        b = ScaledConv2dOperator(
            weight=w_b,
            stride=(1, 1),
            padding=(2, 2),  # matches output H=W=4
            dilation=(1, 1),
            groups=1,
            alpha=torch.randn(3, 4, 4),
            input_shape=(2, 4, 4),
            output_shape=(3, 4, 4),
        )
        summed = a.add(b)
        # Different weight shape → structural fast path rejects → dense fallback.
        assert isinstance(summed, DenseOperator)


class TestConv2dOperatorScaleOverride:
    """Conv2dOperator.scale should return a ScaledConv2dOperator."""

    def test_scale_returns_scaled(self) -> None:
        torch.manual_seed(14)
        w = torch.randn(3, 2, 3, 3)
        conv = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(2, 4, 4),
            output_shape=(3, 4, 4),
        )
        factor = torch.randn(3, 4, 4)
        scaled = conv.scale(factor)
        assert isinstance(scaled, ScaledConv2dOperator)
        assert scaled.weight is w
        assert torch.allclose(scaled.alpha, factor)

    def test_concretize_equals_dense_path(self) -> None:
        """Structural scale then concretize must equal dense scale then concretize."""
        torch.manual_seed(15)
        w = torch.randn(3, 2, 3, 3)
        conv = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(2, 4, 4),
            output_shape=(3, 4, 4),
        )
        factor = torch.randn(3, 4, 4)
        struct_scaled = conv.scale(factor)
        dense_scaled = conv.to_dense().scale(factor)

        region = _make_region((2, 4, 4), seed=15)
        assert torch.allclose(struct_scaled.concretize_min(region), dense_scaled.concretize_min(region), atol=1e-5)
        assert torch.allclose(struct_scaled.concretize_max(region), dense_scaled.concretize_max(region), atol=1e-5)


class TestScaledConv2dOperatorHousekeeping:
    def test_clone_is_independent(self) -> None:
        op = _make_scaled(alpha_seed=16)
        cloned = op.clone()
        op._alpha.fill_(0.0)
        assert not torch.allclose(cloned._alpha, op._alpha)

    def test_to_device(self) -> None:
        op = _make_scaled(alpha_seed=17)
        moved = op.to("cpu")
        assert isinstance(moved, ScaledConv2dOperator)
