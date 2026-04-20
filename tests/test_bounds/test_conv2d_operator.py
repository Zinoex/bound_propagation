"""Unit tests for :class:`Conv2dOperator`."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from bound_propagation.linear_operators import Conv2dOperator, DenseOperator
from bound_propagation.regions import HyperRectangle


def _make_region(shape: tuple[int, ...], seed: int = 0) -> HyperRectangle:
    torch.manual_seed(seed)
    lower = torch.randn(*shape)
    upper = lower + torch.rand(*shape) + 0.1
    return HyperRectangle(lower=lower, upper=upper)


class TestConv2dOperatorConstruction:
    def test_basic(self) -> None:
        w = torch.randn(4, 3, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(3, 5, 5),
            output_shape=(4, 5, 5),
        )
        assert op.output_shape == torch.Size((4, 5, 5))
        assert op.input_shape == torch.Size((3, 5, 5))
        assert op.dtype == w.dtype
        assert op.device == w.device

    def test_with_batch_in_output_shape(self) -> None:
        w = torch.randn(4, 3, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(3, 5, 5),
            output_shape=(2, 4, 5, 5),
        )
        assert op.output_shape == torch.Size((2, 4, 5, 5))

    def test_rejects_non_4d_weight(self) -> None:
        with pytest.raises(ValueError, match="4D"):
            Conv2dOperator(
                weight=torch.randn(3, 3),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                input_shape=(3, 5, 5),
                output_shape=(3, 5, 5),
            )

    def test_rejects_input_shape_wrong_rank(self) -> None:
        with pytest.raises(ValueError, match="3D"):
            Conv2dOperator(
                weight=torch.randn(4, 3, 3, 3),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                input_shape=(5, 5),
                output_shape=(4, 5, 5),
            )

    def test_rejects_output_c_mismatch(self) -> None:
        with pytest.raises(ValueError, match="output_shape"):
            Conv2dOperator(
                weight=torch.randn(4, 3, 3, 3),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                input_shape=(3, 5, 5),
                output_shape=(7, 5, 5),
            )


class TestConv2dOperatorApply:
    def test_apply_matches_F_conv2d(self) -> None:
        torch.manual_seed(0)
        w = torch.randn(4, 3, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(3, 5, 5),
            output_shape=(4, 5, 5),
        )
        x = torch.randn(3, 5, 5)
        got = op.apply(x)
        expected = F.conv2d(x.unsqueeze(0), w, padding=1).squeeze(0)
        assert torch.allclose(got, expected)

    def test_apply_with_batched_input(self) -> None:
        torch.manual_seed(1)
        w = torch.randn(4, 3, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(2, 2),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            input_shape=(3, 6, 6),
            output_shape=(4, 2, 2),
        )
        x = torch.randn(3, 3, 6, 6)
        got = op.apply(x)
        expected = F.conv2d(x, w, stride=2)
        assert torch.allclose(got, expected)

    def test_apply_bad_input_shape_raises(self) -> None:
        w = torch.randn(4, 3, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(3, 5, 5),
            output_shape=(4, 5, 5),
        )
        with pytest.raises(ValueError, match="trailing shape"):
            op.apply(torch.randn(3, 6, 6))


class TestConv2dOperatorConcretize:
    def test_concretize_matches_dense(self) -> None:
        """Conv2dOperator concretization must equal DenseOperator's on the
        materialized Jacobian."""
        torch.manual_seed(5)
        w = torch.randn(4, 3, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(3, 5, 5),
            output_shape=(4, 5, 5),
        )
        region = _make_region((3, 5, 5))

        conv_min = op.concretize_min(region)
        conv_max = op.concretize_max(region)
        dense = op.to_dense()
        dense_min = dense.concretize_min(region)
        dense_max = dense.concretize_max(region)

        assert torch.allclose(conv_min, dense_min, atol=1e-5)
        assert torch.allclose(conv_max, dense_max, atol=1e-5)

    def test_concretize_formula(self) -> None:
        """Direct check: ``min = conv(l, W_pos) + conv(u, W_neg)``."""
        torch.manual_seed(6)
        w = torch.randn(3, 2, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(2, 4, 4),
            output_shape=(3, 4, 4),
        )
        region = _make_region((2, 4, 4), seed=6)

        w_pos = w.clamp(min=0)
        w_neg = w.clamp(max=0)
        expected_min = F.conv2d(region.lower.unsqueeze(0), w_pos, padding=1).squeeze(0) + F.conv2d(
            region.upper.unsqueeze(0), w_neg, padding=1
        ).squeeze(0)
        assert torch.allclose(op.concretize_min(region), expected_min)

    def test_concretize_min_le_max(self) -> None:
        torch.manual_seed(7)
        w = torch.randn(2, 3, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            input_shape=(3, 5, 5),
            output_shape=(2, 3, 3),
        )
        region = _make_region((3, 5, 5), seed=7)
        assert torch.all(op.concretize_min(region) <= op.concretize_max(region) + 1e-6)


class TestConv2dOperatorToDense:
    def test_to_dense_apply_identity(self) -> None:
        """The Jacobian produced by to_dense must apply identically to the
        conv itself on any input."""
        torch.manual_seed(8)
        w = torch.randn(3, 2, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(2, 4, 4),
            output_shape=(3, 4, 4),
        )
        dense = op.to_dense()
        x = torch.randn(2, 4, 4)
        via_conv = op.apply(x)
        via_dense = dense.apply(x)
        assert torch.allclose(via_conv, via_dense, atol=1e-5)

    def test_to_dense_with_batched_output_shape(self) -> None:
        w = torch.randn(3, 2, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(2, 4, 4),
            output_shape=(5, 3, 4, 4),
        )
        dense = op.to_dense()
        assert dense.output_shape == torch.Size((5, 3, 4, 4))
        assert dense.input_shape == torch.Size((2, 4, 4))


class TestConv2dOperatorApplyTranspose:
    def test_apply_transpose_matches_dense(self) -> None:
        torch.manual_seed(9)
        w = torch.randn(3, 2, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(2, 4, 4),
            output_shape=(3, 4, 4),
        )
        dense = op.to_dense()
        y = torch.randn(3, 4, 4)
        via_conv = op.apply_transpose(y)
        via_dense = dense.apply_transpose(y)
        assert torch.allclose(via_conv, via_dense, atol=1e-5)

    def test_apply_adjoint_identity(self) -> None:
        """<y, Wx> == <W^T y, x>."""
        torch.manual_seed(10)
        w = torch.randn(3, 2, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(2, 4, 4),
            output_shape=(3, 4, 4),
        )
        x = torch.randn(2, 4, 4)
        y = torch.randn(3, 4, 4)
        lhs = (y * op.apply(x)).sum()
        rhs = (op.apply_transpose(y) * x).sum()
        assert torch.allclose(lhs, rhs, atol=1e-5)


class TestConv2dOperatorAlgebra:
    def test_neg(self) -> None:
        torch.manual_seed(11)
        w = torch.randn(3, 2, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(2, 4, 4),
            output_shape=(3, 4, 4),
        )
        neg_op = op.neg()
        assert isinstance(neg_op, Conv2dOperator)
        x = torch.randn(2, 4, 4)
        assert torch.allclose(neg_op.apply(x), -op.apply(x), atol=1e-5)

    def test_add_same_hyperparams_stays_structured(self) -> None:
        torch.manual_seed(12)
        w1 = torch.randn(3, 2, 3, 3)
        w2 = torch.randn(3, 2, 3, 3)
        params: dict = {
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "input_shape": (2, 4, 4),
            "output_shape": (3, 4, 4),
        }
        a = Conv2dOperator(weight=w1, **params)
        b = Conv2dOperator(weight=w2, **params)
        summed = a.add(b)
        assert isinstance(summed, Conv2dOperator)
        x = torch.randn(2, 4, 4)
        assert torch.allclose(summed.apply(x), a.apply(x) + b.apply(x), atol=1e-5)

    def test_add_mismatched_falls_back_to_dense(self) -> None:
        w = torch.randn(3, 2, 3, 3)
        conv_op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(2, 4, 4),
            output_shape=(3, 4, 4),
        )
        dense_op = DenseOperator(torch.randn(3, 4, 4, 2, 4, 4), output_shape=(3, 4, 4))
        summed = conv_op.add(dense_op)
        assert isinstance(summed, DenseOperator)


class TestConv2dOperatorShapeOpsFallback:
    def test_flatten_output_falls_back_to_dense(self) -> None:
        w = torch.randn(3, 2, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(2, 4, 4),
            output_shape=(3, 4, 4),
        )
        flattened = op.flatten_output(0, 2)
        assert isinstance(flattened, DenseOperator)
        assert flattened.output_shape == torch.Size((48,))

    def test_scale_returns_scaled_conv_operator(self) -> None:
        """Conv2dOperator.scale now returns a ScaledConv2dOperator (structural)."""
        from bound_propagation.linear_operators import ScaledConv2dOperator

        w = torch.randn(3, 2, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(2, 4, 4),
            output_shape=(3, 4, 4),
        )
        factor = torch.randn(3, 4, 4)
        scaled = op.scale(factor)
        assert isinstance(scaled, ScaledConv2dOperator)
        # Verify numerical correctness against the materialized dense path.
        dense_scaled = op.to_dense().scale(factor)
        assert torch.allclose(scaled.to_dense().tensor, dense_scaled.tensor, atol=1e-5)


class TestConv2dOperatorHousekeeping:
    def test_clone_is_independent(self) -> None:
        w = torch.randn(3, 2, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(2, 4, 4),
            output_shape=(3, 4, 4),
        )
        cloned = op.clone()
        # Mutate the original's weight — clone should be unaffected.
        op._weight.fill_(0.0)
        assert not torch.allclose(cloned._weight, op._weight)

    def test_to_device(self) -> None:
        w = torch.randn(3, 2, 3, 3)
        op = Conv2dOperator(
            weight=w,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            input_shape=(2, 4, 4),
            output_shape=(3, 4, 4),
        )
        moved = op.to("cpu")  # trivially same device, but exercises the code path
        assert isinstance(moved, Conv2dOperator)
        assert moved.device == op.device
