"""Unit tests for :class:`Conv2dPatchOperator` and the conv ∘ scaled-conv
composition helper."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from bound_propagation.linear_operators import (
    Conv2dPatchOperator,
    DenseOperator,
    ScaledConv2dOperator,
    _compose_conv_with_scaled,
)
from bound_propagation.regions import HyperRectangle


def _make_region(shape: tuple[int, ...], seed: int = 0) -> HyperRectangle:
    torch.manual_seed(seed)
    lower = torch.randn(*shape)
    upper = lower + torch.rand(*shape) + 0.1
    return HyperRectangle(lower=lower, upper=upper)


def _make_scaled(
    *,
    c_in: int = 2,
    c_y: int = 3,
    h: int = 4,
    w: int = 4,
    padding: tuple[int, int] = (1, 1),
    seed: int = 0,
) -> ScaledConv2dOperator:
    torch.manual_seed(seed)
    w_kernel = torch.randn(c_y, c_in, 3, 3)
    alpha = torch.randn(c_y, h, w)
    return ScaledConv2dOperator(
        weight=w_kernel,
        stride=(1, 1),
        padding=padding,
        dilation=(1, 1),
        groups=1,
        alpha=alpha,
        input_shape=(c_in, h, w),
        output_shape=(c_y, h, w),
    )


def _compose_for_test(scaled_seed: int = 0, w2_seed: int = 1, c_z: int = 4) -> Conv2dPatchOperator:
    scaled = _make_scaled(seed=scaled_seed)
    torch.manual_seed(w2_seed)
    w2 = torch.randn(c_z, scaled.weight.shape[0], 3, 3)
    return _compose_conv_with_scaled(
        scaled_op=scaled,
        weight2=w2,
        stride2=(1, 1),
        padding2=(1, 1),
        dilation2=(1, 1),
        groups2=1,
        output_shape=(c_z, 4, 4),
    )


class TestConv2dPatchOperatorConstruction:
    def test_basic(self) -> None:
        op = _compose_for_test()
        assert op.output_shape == torch.Size((4, 4, 4))
        assert op.input_shape == torch.Size((2, 4, 4))
        # Combined kernel is 3+3-1 = 5.
        assert op.kernel_size == (5, 5)
        # Combined padding is 1+1 = 2.
        assert op.padding == (2, 2)

    def test_rejects_groups_other_than_one(self) -> None:
        patches = torch.zeros(3, 4, 4, 2, 5, 5)
        with pytest.raises(NotImplementedError, match="groups"):
            Conv2dPatchOperator(
                patches=patches,
                stride=(1, 1),
                padding=(2, 2),
                dilation=(1, 1),
                groups=2,
                input_shape=(2, 4, 4),
                output_shape=(3, 4, 4),
            )

    def test_accepts_non_unit_stride(self) -> None:
        """stride/dilation != 1 is now supported at operator level (compose
        helpers still require stride=dilation=1)."""
        # patches shape: (*output_shape, C_in, k_h, k_w). Output here is (2, 3, 3).
        patches = torch.zeros(2, 3, 3, 2, 3, 3)
        op = Conv2dPatchOperator(
            patches=patches,
            stride=(2, 2),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            input_shape=(2, 7, 7),  # (H_out - 1) * 2 + 3 = 2*2 + 3 = 7.
            output_shape=(2, 3, 3),
        )
        assert op.stride == (2, 2)

    def test_rejects_patches_shape_mismatch(self) -> None:
        bad_patches = torch.zeros(3, 4, 4, 7, 5, 5)  # C_in=7 doesn't match input_shape[0]=2
        with pytest.raises(ValueError, match="patches shape"):
            Conv2dPatchOperator(
                patches=bad_patches,
                stride=(1, 1),
                padding=(2, 2),
                dilation=(1, 1),
                groups=1,
                input_shape=(2, 4, 4),
                output_shape=(3, 4, 4),
            )


class TestConv2dPatchOperatorApply:
    def test_matches_manual_conv_alpha_conv(self) -> None:
        """``patch_op.apply(x) == conv2(alpha * conv1(x))`` for the composed op."""
        torch.manual_seed(0)
        scaled = _make_scaled(seed=0)
        w2 = torch.randn(4, scaled.weight.shape[0], 3, 3)
        composed = _compose_conv_with_scaled(
            scaled_op=scaled,
            weight2=w2,
            stride2=(1, 1),
            padding2=(1, 1),
            dilation2=(1, 1),
            groups2=1,
            output_shape=(4, 4, 4),
        )
        x = torch.randn(*scaled.input_shape)
        got = composed.apply(x)
        y = scaled.apply(x)
        expected = F.conv2d(y.unsqueeze(0), w2, padding=1).squeeze(0)
        assert torch.allclose(got, expected, atol=1e-5)


class TestConv2dPatchOperatorToDense:
    def test_to_dense_applies_identically(self) -> None:
        op = _compose_for_test(scaled_seed=1, w2_seed=2)
        dense = op.to_dense()
        x = torch.randn(*op.input_shape)
        assert torch.allclose(op.apply(x), dense.apply(x), atol=1e-5)


class TestConv2dPatchOperatorConcretize:
    def test_matches_dense(self) -> None:
        op = _compose_for_test(scaled_seed=2, w2_seed=3)
        region = _make_region(op.input_shape, seed=2)
        assert torch.allclose(op.concretize_min(region), op.to_dense().concretize_min(region), atol=1e-5)
        assert torch.allclose(op.concretize_max(region), op.to_dense().concretize_max(region), atol=1e-5)

    def test_min_le_max(self) -> None:
        op = _compose_for_test(scaled_seed=3, w2_seed=4)
        region = _make_region(op.input_shape, seed=3)
        assert torch.all(op.concretize_min(region) <= op.concretize_max(region) + 1e-5)


class TestConv2dPatchOperatorAlgebra:
    def test_neg(self) -> None:
        op = _compose_for_test(scaled_seed=4, w2_seed=5)
        neg = op.neg()
        assert isinstance(neg, Conv2dPatchOperator)
        x = torch.randn(*op.input_shape)
        assert torch.allclose(neg.apply(x), -op.apply(x), atol=1e-5)

    def test_scale(self) -> None:
        op = _compose_for_test(scaled_seed=5, w2_seed=6)
        factor = torch.randn(*op.output_shape)
        scaled = op.scale(factor)
        assert isinstance(scaled, Conv2dPatchOperator)
        # scaled patches should equal original patches * factor broadcast.
        expected_patches = op.patches * factor.reshape(*factor.shape, 1, 1, 1)
        assert torch.allclose(scaled.patches, expected_patches, atol=1e-5)

    def test_add_same_hyperparams_stays_structured(self) -> None:
        op_a = _compose_for_test(scaled_seed=6, w2_seed=7)
        op_b = _compose_for_test(scaled_seed=7, w2_seed=8)
        summed = op_a.add(op_b)
        assert isinstance(summed, Conv2dPatchOperator)
        assert torch.allclose(summed.patches, op_a.patches + op_b.patches)

    def test_add_with_dense_falls_back(self) -> None:
        op = _compose_for_test(scaled_seed=8, w2_seed=9)
        tensor = torch.randn(*op.output_shape, *op.input_shape)
        dense = DenseOperator(tensor, output_shape=op.output_shape)
        summed = op.add(dense)
        assert isinstance(summed, DenseOperator)


class TestCompositionHelper:
    def test_composition_matches_sequential(self) -> None:
        """Direct numerical check: ``composed.apply(x) == conv2(scaled.apply(x))``."""
        torch.manual_seed(10)
        scaled = _make_scaled(seed=10)
        w2 = torch.randn(5, scaled.weight.shape[0], 3, 3)
        composed = _compose_conv_with_scaled(
            scaled_op=scaled,
            weight2=w2,
            stride2=(1, 1),
            padding2=(1, 1),
            dilation2=(1, 1),
            groups2=1,
            output_shape=(5, 4, 4),
        )
        for _ in range(5):
            x = torch.randn(*scaled.input_shape)
            composed_out = composed.apply(x)
            y = scaled.apply(x)
            expected = F.conv2d(y.unsqueeze(0), w2, padding=1).squeeze(0)
            assert torch.allclose(composed_out, expected, atol=1e-5)

    def test_rejects_non_unit_stride(self) -> None:
        scaled = _make_scaled(seed=0)
        w2 = torch.randn(4, scaled.weight.shape[0], 3, 3)
        with pytest.raises(NotImplementedError, match="stride"):
            _compose_conv_with_scaled(
                scaled_op=scaled,
                weight2=w2,
                stride2=(2, 2),
                padding2=(1, 1),
                dilation2=(1, 1),
                groups2=1,
                output_shape=(4, 2, 2),
            )

    def test_rejects_non_unit_groups(self) -> None:
        # scaled must also have groups=1; a groups=2 request on the second conv is rejected.
        scaled = _make_scaled(seed=0, c_y=4, c_in=4)
        w2 = torch.randn(4, 2, 3, 3)  # groups=2 expects weight.shape[1] = C_y/groups = 2.
        with pytest.raises(NotImplementedError, match="groups"):
            _compose_conv_with_scaled(
                scaled_op=scaled,
                weight2=w2,
                stride2=(1, 1),
                padding2=(1, 1),
                dilation2=(1, 1),
                groups2=2,
                output_shape=(4, 4, 4),
            )


class TestConv2dPatchOperatorHousekeeping:
    def test_clone_is_independent(self) -> None:
        op = _compose_for_test(scaled_seed=20, w2_seed=21)
        cloned = op.clone()
        op._patches.fill_(0.0)
        assert not torch.allclose(cloned._patches, op._patches)

    def test_to_device(self) -> None:
        op = _compose_for_test(scaled_seed=22, w2_seed=23)
        moved = op.to("cpu")
        assert isinstance(moved, Conv2dPatchOperator)
