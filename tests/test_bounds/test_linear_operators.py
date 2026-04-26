"""Unit tests for the ``LinearOperator`` abstraction and ``DenseOperator``."""

from __future__ import annotations

import pytest
import torch

from bound_propagation.linear_operators import (
    DenseOperator,
    IdentityOperator,
    ReshapeOperator,
    ZeroOperator,
    cat_output,
    stack_output,
)
from bound_propagation.regions import HyperRectangle


def _make_region(shape: tuple[int, ...]) -> HyperRectangle:
    torch.manual_seed(0)
    lower = torch.randn(*shape)
    upper = lower + torch.rand(*shape) + 0.1
    return HyperRectangle(lower=lower, upper=upper)


class TestDenseOperatorConstruction:
    def test_basic(self) -> None:
        tensor = torch.randn(2, 3, 4)
        op = DenseOperator(tensor, output_shape=(2, 3))
        assert op.output_shape == torch.Size((2, 3))
        assert op.input_shape == torch.Size((4,))
        assert op.output_ndim == 2
        assert op.input_ndim == 1
        assert op.dtype == tensor.dtype
        assert op.device == tensor.device

    def test_zero_output_dims(self) -> None:
        tensor = torch.randn(5)
        op = DenseOperator(tensor, output_shape=())
        assert op.output_shape == torch.Size(())
        assert op.input_shape == torch.Size((5,))

    def test_zero_input_dims(self) -> None:
        tensor = torch.randn(2, 3)
        op = DenseOperator(tensor, output_shape=(2, 3))
        assert op.input_shape == torch.Size(())

    def test_rank_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            DenseOperator(torch.randn(2), output_shape=(2, 3))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="leading shape"):
            DenseOperator(torch.randn(2, 3, 4), output_shape=(5, 6))


class TestDenseOperatorAlgebra:
    def test_neg(self) -> None:
        tensor = torch.randn(2, 3, 4)
        op = DenseOperator(tensor, output_shape=(2, 3))
        result = op.neg()
        assert torch.equal(result.tensor, -tensor)
        assert result.output_shape == op.output_shape

    def test_scale_scalar(self) -> None:
        tensor = torch.randn(2, 3, 4)
        op = DenseOperator(tensor, output_shape=(2, 3))
        factor = torch.tensor(2.0)
        result = op.scale(factor)
        assert torch.allclose(result.tensor, 2.0 * tensor)

    def test_scale_per_output_element(self) -> None:
        tensor = torch.randn(2, 3, 4, 5)
        op = DenseOperator(tensor, output_shape=(2, 3))
        factor = torch.randn(2, 3)
        result = op.scale(factor)
        expected = tensor * factor.reshape(2, 3, 1, 1)
        assert torch.allclose(result.tensor, expected)

    def test_scale_rank_too_large_raises(self) -> None:
        tensor = torch.randn(2, 3, 4)
        op = DenseOperator(tensor, output_shape=(2, 3))
        with pytest.raises(ValueError, match="rank"):
            op.scale(torch.randn(2, 3, 4))

    def test_add(self) -> None:
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 3, 4)
        op_a = DenseOperator(a, output_shape=(2, 3))
        op_b = DenseOperator(b, output_shape=(2, 3))
        result = op_a.add(op_b)
        assert torch.allclose(result.to_dense().tensor, a + b)

    def test_sub(self) -> None:
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 3, 4)
        op_a = DenseOperator(a, output_shape=(2, 3))
        op_b = DenseOperator(b, output_shape=(2, 3))
        result = op_a.sub(op_b)
        assert torch.allclose(result.to_dense().tensor, a - b)

    def test_add_mismatched_shape_raises(self) -> None:
        op_a = DenseOperator(torch.randn(2, 3, 4), output_shape=(2, 3))
        op_b = DenseOperator(torch.randn(2, 4, 4), output_shape=(2, 4))
        with pytest.raises(ValueError, match="output shapes"):
            op_a.add(op_b)


class TestDenseOperatorConcretize:
    def test_concretize_min_matches_manual(self) -> None:
        torch.manual_seed(42)
        linear = torch.randn(4, 5)  # output_shape=(4,), input_shape=(5,)
        op = DenseOperator(linear, output_shape=(4,))
        region = _make_region((5,))
        got = op.concretize_min(region)
        linear_pos = linear.clamp(min=0)
        linear_neg = linear.clamp(max=0)
        expected = linear_pos @ region.lower + linear_neg @ region.upper
        assert torch.allclose(got, expected)

    def test_concretize_max_matches_manual(self) -> None:
        torch.manual_seed(42)
        linear = torch.randn(4, 5)
        op = DenseOperator(linear, output_shape=(4,))
        region = _make_region((5,))
        got = op.concretize_max(region)
        linear_pos = linear.clamp(min=0)
        linear_neg = linear.clamp(max=0)
        expected = linear_pos @ region.upper + linear_neg @ region.lower
        assert torch.allclose(got, expected)

    def test_concretize_min_le_max(self) -> None:
        torch.manual_seed(7)
        linear = torch.randn(3, 4, 6)
        op = DenseOperator(linear, output_shape=(3, 4))
        region = _make_region((6,))
        low = op.concretize_min(region)
        high = op.concretize_max(region)
        assert torch.all(low <= high + 1e-6)

    def test_concretize_with_batch_region(self) -> None:
        torch.manual_seed(3)
        # Region has (batch=2, input=5); linear: output=(2,4), input=(5,)
        region = _make_region((2, 5))
        linear = torch.randn(2, 4, 5)
        op = DenseOperator(linear, output_shape=(2, 4))
        low = op.concretize_min(region)
        high = op.concretize_max(region)
        assert low.shape == torch.Size((2, 4))
        assert torch.all(low <= high + 1e-6)


class TestDenseOperatorApplyAdjoint:
    def test_apply_contracts_input_axes(self) -> None:
        torch.manual_seed(0)
        linear = torch.randn(3, 4)
        op = DenseOperator(linear, output_shape=(3,))
        x = torch.randn(4)
        got = op.apply(x)
        expected = linear @ x
        assert torch.allclose(got, expected)

    def test_apply_transpose_contracts_output_axes(self) -> None:
        torch.manual_seed(0)
        linear = torch.randn(3, 4)
        op = DenseOperator(linear, output_shape=(3,))
        y = torch.randn(3)
        got = op.apply_transpose(y)
        expected = linear.t() @ y
        assert torch.allclose(got, expected)

    def test_apply_inner_product_identity(self) -> None:
        # <y, Wx> == <W^T y, x>
        torch.manual_seed(1)
        linear = torch.randn(5, 6)
        op = DenseOperator(linear, output_shape=(5,))
        x = torch.randn(6)
        y = torch.randn(5)
        lhs = (y * op.apply(x)).sum()
        rhs = (op.apply_transpose(y) * x).sum()
        assert torch.allclose(lhs, rhs, atol=1e-5)


class TestDenseOperatorOutputShapeOps:
    def test_flatten_output(self) -> None:
        tensor = torch.randn(2, 3, 4, 5)
        op = DenseOperator(tensor, output_shape=(2, 3, 4))
        result = op.flatten_output(0, 2)
        assert result.output_shape == torch.Size((24,))
        assert result.input_shape == torch.Size((5,))
        assert torch.equal(result.tensor, tensor.reshape(24, 5))

    def test_reshape_output(self) -> None:
        tensor = torch.randn(6, 5)
        op = DenseOperator(tensor, output_shape=(6,))
        result = op.reshape_output((2, 3))
        assert result.output_shape == torch.Size((2, 3))
        assert result.input_shape == torch.Size((5,))
        assert torch.equal(result.tensor, tensor.reshape(2, 3, 5))

    def test_unsqueeze_squeeze(self) -> None:
        tensor = torch.randn(3, 4)
        op = DenseOperator(tensor, output_shape=(3,))
        expanded = op.unsqueeze_output(0)
        assert expanded.output_shape == torch.Size((1, 3))
        squeezed = expanded.squeeze_output(0)
        assert squeezed.output_shape == torch.Size((3,))

    def test_transpose_permute(self) -> None:
        tensor = torch.randn(2, 3, 4, 5)
        op = DenseOperator(tensor, output_shape=(2, 3, 4))
        transposed = op.transpose_output(0, 2)
        assert transposed.output_shape == torch.Size((4, 3, 2))
        assert transposed.tensor.shape == torch.Size((4, 3, 2, 5))
        permuted = op.permute_output((2, 0, 1))
        assert permuted.output_shape == torch.Size((4, 2, 3))
        assert permuted.tensor.shape == torch.Size((4, 2, 3, 5))

    def test_select_output(self) -> None:
        tensor = torch.randn(2, 3, 4, 5)
        op = DenseOperator(tensor, output_shape=(2, 3, 4))
        selected = op.select_output(1, 2)
        assert selected.output_shape == torch.Size((2, 4))
        assert torch.equal(selected.tensor, tensor[:, 2, :, :])

    def test_sum_mean_output(self) -> None:
        tensor = torch.randn(2, 3, 4, 5)
        op = DenseOperator(tensor, output_shape=(2, 3, 4))
        summed = op.sum_output(dim=1, keepdim=False)
        assert summed.output_shape == torch.Size((2, 4))
        assert torch.allclose(summed.tensor, tensor.sum(dim=1))
        meaned = op.mean_output(dim=(0, 2), keepdim=True)
        assert meaned.output_shape == torch.Size((1, 3, 1))
        assert torch.allclose(meaned.tensor, tensor.mean(dim=(0, 2), keepdim=True))

    def test_sum_all_dims(self) -> None:
        tensor = torch.randn(2, 3, 5)
        op = DenseOperator(tensor, output_shape=(2, 3))
        summed = op.sum_output(dim=None, keepdim=False)
        assert summed.output_shape == torch.Size(())
        assert torch.allclose(summed.tensor, tensor.sum(dim=(0, 1)))

    def test_getitem_output_with_ellipsis(self) -> None:
        tensor = torch.randn(4, 3, 5)
        op = DenseOperator(tensor, output_shape=(4, 3))
        sliced = op.getitem_output((slice(1, 3), Ellipsis))
        assert sliced.output_shape == torch.Size((2, 3))
        assert sliced.input_shape == torch.Size((5,))


class TestCatStack:
    def test_cat(self) -> None:
        a = torch.randn(2, 3, 5)
        b = torch.randn(2, 4, 5)
        op_a = DenseOperator(a, output_shape=(2, 3))
        op_b = DenseOperator(b, output_shape=(2, 4))
        result = cat_output([op_a, op_b], dim=1)
        assert result.output_shape == torch.Size((2, 7))
        assert torch.equal(result.to_dense().tensor, torch.cat([a, b], dim=1))

    def test_stack(self) -> None:
        a = torch.randn(3, 5)
        b = torch.randn(3, 5)
        op_a = DenseOperator(a, output_shape=(3,))
        op_b = DenseOperator(b, output_shape=(3,))
        result = stack_output([op_a, op_b], dim=0)
        assert result.output_shape == torch.Size((2, 3))
        assert torch.equal(result.to_dense().tensor, torch.stack([a, b], dim=0))


class TestIdentityOperator:
    def test_basic_3d(self) -> None:
        op = IdentityOperator(feature_shape=(2, 4, 4), dtype=torch.float32, device=torch.device("cpu"))
        assert op.output_shape == torch.Size((2, 4, 4))
        assert op.input_shape == torch.Size((2, 4, 4))

    def test_with_batch_ones(self) -> None:
        op = IdentityOperator(feature_shape=(3, 4), dtype=torch.float32, device=torch.device("cpu"), batch_shape=(1,))
        assert op.output_shape == torch.Size((1, 3, 4))
        assert op.input_shape == torch.Size((3, 4))

    def test_apply_is_identity(self) -> None:
        op = IdentityOperator(feature_shape=(2, 4, 4), dtype=torch.float32, device=torch.device("cpu"))
        x = torch.randn(2, 4, 4)
        assert torch.equal(op.apply(x), x)

    def test_concretize_returns_region_endpoints(self) -> None:
        op = IdentityOperator(feature_shape=(2, 3), dtype=torch.float32, device=torch.device("cpu"))
        region = _make_region((2, 3))
        assert torch.equal(op.concretize_min(region), region.lower)
        assert torch.equal(op.concretize_max(region), region.upper)

    def test_to_dense_matches_eye(self) -> None:
        op = IdentityOperator(feature_shape=(2, 3), dtype=torch.float32, device=torch.device("cpu"))
        dense = op.to_dense()
        expected = torch.eye(6).reshape(2, 3, 2, 3)
        assert torch.allclose(dense.tensor, expected)

    def test_to_dense_with_batch_shape(self) -> None:
        op = IdentityOperator(feature_shape=(3,), dtype=torch.float32, device=torch.device("cpu"), batch_shape=(1,))
        dense = op.to_dense()
        assert dense.output_shape == torch.Size((1, 3))
        assert dense.input_shape == torch.Size((3,))
        assert torch.allclose(dense.tensor, torch.eye(3).reshape(1, 3, 3))

    def test_clone_is_independent_type(self) -> None:
        op = IdentityOperator(feature_shape=(2, 3), dtype=torch.float32, device=torch.device("cpu"))
        cloned = op.clone()
        assert isinstance(cloned, IdentityOperator)
        assert cloned.feature_shape == op.feature_shape


class TestClampDecomposition:
    """``clamp_min`` / ``clamp_max`` are load-bearing for backward sign decomposition."""

    def test_dense_clamp_min_matches_tensor(self) -> None:
        torch.manual_seed(0)
        tensor = torch.randn(3, 4, 5)
        op = DenseOperator(tensor, output_shape=(3, 4))
        clamped = op.clamp_min(0.0)
        assert isinstance(clamped, DenseOperator)
        assert torch.equal(clamped.tensor, tensor.clamp(min=0.0))
        assert clamped.output_shape == op.output_shape

    def test_dense_clamp_max_matches_tensor(self) -> None:
        torch.manual_seed(0)
        tensor = torch.randn(3, 4, 5)
        op = DenseOperator(tensor, output_shape=(3, 4))
        clamped = op.clamp_max(0.0)
        assert isinstance(clamped, DenseOperator)
        assert torch.equal(clamped.tensor, tensor.clamp(max=0.0))

    def test_dense_clamp_min_nonzero_value(self) -> None:
        torch.manual_seed(1)
        tensor = torch.randn(2, 3)
        op = DenseOperator(tensor, output_shape=(2,))
        clamped = op.clamp_min(0.5)
        assert torch.equal(clamped.tensor, tensor.clamp(min=0.5))

    def test_identity_clamp_min_zero_returns_self(self) -> None:
        op = IdentityOperator(feature_shape=(3,), dtype=torch.float32, device=torch.device("cpu"))
        result = op.clamp_min(0.0)
        assert result is op

    def test_identity_clamp_min_negative_returns_self(self) -> None:
        # Identity entries are 0 or 1; any non-positive lower clamp leaves them unchanged.
        op = IdentityOperator(feature_shape=(2, 3), dtype=torch.float32, device=torch.device("cpu"))
        result = op.clamp_min(-1.5)
        assert result is op

    def test_identity_clamp_max_above_one_returns_self(self) -> None:
        op = IdentityOperator(feature_shape=(3,), dtype=torch.float32, device=torch.device("cpu"))
        result = op.clamp_max(2.0)
        assert result is op

    def test_identity_clamp_max_zero_is_all_zeros(self) -> None:
        op = IdentityOperator(feature_shape=(3,), dtype=torch.float32, device=torch.device("cpu"))
        result = op.clamp_max(0.0)
        dense = result.to_dense()
        assert dense.output_shape == op.output_shape
        assert dense.input_shape == op.input_shape
        assert torch.equal(dense.tensor, torch.zeros_like(dense.tensor))

    def test_identity_clamp_min_intermediate_falls_back_to_dense(self) -> None:
        op = IdentityOperator(feature_shape=(3,), dtype=torch.float32, device=torch.device("cpu"))
        # 0 < 0.5 < 1 — neither fast path applies; should yield a dense matrix
        # whose zeros became 0.5 and whose ones stayed.
        result = op.clamp_min(0.5)
        dense = result.to_dense()
        expected = torch.eye(3).clamp(min=0.5)
        assert torch.allclose(dense.tensor, expected)

    def test_reshape_clamp_delegates_and_preserves_shape(self) -> None:
        torch.manual_seed(2)
        tensor = torch.randn(6, 5)
        inner = DenseOperator(tensor, output_shape=(6,))
        op = ReshapeOperator(inner, output_shape=(2, 3))

        clamped_min = op.clamp_min(0.0)
        assert isinstance(clamped_min, ReshapeOperator)
        assert clamped_min.output_shape == torch.Size((2, 3))
        assert torch.equal(
            clamped_min.to_dense().tensor,
            tensor.clamp(min=0.0).reshape(2, 3, 5),
        )

        clamped_max = op.clamp_max(0.0)
        assert isinstance(clamped_max, ReshapeOperator)
        assert torch.equal(
            clamped_max.to_dense().tensor,
            tensor.clamp(max=0.0).reshape(2, 3, 5),
        )

    def test_clamp_default_falls_back_to_dense(self) -> None:
        # Verify the ABC default produces correct dense output via to_dense.
        # Use IdentityOperator + non-fast-path value to exercise the default.
        op = IdentityOperator(feature_shape=(3,), dtype=torch.float32, device=torch.device("cpu"))
        result = op.clamp_max(0.5)
        dense = result.to_dense()
        expected = torch.eye(3).clamp(max=0.5)
        assert torch.allclose(dense.tensor, expected)


class TestZeroOperator:
    """``ZeroOperator`` is the additive identity and absorbing under ``scale``/``neg``."""

    def _zero(self, output_shape: tuple[int, ...] = (3,), input_shape: tuple[int, ...] = (4,)) -> ZeroOperator:
        return ZeroOperator(
            output_shape=output_shape, input_shape=input_shape, dtype=torch.float32, device=torch.device("cpu")
        )

    def test_metadata(self) -> None:
        op = self._zero(output_shape=(2, 3), input_shape=(4,))
        assert op.output_shape == torch.Size((2, 3))
        assert op.input_shape == torch.Size((4,))
        assert op.dtype == torch.float32

    def test_apply_returns_zeros(self) -> None:
        op = self._zero(output_shape=(3,), input_shape=(4,))
        x = torch.randn(4)
        result = op.apply(x)
        assert torch.equal(result, torch.zeros(3))

    def test_apply_transpose_returns_zeros_with_leading(self) -> None:
        op = self._zero(output_shape=(3,), input_shape=(4,))
        y = torch.randn(7, 3)
        result = op.apply_transpose(y)
        assert result.shape == torch.Size((7, 4))
        assert torch.all(result == 0)

    def test_concretize_is_zero(self) -> None:
        op = self._zero(output_shape=(3,), input_shape=(4,))
        region = _make_region((4,))
        assert torch.equal(op.concretize_min(region), torch.zeros(3))
        assert torch.equal(op.concretize_max(region), torch.zeros(3))

    def test_neg_is_self(self) -> None:
        op = self._zero()
        assert op.neg() is op

    def test_scale_is_self(self) -> None:
        op = self._zero()
        assert op.scale(torch.tensor(2.5)) is op

    def test_clamp_min_zero_is_self(self) -> None:
        op = self._zero()
        assert op.clamp_min(0.0) is op
        assert op.clamp_min(-3.0) is op

    def test_clamp_max_zero_is_self(self) -> None:
        op = self._zero()
        assert op.clamp_max(0.0) is op
        assert op.clamp_max(2.5) is op

    def test_clamp_min_positive_falls_back_to_dense(self) -> None:
        op = self._zero(output_shape=(3,), input_shape=(2,))
        result = op.clamp_min(1.0)
        dense = result.to_dense()
        # Every entry of the zero map clamped up to 1.0 is 1.0.
        assert torch.allclose(dense.tensor, torch.ones(3, 2))

    def test_add_zero_to_dense_returns_dense_unchanged(self) -> None:
        torch.manual_seed(0)
        tensor = torch.randn(3, 4)
        dense = DenseOperator(tensor, output_shape=(3,))
        zero = self._zero(output_shape=(3,), input_shape=(4,))
        result = dense.add(zero)
        # Returns the same dense object — no materialization, no allocation.
        assert result is dense

    def test_add_zero_to_identity_returns_identity_unchanged(self) -> None:
        # IdentityOperator uses the ABC default ``add``; verify the zero
        # short-circuit works through the default path without materializing eye.
        identity = IdentityOperator(feature_shape=(3,), dtype=torch.float32, device=torch.device("cpu"))
        zero = ZeroOperator(
            output_shape=identity.output_shape,
            input_shape=identity.input_shape,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        result = identity.add(zero)
        assert result is identity

    def test_add_zero_to_reshape_returns_reshape_unchanged(self) -> None:
        torch.manual_seed(2)
        tensor = torch.randn(6, 5)
        inner = DenseOperator(tensor, output_shape=(6,))
        reshape = ReshapeOperator(inner, output_shape=(2, 3))
        zero = self._zero(output_shape=(2, 3), input_shape=(5,))
        result = reshape.add(zero)
        assert result is reshape

    def test_zero_add_other_returns_other(self) -> None:
        torch.manual_seed(3)
        tensor = torch.randn(3, 4)
        dense = DenseOperator(tensor, output_shape=(3,))
        zero = self._zero(output_shape=(3,), input_shape=(4,))
        result = zero.add(dense)
        assert result is dense

    def test_zero_add_zero_stays_zero(self) -> None:
        a = self._zero(output_shape=(3,), input_shape=(4,))
        b = self._zero(output_shape=(3,), input_shape=(4,))
        result = a.add(b)
        assert isinstance(result, ZeroOperator)
        assert result.output_shape == torch.Size((3,))
        assert result.input_shape == torch.Size((4,))

    def test_to_dense_matches_zeros(self) -> None:
        op = self._zero(output_shape=(2, 3), input_shape=(4,))
        dense = op.to_dense()
        assert dense.output_shape == torch.Size((2, 3))
        assert dense.input_shape == torch.Size((4,))
        assert torch.equal(dense.tensor, torch.zeros(2, 3, 4))

    def test_shape_ops_stay_zero(self) -> None:
        op = self._zero(output_shape=(2, 3), input_shape=(4,))
        flat = op.flatten_output(0, 1)
        assert isinstance(flat, ZeroOperator)
        assert flat.output_shape == torch.Size((6,))

        unsq = op.unsqueeze_output(0)
        assert isinstance(unsq, ZeroOperator)
        assert unsq.output_shape == torch.Size((1, 2, 3))

        summed = op.sum_output(dim=0, keepdim=False)
        assert isinstance(summed, ZeroOperator)
        assert summed.output_shape == torch.Size((3,))

    def test_identity_clamp_max_zero_returns_zero_operator(self) -> None:
        identity = IdentityOperator(feature_shape=(3,), dtype=torch.float32, device=torch.device("cpu"))
        result = identity.clamp_max(0.0)
        assert isinstance(result, ZeroOperator)
        assert result.output_shape == identity.output_shape
        assert result.input_shape == identity.input_shape


class TestRoundTrip:
    def test_concretize_via_linear_bounds(self) -> None:
        """Concretize through LinearBounds.concretize and confirm ends match DenseOperator."""
        from bound_propagation.bounds import LinearBounds

        torch.manual_seed(99)
        linear = torch.randn(3, 4, 5)
        output_shape = torch.Size((3, 4))
        region = _make_region((5,))
        bias = torch.zeros(output_shape)

        bounds = LinearBounds(
            bias_lower=bias,
            bias_upper=bias,
            linear_lower=linear,
            linear_upper=linear,
            regions=region,
            input_ids=0,
        )
        concrete = bounds.concretize()

        op = DenseOperator(linear, output_shape=output_shape)
        got_min = op.concretize_min(region)
        got_max = op.concretize_max(region)

        assert torch.allclose(concrete.lower, got_min)
        assert torch.allclose(concrete.upper, got_max)
