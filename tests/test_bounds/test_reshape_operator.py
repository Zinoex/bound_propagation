"""Unit tests for the lazy :class:`ReshapeOperator` wrapper."""

from __future__ import annotations

import pytest
import torch

from bound_propagation.linear_operators import (
    DenseOperator,
    IdentityOperator,
    ReshapeOperator,
)
from bound_propagation.regions import HyperRectangle


def _make_region(shape: tuple[int, ...]) -> HyperRectangle:
    torch.manual_seed(0)
    lower = torch.randn(*shape)
    upper = lower + torch.rand(*shape) + 0.1
    return HyperRectangle(lower=lower, upper=upper)


class TestReshapeOperatorConstruction:
    def test_reshape_preserves_numel(self) -> None:
        inner = DenseOperator(torch.randn(2, 3, 4, 5), output_shape=(2, 3, 4))
        op = ReshapeOperator(inner, (6, 4))
        assert op.output_shape == torch.Size((6, 4))
        assert op.input_shape == torch.Size((5,))

    def test_numel_mismatch_raises(self) -> None:
        inner = DenseOperator(torch.randn(2, 3, 4), output_shape=(2, 3))
        with pytest.raises(ValueError, match="total size must match"):
            ReshapeOperator(inner, (5, 2))

    def test_nested_wrapper_is_flattened(self) -> None:
        base = DenseOperator(torch.randn(2, 3, 5), output_shape=(2, 3))
        first = ReshapeOperator(base, (6,))
        second = ReshapeOperator(first, (3, 2))
        assert second.inner is base
        assert second.output_shape == torch.Size((3, 2))


class TestReshapeOperatorApply:
    def test_apply_matches_dense_reshape(self) -> None:
        torch.manual_seed(0)
        tensor = torch.randn(2, 3, 4)
        inner = DenseOperator(tensor, output_shape=(2, 3))
        op = ReshapeOperator(inner, (6,))

        x = torch.randn(4)
        expected = inner.apply(x).reshape(6)
        assert torch.allclose(op.apply(x), expected)

    def test_apply_transpose_roundtrips(self) -> None:
        torch.manual_seed(0)
        tensor = torch.randn(2, 3, 4)
        inner = DenseOperator(tensor, output_shape=(2, 3))
        op = ReshapeOperator(inner, (6,))

        y = torch.randn(7, 6)  # (leading, *new_output_shape)
        expected = inner.apply_transpose(y.reshape(7, 2, 3))
        assert torch.allclose(op.apply_transpose(y), expected)

    def test_apply_transpose_trailing_shape_mismatch(self) -> None:
        inner = DenseOperator(torch.randn(2, 3, 4), output_shape=(2, 3))
        op = ReshapeOperator(inner, (6,))
        with pytest.raises(ValueError, match="does not match output_shape"):
            op.apply_transpose(torch.randn(3, 5))

    def test_concretize_min_and_max_match_dense(self) -> None:
        torch.manual_seed(1)
        tensor = torch.randn(2, 3, 4)
        inner = DenseOperator(tensor, output_shape=(2, 3))
        op = ReshapeOperator(inner, (6,))

        region = _make_region((4,))
        assert torch.allclose(op.concretize_min(region), inner.concretize_min(region).reshape(6))
        assert torch.allclose(op.concretize_max(region), inner.concretize_max(region).reshape(6))


class TestReshapeOperatorAlgebra:
    def test_neg_preserves_wrapping(self) -> None:
        inner = DenseOperator(torch.randn(2, 3, 4), output_shape=(2, 3))
        op = ReshapeOperator(inner, (6,))
        negged = op.neg()
        assert isinstance(negged, ReshapeOperator)
        assert torch.allclose(negged.to_dense().tensor, -op.to_dense().tensor)

    def test_scalar_scale_preserves_wrapping(self) -> None:
        inner = DenseOperator(torch.randn(2, 3, 4), output_shape=(2, 3))
        op = ReshapeOperator(inner, (6,))
        scaled = op.scale(torch.tensor(2.5))
        assert isinstance(scaled, ReshapeOperator)
        assert torch.allclose(scaled.to_dense().tensor, op.to_dense().tensor * 2.5)

    def test_matching_shape_scale_preserves_wrapping(self) -> None:
        inner = DenseOperator(torch.randn(2, 3, 4), output_shape=(2, 3))
        op = ReshapeOperator(inner, (6,))
        factor = torch.arange(6, dtype=torch.float32)
        scaled = op.scale(factor)
        assert isinstance(scaled, ReshapeOperator)
        expected = op.to_dense().tensor * factor.reshape(6, 1)
        assert torch.allclose(scaled.to_dense().tensor, expected)

    def test_add_with_matching_reshape_combines_inners(self) -> None:
        a = DenseOperator(torch.randn(2, 3, 4), output_shape=(2, 3))
        b = DenseOperator(torch.randn(2, 3, 4), output_shape=(2, 3))
        op_a = ReshapeOperator(a, (6,))
        op_b = ReshapeOperator(b, (6,))
        combined = op_a.add(op_b)
        assert isinstance(combined, ReshapeOperator)
        assert torch.allclose(combined.to_dense().tensor, op_a.to_dense().tensor + op_b.to_dense().tensor)


class TestReshapeOperatorOutputShapeOps:
    def test_further_reshape_composes(self) -> None:
        inner = DenseOperator(torch.randn(2, 3, 4), output_shape=(2, 3))
        op = ReshapeOperator(inner, (6,))
        op2 = op.reshape_output((3, 2))
        assert isinstance(op2, ReshapeOperator)
        assert op2.inner is inner
        assert op2.output_shape == torch.Size((3, 2))

    def test_flatten_composes(self) -> None:
        inner = DenseOperator(torch.randn(2, 3, 4, 5), output_shape=(2, 3, 4))
        op = ReshapeOperator(inner, (2, 3, 4))  # identity reshape
        flattened = op.flatten_output(0, -1)
        assert isinstance(flattened, ReshapeOperator)
        assert flattened.output_shape == torch.Size((24,))

    def test_unsqueeze_composes(self) -> None:
        inner = DenseOperator(torch.randn(6, 4), output_shape=(6,))
        op = ReshapeOperator(inner, (6,))
        unsq = op.unsqueeze_output(0)
        assert isinstance(unsq, ReshapeOperator)
        assert unsq.output_shape == torch.Size((1, 6))

    def test_squeeze_composes(self) -> None:
        inner = DenseOperator(torch.randn(1, 6, 4), output_shape=(1, 6))
        op = ReshapeOperator(inner, (1, 6))
        sq = op.squeeze_output(0)
        assert isinstance(sq, ReshapeOperator)
        assert sq.output_shape == torch.Size((6,))


class TestReshapeOperatorPreservesInnerIdentity:
    def test_wrapping_identity_does_not_materialize(self) -> None:
        """ReshapeOperator should not eagerly materialize a structured inner operator."""
        identity = IdentityOperator(feature_shape=(3, 4), dtype=torch.float32, device=torch.device("cpu"))
        op = ReshapeOperator(identity, (12,))
        assert op.inner is identity
        assert op.output_shape == torch.Size((12,))
        assert op.input_shape == torch.Size((3, 4))
