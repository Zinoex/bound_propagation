"""Verify Stage 3 contract: reshape-family forward-LBP ops keep the inner
operator unmaterialized by wrapping it in :class:`ReshapeOperator`.

This locks in the property that structured operators (e.g.
:class:`Conv2dPatchOperator`) survive through ``flatten`` / ``view`` /
``reshape`` / ``squeeze`` / ``unsqueeze`` in the forward-LBP pipeline
without being flattened to a dense tensor.
"""

from __future__ import annotations

import torch

from bound_propagation import LinearOperator, ReshapeOperator
from bound_propagation.bounds import LinearBounds
from bound_propagation.linear_operators import IdentityOperator
from bound_propagation.propagation.forward_lbp.shape import (
    ForwardLBPFlatten,
    ForwardLBPReshape,
    ForwardLBPSqueeze,
    ForwardLBPUnsqueeze,
    ForwardLBPView,
)
from bound_propagation.regions import HyperRectangle
from tests.helpers import propagate


def _make_identity_bounds(feature_shape: tuple[int, ...]) -> LinearBounds:
    region = HyperRectangle(
        lower=torch.zeros(*feature_shape),
        upper=torch.ones(*feature_shape),
    )
    identity = IdentityOperator(
        feature_shape=feature_shape,
        dtype=torch.float32,
        device=torch.device("cpu"),
        batch_shape=(),
    )
    bias = torch.zeros(*feature_shape)
    return LinearBounds(
        regions=[region],
        input_ids=[0],
        linear_lower=[identity],
        bias_lower=bias,
        linear_upper=[identity],
        bias_upper=bias,
    )


def _unwrap_inner(op: LinearOperator) -> LinearOperator:
    return op.inner if isinstance(op, ReshapeOperator) else op


class TestFlattenPreservesInnerOperator:
    def test_flatten_wraps_identity_operator(self) -> None:
        bounds = _make_identity_bounds((2, 3, 4))
        result = propagate(ForwardLBPFlatten(), bounds, start_dim=0, end_dim=-1)

        assert result.bias_lower.shape == (24,)
        (lower_op,) = result.linear_lowers_op
        assert isinstance(lower_op, ReshapeOperator)
        assert isinstance(lower_op.inner, IdentityOperator)
        assert lower_op.output_shape == torch.Size((24,))


class TestReshapeAndViewPreserveInnerOperator:
    def test_reshape_wraps_identity_operator(self) -> None:
        bounds = _make_identity_bounds((2, 3, 4))
        result = propagate(ForwardLBPReshape(), bounds, (6, 4))

        (lower_op,) = result.linear_lowers_op
        assert isinstance(lower_op, ReshapeOperator)
        assert isinstance(lower_op.inner, IdentityOperator)

    def test_view_wraps_identity_operator(self) -> None:
        bounds = _make_identity_bounds((2, 3, 4))
        result = propagate(ForwardLBPView(), bounds, (6, 4))

        (lower_op,) = result.linear_lowers_op
        assert isinstance(lower_op, ReshapeOperator)
        assert isinstance(lower_op.inner, IdentityOperator)


class TestSqueezeAndUnsqueezePreserveInnerOperator:
    def test_squeeze_wraps_identity_operator(self) -> None:
        bounds = _make_identity_bounds((1, 3, 4))
        result = propagate(ForwardLBPSqueeze(), bounds, 0)

        assert result.bias_lower.shape == (3, 4)
        (lower_op,) = result.linear_lowers_op
        assert isinstance(lower_op, ReshapeOperator)
        assert isinstance(lower_op.inner, IdentityOperator)

    def test_unsqueeze_wraps_identity_operator(self) -> None:
        bounds = _make_identity_bounds((3, 4))
        result = propagate(ForwardLBPUnsqueeze(), bounds, 0)

        assert result.bias_lower.shape == (1, 3, 4)
        (lower_op,) = result.linear_lowers_op
        assert isinstance(lower_op, ReshapeOperator)
        assert isinstance(lower_op.inner, IdentityOperator)


class TestChainedReshapeDoesNotNest:
    def test_flatten_then_reshape_collapses_wrappers(self) -> None:
        bounds = _make_identity_bounds((2, 3, 4))
        flattened = propagate(ForwardLBPFlatten(), bounds, start_dim=0, end_dim=-1)
        reshaped = propagate(ForwardLBPReshape(), flattened, (4, 6))

        (lower_op,) = reshaped.linear_lowers_op
        assert isinstance(lower_op, ReshapeOperator)
        assert not isinstance(lower_op.inner, ReshapeOperator)
        assert isinstance(lower_op.inner, IdentityOperator)
