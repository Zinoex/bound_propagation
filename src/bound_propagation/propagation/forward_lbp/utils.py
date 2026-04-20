"""
Utility functions for forward LBP strategies.

Helper functions for working with linear bounds in LBP-style propagation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

from ...bounds import LinearBounds
from ...linear_operators import IdentityOperator, LinearOperator
from ...regions import SimpleRegion


def transform_linear_terms(
    linear_terms: list[torch.Tensor],
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:
    """Apply a tensor transform to each affine coefficient tensor."""
    return [transform(linear_term) for linear_term in linear_terms]


def combine_linear_terms(
    components: list[tuple[LinearBounds, Literal["lower", "upper"], float]],
) -> tuple[list[SimpleRegion], list[LinearOperator], list[int]]:
    """Combine affine terms from multiple bounds, aligned by input IDs.

    Returns operator-backed coefficient terms; callers may pass the list
    directly back to :class:`LinearBounds` which accepts either tensors or
    operators.
    """
    return LinearBounds.combine_linear_terms(components)


def create_identity_bounds(id: int, region: SimpleRegion, shape: tuple[int, ...]) -> LinearBounds:
    """
    Create identity linear bounds (output = input).

    Used for input nodes in forward-mode LBP. The coefficient operators are
    :class:`IdentityOperator` instances so downstream strategies can
    type-dispatch on "this is the raw input identity" to avoid unnecessary
    materialization (e.g. :class:`ForwardLBPConv2d` emits a
    :class:`Conv2dOperator` when both input operators are identity).

    Args:
        region: Input region
        shape: Shape of the output

    Returns:
        LinearBounds with identity mapping
    """
    n_batch = len(region.shape) - len(shape)
    batch_ones = (1,) * n_batch
    bias = torch.zeros(batch_ones + tuple(shape), dtype=region.dtype, device=region.device)

    identity = IdentityOperator(
        feature_shape=tuple(shape),
        dtype=region.dtype,
        device=region.device,
        batch_shape=batch_ones,
    )

    return LinearBounds(
        regions=[region],
        input_ids=[id],
        linear_lower=[identity],
        bias_lower=bias,
        linear_upper=[identity],
        bias_upper=bias,
    )
