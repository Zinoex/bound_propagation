"""
Utility functions for forward LBP strategies.

Helper functions for working with linear bounds in LBP-style propagation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import torch

from ...bounds import LinearBounds
from ...regions import SimpleRegion


def transform_linear_terms(
    linear_terms: list[torch.Tensor],
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:
    """Apply a tensor transform to each affine coefficient tensor."""
    return [transform(linear_term) for linear_term in linear_terms]


def combine_linear_terms(
    components: list[tuple[LinearBounds, Literal["lower", "upper"], float]],
) -> tuple[list, list[torch.Tensor], list[int]]:
    """Combine affine terms from multiple bounds, aligned by input IDs."""
    return LinearBounds.combine_linear_terms(components)


def create_identity_bounds(id: int, region: SimpleRegion, shape: tuple[int, ...]) -> LinearBounds:
    """
    Create identity linear bounds (output = input).

    Used for input nodes in forward-mode LBP.

    Args:
        region: Input region
        shape: Shape of the output

    Returns:
        LinearBounds with identity mapping
    """

    # Identity for (*batch_dims, *output_dims, *input_dims) where output_dims = input_dims = shape.
    # Region has shape (*batch_dims, *shape), so batch_ndim = len(region.shape) - len(shape).
    # identity[:, I, I] = 1 for any multi-dimensional index I, rest is zero.
    numel = math.prod(shape)
    n_batch = len(region.shape) - len(shape)
    eye = torch.eye(numel, dtype=region.dtype, device=region.device)
    identity = eye.reshape((1,) * n_batch + tuple(shape) + tuple(shape)).expand(*region.shape, *shape)

    bias = torch.zeros(region.shape, dtype=region.dtype, device=region.device)

    return LinearBounds(
        regions=[region],
        input_ids=[id],
        linear_lower=[identity],
        bias_lower=bias,
        linear_upper=[identity],
        bias_upper=bias,
    )
