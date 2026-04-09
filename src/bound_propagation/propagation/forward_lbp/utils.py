"""
Utility functions for forward LBP strategies.

Helper functions for working with linear bounds in LBP-style propagation.
"""

from __future__ import annotations

import torch

from ...bounds import LinearBounds


def create_identity_bounds(region, shape: tuple[int, ...]) -> LinearBounds:
    """
    Create identity linear bounds (output = input).

    Used for input nodes in forward-mode LBP.

    Args:
        region: Input region
        shape: Shape of the output

    Returns:
        LinearBounds with identity mapping
    """
    # Flatten input dimension
    input_size = region.lower.numel()
    output_size = torch.Size(shape).numel()

    # Create identity matrix if input and output sizes match
    if input_size == output_size:
        identity = torch.eye(output_size, dtype=region.dtype, device=region.device)
        bias = torch.zeros(output_size, dtype=region.dtype, device=region.device)

        return LinearBounds(
            region=region,
            linear_lower=identity,
            bias_lower=bias,
            linear_upper=identity,
            bias_upper=bias,
        )
    else:
        # If sizes don't match, we can't create identity mapping
        # Fall back to IntervalBounds converted to LinearBounds

        lower, upper = region.lower, region.upper
        return LinearBounds(
            region=region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )


def create_element_wise_relaxation_bounds(
    region,
    alpha_lower: torch.Tensor,
    beta_lower: torch.Tensor,
    alpha_upper: torch.Tensor,
    beta_upper: torch.Tensor,
) -> LinearBounds:
    """
    Create LinearBounds representing an element-wise linear relaxation.

    For element-wise nonlinear functions, the relaxation is:
    y_i = alpha_i * x_i + beta_i

    This creates bounds with diagonal weight matrices where the diagonal
    contains the alpha values.

    Args:
        region: Input region (shape: *, elem_shape)
        alpha_lower: Slope for lower bound relaxation (shape: *, elem_shape)
        beta_lower: Bias for lower bound relaxation (shape: *, elem_shape)
        alpha_upper: Slope for upper bound relaxation (shape: *, elem_shape)
        beta_upper: Bias for upper bound relaxation (shape: *, elem_shape)

    Returns:
        LinearBounds with diagonal structure representing the relaxation
    """
    # Create diagonal matrices
    linear_lower = torch.diag(alpha_lower.flatten())
    linear_upper = torch.diag(alpha_upper.flatten())

    return LinearBounds(
        region=region,
        linear_lower=linear_lower,
        bias_lower=beta_lower,
        linear_upper=linear_upper,
        bias_upper=beta_upper,
    )


def apply_linear_relaxation(
    bounds: LinearBounds,
    alpha_lower: torch.Tensor,
    beta_lower: torch.Tensor,
    alpha_upper: torch.Tensor,
    beta_upper: torch.Tensor,
) -> LinearBounds:
    """
    Apply element-wise linear relaxation to bounds using forward composition.

    For unary nonlinear functions according to Table 6 in the LBP paper:
    - W_out = diag(alpha) @ W_in
    - b_out = diag(alpha) @ b_in + beta

    Args:
        bounds: Input LinearBounds
        alpha_lower: Slope for lower bound relaxation (shape: output_shape)
        beta_lower: Bias for lower bound relaxation (shape: output_shape)
        alpha_upper: Slope for upper bound relaxation (shape: output_shape)
        beta_upper: Bias for upper bound relaxation (shape: output_shape)

    Returns:
        LinearBounds with relaxation applied
    """
    # Create the relaxation as LinearBounds with diagonal structure
    relaxation = create_element_wise_relaxation_bounds(
        bounds.region,
        alpha_lower,
        beta_lower,
        alpha_upper,
        beta_upper,
    )

    # Use forward composition to apply the relaxation
    return bounds.forward_compose(relaxation)
