"""
Utility functions for backward LBP strategies.

Helper functions for working with linear bounds in backward LBP-style propagation.
"""

from __future__ import annotations

import torch

from ...bounds import LinearBounds


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
        region: Input region
        alpha_lower: Slope for lower bound relaxation (shape: output_shape)
        beta_lower: Bias for lower bound relaxation (shape: output_shape)
        alpha_upper: Slope for upper bound relaxation (shape: output_shape)
        beta_upper: Bias for upper bound relaxation (shape: output_shape)

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


def apply_linear_relaxation_backward(
    bounds: LinearBounds,
    alpha_lower: torch.Tensor,
    beta_lower: torch.Tensor,
    alpha_upper: torch.Tensor,
    beta_upper: torch.Tensor,
) -> LinearBounds:
    """
    Apply element-wise linear relaxation to bounds using backward composition.

    For unary nonlinear functions in backward mode, we compute the inverse relaxation.
    If y = f(x) and we have bounds on y, we compute bounds on x.

    Args:
        bounds: Output LinearBounds (bounds on y)
        alpha_lower: Slope for lower bound relaxation (shape: input_shape)
        beta_lower: Bias for lower bound relaxation (shape: input_shape)
        alpha_upper: Slope for upper bound relaxation (shape: input_shape)
        beta_upper: Bias for upper bound relaxation (shape: input_shape)

    Returns:
        LinearBounds with relaxation applied via backward composition
    """
    # Create the relaxation as LinearBounds with diagonal structure
    relaxation = create_element_wise_relaxation_bounds(
        bounds.region,
        alpha_lower,
        beta_lower,
        alpha_upper,
        beta_upper,
    )

    # Use backward composition to apply the relaxation
    # bounds.backward_compose(relaxation) computes bounds o relaxation
    return bounds.backward_compose(relaxation)
