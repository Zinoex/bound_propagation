"""
Utility functions for forward CROWN strategies.

Helper functions for working with linear bounds in CROWN-style propagation.
"""

from __future__ import annotations

import torch

from ...bounds import IntervalBounds, LinearBounds


def verify_linear_bounds(input_bounds: list) -> None:
    """
    Verify that input bounds are LinearBounds.

    Args:
        input_bounds: List of bounds to verify

    Raises:
        TypeError: If any bounds are not LinearBounds
    """
    for i, bounds in enumerate(input_bounds):
        if not isinstance(bounds, LinearBounds):
            raise TypeError(
                f"CROWN strategies require LinearBounds, "
                f"but input {i} has type {type(bounds).__name__}"
            )


def linearize_bounds(bounds: IntervalBounds | LinearBounds) -> LinearBounds:
    """
    Convert bounds to LinearBounds if needed.

    If bounds are already LinearBounds, returns as-is.
    If bounds are IntervalBounds, creates LinearBounds with zero coefficients (constants).

    Args:
        bounds: Bounds to linearize

    Returns:
        LinearBounds representation
    """
    if isinstance(bounds, LinearBounds):
        return bounds

    if isinstance(bounds, IntervalBounds):
        # Convert IntervalBounds to LinearBounds with zero coefficients
        # This represents constant bounds: lower = 0 @ x + lower, upper = 0 @ x + upper
        return LinearBounds(
            region=bounds.region,
            linear_lower=None,  # No linear dependency
            bias_lower=bounds.lower,
            linear_upper=None,  # No linear dependency
            bias_upper=bounds.upper,
        )

    raise TypeError(f"Cannot linearize bounds of type {type(bounds).__name__}")


def create_identity_bounds(region, shape: tuple[int, ...]) -> LinearBounds:
    """
    Create identity linear bounds (output = input).

    Used for input nodes in forward-mode CROWN.

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
