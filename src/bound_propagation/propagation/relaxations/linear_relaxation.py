"""
LinearRelaxation: Data structure for storing linear approximations of operations.

A LinearRelaxation represents a linear approximation of an operation output
with respect to its inputs: z_lower ≥ Σ(W_i @ x_i) + b_lower
                            z_upper ≤ Σ(W_i @ x_i) + b_upper

For multi-input operations, coefficients are stored as a list (one per input).
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LinearRelaxation:
    """
    Represents a linear relaxation of an operation.

    For an operation z = f(x_1, x_2, ..., x_n), the relaxation is:
        z_lower ≥ W_1^L @ x_1 + W_2^L @ x_2 + ... + W_n^L @ x_n + b^L
        z_upper ≤ W_1^U @ x_1 + W_2^U @ x_2 + ... + W_n^U @ x_n + b^U

    Attributes:
        coeffs_lower: List of tensors, one per input. Each tensor represents
                      the linear coefficient matrix for that input's contribution
                      to the lower bound.
        coeffs_upper: List of tensors, one per input. Each tensor represents
                      the linear coefficient matrix for that input's contribution
                      to the upper bound.
        bias_lower: Bias vector for the lower bound.
        bias_upper: Bias vector for the upper bound.
        input_shapes: Optional list of input tensor shapes for validation.
        output_shape: Optional output tensor shape for validation.
    """

    coeffs_lower: list[torch.Tensor]
    coeffs_upper: list[torch.Tensor]
    bias_lower: torch.Tensor
    bias_upper: torch.Tensor
    input_shapes: list[torch.Size] | None = None
    output_shape: torch.Size | None = None

    def __post_init__(self):
        """Validate the relaxation structure."""
        if len(self.coeffs_lower) != len(self.coeffs_upper):
            raise ValueError(
                f"Number of lower coefficients ({len(self.coeffs_lower)}) must match upper coefficients ({len(self.coeffs_upper)})"
            )

        if len(self.coeffs_lower) == 0:
            raise ValueError("At least one input coefficient is required")

        # Check that bias shapes match
        if self.bias_lower.shape != self.bias_upper.shape:
            raise ValueError(f"Bias shapes don't match: lower={self.bias_lower.shape}, upper={self.bias_upper.shape}")

        # Validate input shapes if provided
        if self.input_shapes is not None:
            if len(self.input_shapes) != len(self.coeffs_lower):
                raise ValueError(
                    f"Number of input shapes ({len(self.input_shapes)}) must match number of coefficients ({len(self.coeffs_lower)})"
                )

    @property
    def num_inputs(self) -> int:
        """Return the number of inputs this relaxation covers."""
        return len(self.coeffs_lower)

    def get_input_coeff(self, input_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the coefficient tensors for a specific input.

        Args:
            input_idx: Index of the input (0-based).

        Returns:
            Tuple of (lower_coeff, upper_coeff) for the specified input.
        """
        if input_idx < 0 or input_idx >= self.num_inputs:
            raise IndexError(f"Input index {input_idx} out of range [0, {self.num_inputs})")
        return self.coeffs_lower[input_idx], self.coeffs_upper[input_idx]

    def to(self, device: torch.device) -> "LinearRelaxation":
        """
        Move all tensors in the relaxation to the specified device.

        Args:
            device: Target device.

        Returns:
            New LinearRelaxation with tensors on the target device.
        """
        return LinearRelaxation(
            coeffs_lower=[c.to(device) for c in self.coeffs_lower],
            coeffs_upper=[c.to(device) for c in self.coeffs_upper],
            bias_lower=self.bias_lower.to(device),
            bias_upper=self.bias_upper.to(device),
            input_shapes=self.input_shapes,
            output_shape=self.output_shape,
        )

    def is_exact(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """
        Check if the relaxation is exact (lower == upper).

        This is true for linear operations where the relaxation
        is exact rather than approximate.

        Args:
            rtol: Relative tolerance for comparison.
            atol: Absolute tolerance for comparison.

        Returns:
            True if lower and upper bounds are equal (within tolerance).
        """
        coeffs_match = all(
            torch.allclose(c_l, c_u, rtol=rtol, atol=atol) for c_l, c_u in zip(self.coeffs_lower, self.coeffs_upper)
        )
        bias_match = torch.allclose(self.bias_lower, self.bias_upper, rtol=rtol, atol=atol)
        return coeffs_match and bias_match

    @staticmethod
    def create_identity(
        input_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "LinearRelaxation":
        """
        Create an identity relaxation (output = input).

        This is useful for operations that are pass-through or
        for initializing relaxations.

        Args:
            input_shape: Shape of the input/output tensor.
            device: Device for the tensors.
            dtype: Data type for the tensors.

        Returns:
            LinearRelaxation representing the identity function.
        """
        # For identity, coefficient is 1.0 (scalar or broadcast)
        # We use a scalar coefficient that will be broadcast during composition
        coeff = torch.ones(1, device=device, dtype=dtype)
        bias = torch.zeros(input_shape, device=device, dtype=dtype)

        return LinearRelaxation(
            coeffs_lower=[coeff],
            coeffs_upper=[coeff],
            bias_lower=bias,
            bias_upper=bias,
            input_shapes=[input_shape],
            output_shape=input_shape,
        )

    @staticmethod
    def create_diagonal(
        alpha_lower: torch.Tensor,
        alpha_upper: torch.Tensor,
        beta_lower: torch.Tensor,
        beta_upper: torch.Tensor,
    ) -> "LinearRelaxation":
        """
        Create a diagonal (element-wise) relaxation from alpha/beta parameters.

        For element-wise operations, the relaxation has the form:
            y_i_lower = alpha_lower_i * x_i + beta_lower_i
            y_i_upper = alpha_upper_i * x_i + beta_upper_i

        Args:
            alpha_lower: Element-wise slopes for lower bound.
            alpha_upper: Element-wise slopes for upper bound.
            beta_lower: Element-wise biases for lower bound.
            beta_upper: Element-wise biases for upper bound.

        Returns:
            LinearRelaxation with diagonal coefficients.
        """
        # For diagonal operations, we store alpha directly as the coefficient
        # This will be broadcast/multiplied element-wise during composition
        return LinearRelaxation(
            coeffs_lower=[alpha_lower],
            coeffs_upper=[alpha_upper],
            bias_lower=beta_lower,
            bias_upper=beta_upper,
            input_shapes=[alpha_lower.shape],
            output_shape=beta_lower.shape,
        )
