"""
Backward propagation strategy for RELU operation.

For z = relu(x), backward propagation uses linear relaxations based on concrete bounds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds, LinearBounds
from ..backward_strategy import BackwardBoundingStrategy

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from..config import StrategyConfig


class BackwardReluStrategy(BackwardBoundingStrategy):
    """
    Backward propagation strategy for RELU operation.

    For z = relu(x):
    - Use concrete bounds on x to determine regime:
      - x_upper <= 0: z = 0 (A_x = 0)
      - x_lower >= 0: z = x (A_x = A_z)
      - Otherwise: linear relaxation
    
    The relaxation depends on the concrete bounds from the forward pass.
    """

    @property
    def method_name(self) -> str:
        """Return method name."""
        return "backward"

    def propagate_backward(
        self,
        node: Node,
        input_idx: int,
        output_bounds: LinearBounds,
        concrete_input_bounds: list[AbstractBounds],
        config: StrategyConfig,
    ) -> LinearBounds:
        """
        Propagate bounds backward through RELU.

        Uses concrete bounds to compute linear relaxation, then propagates
        output_bounds through the relaxation.

        Args:
            node: The RELU node
            input_idx: Index of input (must be 0)
            output_bounds: Linear bounds on the ReLU output (A_z, Ā_z)
            concrete_input_bounds: Concrete bounds for input (from forward pass)
            config: Strategy configuration

        Returns:
            Contribution to the input
        """
        if node.num_inputs != 1:
            raise ValueError(f"RELU expects 1 input, got {node.num_inputs}")
        if input_idx != 0:
            raise ValueError(f"RELU input_idx must be 0, got {input_idx}")

        # Get concrete bounds for the input
        concrete_bounds = concrete_input_bounds[0]
        lower, upper = concrete_bounds.concretize()

        # Flatten for element-wise processing
        lower_flat = lower.flatten()
        upper_flat = upper.flatten()

        # Compute relaxation slopes for each element
        alpha_lower, beta_lower, alpha_upper, beta_upper = self._compute_relaxation(
            lower_flat, upper_flat
        )

        # Apply relaxation to output bounds
        return self._apply_relaxation(
            output_bounds,
            alpha_lower,
            beta_lower,
            alpha_upper,
            beta_upper,
        )

    def _compute_relaxation(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ReLU relaxation slopes based on concrete bounds.

        For each element:
        - If upper <= 0: always zero (α=0, β=0)
        - If lower >= 0: identity (α=1, β=0)
        - Otherwise: linear relaxation

        Lower relaxation: relu(x) >= α_l * x + β_l
        Upper relaxation: relu(x) <= α_u * x + β_u

        Args:
            lower: Lower bounds (flattened)
            upper: Upper bounds (flattened)

        Returns:
            Tuple of (alpha_lower, beta_lower, alpha_upper, beta_upper)
        """
        # Initialize slopes and biases
        alpha_lower = torch.zeros_like(lower)
        beta_lower = torch.zeros_like(lower)
        alpha_upper = torch.zeros_like(lower)
        beta_upper = torch.zeros_like(lower)

        # Case 1: upper <= 0 -> always zero
        # Already initialized to zero

        # Case 2: lower >= 0 -> identity
        positive_mask = lower >= 0
        alpha_lower[positive_mask] = 1.0
        alpha_upper[positive_mask] = 1.0

        # Case 3: crossing zero -> linear relaxation
        crossing_mask = (lower < 0) & (upper > 0)
        
        # Lower relaxation: y >= 0 (already set)
        
        # Upper relaxation: secant line from (lower, 0) to (upper, upper)
        # slope = (upper - 0) / (upper - lower) = upper / (upper - lower)
        # y <= slope * (x - lower) + 0 = slope * x - slope * lower
        
        if crossing_mask.any():
            slope = upper[crossing_mask] / (upper[crossing_mask] - lower[crossing_mask])
            alpha_upper[crossing_mask] = slope
            beta_upper[crossing_mask] = -slope * lower[crossing_mask]

        return alpha_lower, beta_lower, alpha_upper, beta_upper

    def _apply_relaxation(
        self,
        output_bounds: LinearBounds,
        alpha_lower: torch.Tensor,
        beta_lower: torch.Tensor,
        alpha_upper: torch.Tensor,
        beta_upper: torch.Tensor,
    ) -> LinearBounds:
        """
        Apply linear relaxation to output bounds.

        For z = α * x + β:
        - If A_z represents output dependency on z
        - Then A_x = A_z * α (scale by relaxation slope)
        - And bias contribution is A_z * β

        Args:
            output_bounds: Linear bounds on ReLU output (A_z, Ā_z)
            alpha_lower: Lower bound slopes
            beta_lower: Lower bound biases
            alpha_upper: Upper bound slopes
            beta_upper: Upper bound biases

        Returns:
            Linear bounds for input
        """
        # For lower bound: A_z_lower @ (α_lower * x + β_lower)
        # = (A_z_lower * α_lower) @ x + A_z_lower @ β_lower
        
        # For upper bound: A_z_upper @ (α_upper * x + β_upper)
        # = (A_z_upper * α_upper) @ x + A_z_upper @ β_upper

        # Apply slopes to linear coefficients
        if output_bounds.linear_lower is not None:
            # Scale each column by the corresponding alpha
            linear_lower = output_bounds.linear_lower * alpha_lower.unsqueeze(0)
        else:
            linear_lower = None

        if output_bounds.linear_upper is not None:
            linear_upper = output_bounds.linear_upper * alpha_upper.unsqueeze(0)
        else:
            linear_upper = None

        # Apply to biases
        # bias_contribution = A @ beta + original_bias
        if output_bounds.linear_lower is not None:
            bias_contribution_lower = output_bounds.linear_lower @ beta_lower
        else:
            bias_contribution_lower = torch.zeros_like(output_bounds.bias_lower)
        bias_lower = output_bounds.bias_lower + bias_contribution_lower

        if output_bounds.linear_upper is not None:
            bias_contribution_upper = output_bounds.linear_upper @ beta_upper
        else:
            bias_contribution_upper = torch.zeros_like(output_bounds.bias_upper)
        bias_upper = output_bounds.bias_upper + bias_contribution_upper

        return LinearBounds(
            region=output_bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
