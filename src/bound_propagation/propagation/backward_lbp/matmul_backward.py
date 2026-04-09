"""
Backward propagation strategy for MATMUL operation.

For z = x @ W where W is a constant matrix, backward propagation applies W^T to the bounds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from ..backward_strategy import BackwardBoundingStrategy

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class BackwardMatmulStrategy(BackwardBoundingStrategy):
    """
    Backward propagation strategy for MATMUL operation.

    For z = x @ W where W is constant:
    - ∂z/∂x = W^T
    - A_x = A_z @ W^T
    
    This is exact for linear operations with constant weights.
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
        Propagate bounds backward through MATMUL.

        For z = x @ W, we have:
        - A_z represents how output depends on z
        - A_x = A_z @ W^T tells us how output depends on x

        Args:
            node: The MATMUL node
            input_idx: Index of input (0 for x, 1 for W)
            output_bounds: Linear bounds on the product (A_z, Ā_z)
            concrete_input_bounds: Concrete bounds (not used for constant weights)
            config: Strategy configuration

        Returns:
            Contribution to the specified input
        """
        if node.num_inputs != 2:
            raise ValueError(f"MATMUL expects 2 inputs, got {node.num_inputs}")

        # Get the weight matrix from node inputs
        weight_node = node.inputs[1]
        if not weight_node.is_constant and not weight_node.is_parameter:
            raise NotImplementedError(
                "Backward MATMUL only supports constant/parameter weights. "
                "Got weights that vary in the input region."
            )

        # Get weight value
        weight = weight_node.attributes.get("value")
        if weight is None:
            raise ValueError(f"Weight node {weight_node.id} missing 'value' attribute")

        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight)

        if input_idx == 0:
            # Propagating to x: A_x = A_z @ W^T
            return self._propagate_to_input(output_bounds, weight)
        elif input_idx == 1:
            # Propagating to W: A_W = x^T @ A_z
            # For constant weights, contribution is zero (W doesn't depend on input region)
            # Return zero bounds
            return self._create_zero_contribution(output_bounds, weight.shape)
        else:
            raise ValueError(f"MATMUL expects input_idx in [0, 1], got {input_idx}")

    def _propagate_to_input(
        self, 
        output_bounds: LinearBounds, 
        weight: torch.Tensor
    ) -> LinearBounds:
        """
        Propagate bounds to the input: A_x = A_z @ W^T.

        Args:
            output_bounds: Linear bounds on output (A_z, Ā_z)
            weight: Weight matrix W

        Returns:
            Linear bounds for input
        """
        # Transpose weight
        weight_t = weight.t()

        # Apply to linear coefficients
        if output_bounds.linear_lower is not None:
            linear_lower = output_bounds.linear_lower @ weight_t
        else:
            linear_lower = None

        if output_bounds.linear_upper is not None:
            linear_upper = output_bounds.linear_upper @ weight_t
        else:
            linear_upper = None

        # Apply to biases  
        bias_lower = output_bounds.bias_lower @ weight_t
        bias_upper = output_bounds.bias_upper @ weight_t

        return LinearBounds(
            region=output_bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )

    def _create_zero_contribution(
        self, 
        output_bounds: LinearBounds, 
        shape: tuple[int, ...]
    ) -> LinearBounds:
        """
        Create zero contribution for constant parameters.

        Args:
            output_bounds: Output bounds (for region and dtype)
            shape: Shape of the weight matrix

        Returns:
            Zero linear bounds
        """
        # For constant weights, there's no contribution to the backward bounds
        # (constants don't depend on the input region)
        numel = int(torch.tensor(shape).prod().item())
        
        bias_lower = torch.zeros(
            numel, 
            dtype=output_bounds.region.dtype,
            device=output_bounds.region.device
        )
        bias_upper = torch.zeros_like(bias_lower)

        return LinearBounds(
            region=output_bounds.region,
            linear_lower=None,  # No linear dependency
            bias_lower=bias_lower,
            linear_upper=None,
            bias_upper=bias_upper,
        )
