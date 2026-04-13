from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch

    from ...ir import Node


class ForwardLBPTranspose(ForwardLBPStrategy):
    """Forward LBP strategy for TRANSPOSE and PERMUTE operations."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"transpose requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPTranspose requires input to be LinearBounds")

        bounds = input_bounds[0]

        # Get permutation (either dims for transpose or full permutation for permute)
        if "dims" in node.attributes:
            # Permute
            dims = node.attributes["dims"]
            linear_lower = bounds.linear_lower.permute(*dims) if bounds.linear_lower is not None else None
            linear_upper = bounds.linear_upper.permute(*dims) if bounds.linear_upper is not None else None
            bias_lower = bounds.bias_lower.permute(*dims)
            bias_upper = bounds.bias_upper.permute(*dims)
        else:
            # Transpose (swap dim0 and dim1)
            dim0 = node.attributes.get("dim0", 0)
            dim1 = node.attributes.get("dim1", 1)
            linear_lower = bounds.linear_lower.transpose(dim0, dim1) if bounds.linear_lower is not None else None
            linear_upper = bounds.linear_upper.transpose(dim0, dim1) if bounds.linear_upper is not None else None
            bias_lower = bounds.bias_lower.transpose(dim0, dim1)
            bias_upper = bounds.bias_upper.transpose(dim0, dim1)

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
