"""
AbstractLinearRelaxation: Type hierarchy for linear approximations of operations.

Provides:
  AbstractLinearRelaxation – base class guaranteeing forward and backward_compose.
  ElementwiseLinearRelaxation – for unary element-wise operations: y ≥ alpha*x + beta.
  PairedLinearRelaxation – for binary operations: z ≥ alpha1*x + alpha2*y + beta.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

import torch

from ...bounds import LinearBounds
from ...regions import SimpleRegion


class AbstractLinearRelaxation(ABC):
    """
    Abstract base for linear relaxations of operations.

    Subtypes must implement forward and backward_compose to compose
    the relaxation with linear bounds in the forward and backward directions.
    """

    @abstractmethod
    def forward(self, input_bounds: list[LinearBounds]) -> LinearBounds:
        """
        Compose this relaxation with incoming linear bounds (forward direction).

        Given linear bounds on the inputs (each expressed as an affine function of
        some root variable x0), substitute them into the relaxation to obtain linear
        bounds on the output, also expressed as an affine function of x0.

        Args:
            input_bounds: One LinearBounds per input of the relaxed operation.

        Returns:
            LinearBounds representing the output bounds in terms of x0.
        """

    @abstractmethod
    def symbolic_forward(self, inputs: list[SymbolicLinearRelaxation]) -> SymbolicLinearRelaxation:
        """
        Compose this relaxation with incoming symbolic relaxations (forward direction).

        Similar to forward but operates on symbolic relaxations instead of
        concretized LinearBounds. This allows maintaining a symbolic representation
        of the relaxations during forward propagation.

        Args:
            inputs: One SymbolicLinearRelaxation per input of the relaxed operation.

        Returns:
            A SymbolicLinearRelaxation representing the output relaxation.
        """
        ...


class SymbolicLinearRelaxation(ABC):
    @abstractmethod
    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        """
        Compose backward through this symbolic relaxation.

        Propagates A-matrices backward through the relaxation tree.  Given
        coefficients A that linearly combine this node's outputs, produce
        LinearBounds expressing the same combination in terms of the network
        inputs.

        A tensors have shape ``(*batch, *bounded_out, *node_out)`` where:
        - ``*batch`` are leading batch dimensions (count = ``batch_ndim``),
        - ``*bounded_out`` are the output dimensions being bounded,
        - ``*node_out`` are this node's output feature dimensions.

        Args:
            A_lower: Lower-bound coefficient matrix.
            A_upper: Upper-bound coefficient matrix.
            batch_ndim: Number of leading batch dimensions shared by A
                and the relaxation parameters (alpha / beta / coeffs).

        Returns:
            LinearBounds expressing the bounded quantity in terms of the
            network inputs.
        """
        ...


@final
@dataclass
class OutputLinearRelaxation(SymbolicLinearRelaxation):
    """Wrapper around the final output node's symbolic input(s).

    Not composed via ``backward``; instead, the propagator calls
    ``concretize`` which constructs identity A-matrices and kicks off
    the recursive backward pass.
    """

    inputs: list[SymbolicLinearRelaxation]
    output_shape: tuple[int, ...]

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        raise NotImplementedError("OutputLinearRelaxation does not support backward composition")

    def concretize(self, batch_ndim: int, dtype: torch.dtype, device: torch.device) -> LinearBounds:
        """Build identity A-matrices and backward-propagate to obtain LinearBounds.

        Args:
            batch_ndim: Number of leading batch dimensions in the output shape.
            dtype: Tensor dtype for the identity matrices.
            device: Tensor device for the identity matrices.
        """
        if len(self.inputs) != 1:
            raise ValueError(f"OutputLinearRelaxation expects exactly one input, got {len(self.inputs)}")

        batch_shape = self.output_shape[:batch_ndim]
        feature_shape = self.output_shape[batch_ndim:]
        feature_numel = 1
        for d in feature_shape:
            feature_numel *= d

        # Identity over features: each output element maps to itself.
        identity = torch.eye(feature_numel, dtype=dtype, device=device)
        identity = identity.reshape(*feature_shape, *feature_shape)

        if batch_shape:
            identity = identity.expand(*batch_shape, *identity.shape)

        return self.inputs[0].backward(identity, identity, batch_ndim)


@final
@dataclass
class InputIdentityRelaxation(SymbolicLinearRelaxation):
    """Identity relaxation at a network input (placeholder) node.

    ``backward`` produces LinearBounds whose linear terms are the
    incoming A-matrices themselves (the identity composition) and whose
    biases are zero.
    """

    input_region: SimpleRegion

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        input_ndim = len(self.input_region.shape) - batch_ndim
        bias_shape = A_lower.shape[: A_lower.ndim - input_ndim]
        return LinearBounds(
            regions=[self.input_region],
            linear_lower=[A_lower],
            bias_lower=torch.zeros(bias_shape, dtype=A_lower.dtype, device=A_lower.device),
            linear_upper=[A_upper],
            bias_upper=torch.zeros(bias_shape, dtype=A_upper.dtype, device=A_upper.device),
            input_ids=[id(self.input_region)],
            validate=False,
        )


def _merge_backward_bounds(
    bounds_list: list[LinearBounds],
    bias_lower: torch.Tensor,
    bias_upper: torch.Tensor,
) -> LinearBounds:
    """Merge LinearBounds from multiple backward calls by input_id."""
    merged: dict[int, tuple[SimpleRegion, torch.Tensor, torch.Tensor]] = {}
    ordered_ids: list[int] = []

    for bounds in bounds_list:
        for iid, region, wl, wu in zip(
            bounds.input_ids, bounds.regions, bounds.linear_lowers, bounds.linear_uppers, strict=True
        ):
            if iid in merged:
                merged[iid] = (merged[iid][0], merged[iid][1] + wl, merged[iid][2] + wu)
            else:
                ordered_ids.append(iid)
                merged[iid] = (region, wl, wu)

        bias_lower = bias_lower + bounds.bias_lower
        bias_upper = bias_upper + bounds.bias_upper

    regions = [merged[iid][0] for iid in ordered_ids]
    linear_lower = [merged[iid][1] for iid in ordered_ids]
    linear_upper = [merged[iid][2] for iid in ordered_ids]

    return LinearBounds(
        regions=regions,
        linear_lower=linear_lower or None,
        bias_lower=bias_lower,
        linear_upper=linear_upper or None,
        bias_upper=bias_upper,
        input_ids=ordered_ids or None,
        validate=False,
    )


# ======================================================================
# Symbolic types for exact (non-relaxation) operations
# ======================================================================


@final
@dataclass
class SymbolicIntervalLeaf(SymbolicLinearRelaxation):
    """Leaf node holding fixed interval bounds (no linear dependency on inputs).

    Used for constants or for operations that break the symbolic chain
    (e.g., nonlinear reductions).  On ``backward``, produces bias-only
    LinearBounds by contracting A over the interval via sign decomposition.
    """

    lower: torch.Tensor
    upper: torch.Tensor

    def backward(self, A_lower: torch.Tensor, A_upper: torch.Tensor, batch_ndim: int) -> LinearBounds:
        node_ndim = self.lower.ndim - batch_ndim
        bounded_ndim = A_lower.ndim - self.lower.ndim

        def bc(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(t.shape[:batch_ndim] + (1,) * bounded_ndim + t.shape[batch_ndim:])

        A_l_pos = A_lower.clamp(min=0)
        A_l_neg = A_lower.clamp(max=0)
        A_u_pos = A_upper.clamp(min=0)
        A_u_neg = A_upper.clamp(max=0)

        sum_dims = tuple(range(-node_ndim, 0)) if node_ndim > 0 else ()
        bias_lower = A_l_pos * bc(self.lower) + A_l_neg * bc(self.upper)
        bias_upper = A_u_pos * bc(self.upper) + A_u_neg * bc(self.lower)
        if sum_dims:
            bias_lower = bias_lower.sum(dim=sum_dims)
            bias_upper = bias_upper.sum(dim=sum_dims)

        return LinearBounds(
            regions=[],
            linear_lower=[],
            bias_lower=bias_lower,
            linear_upper=[],
            bias_upper=bias_upper,
            validate=False,
        )
