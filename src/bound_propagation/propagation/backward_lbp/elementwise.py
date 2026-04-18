"""Backward LBP strategies for element-wise nonlinear operations.

These strategies use the tape to concretize the input node's bounds,
compute element-wise linear relaxation parameters, then wrap them in
an ``ElementwiseBackwardRelaxation`` that implements the sign-decomposition
backward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.fx as fx
from beartype.typing import final

from ..linear_relaxations.elementwise import (
    ElementwiseParams,
    compute_abs_relaxation,
    compute_clamp_relaxation,
    compute_cos_relaxation,
    compute_exp_relaxation,
    compute_log_relaxation,
    compute_reciprocal_relaxation,
    compute_relu_relaxation,
    compute_sigmoid_relaxation,
    compute_sin_relaxation,
    compute_sqrt_relaxation,
    compute_tan_relaxation,
    compute_tanh_relaxation,
)
from .base import BackwardContributions, BackwardLBPStrategy, BackwardRelaxation

if TYPE_CHECKING:
    from ...bounds import IntervalBounds
    from .tape import BackwardTape


@final
@dataclass
class ElementwiseBackwardRelaxation(BackwardRelaxation):
    """Backward relaxation for element-wise nonlinear operations.

    Performs sign decomposition on the alpha/beta coefficients to propagate
    A-matrices backward through the element-wise relaxation.

    Attributes
    ----------
    params : ElementwiseParams
        The alpha/beta relaxation parameters for this element-wise operation.
    input_node : fx.Node
        The single predecessor node (input to the element-wise operation).
    """

    params: ElementwiseParams
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        """Return the single input node as the only predecessor."""
        return [self.input_node]

    def backward_through(
        self,
        A_lower: torch.Tensor,
        A_upper: torch.Tensor,
        batch_ndim: int,
    ) -> BackwardContributions:
        """Propagate A-matrices backward through this element-wise relaxation.

        Uses sign decomposition: where A > 0, use same-side relaxation;
        where A < 0, use opposite-side relaxation.

        Parameters
        ----------
        A_lower : torch.Tensor
            Lower A-matrix from the output, shape ``(*batch, *bounded_out, *node)``.
        A_upper : torch.Tensor
            Upper A-matrix from the output, shape ``(*batch, *bounded_out, *node)``.
        batch_ndim : int
            Number of leading batch dimensions in the A-matrices.

        Returns
        -------
        BackwardContributions
            The propagated A-terms for the input node and bias contributions.
        """
        node_ndim = self.params.alpha_lower.ndim - batch_ndim
        bounded_ndim = A_lower.ndim - self.params.alpha_lower.ndim

        def bc(t: torch.Tensor) -> torch.Tensor:
            """Broadcast ``(*batch, *node)`` to ``(*batch, *bounded_out, *node)``."""
            return t.reshape(t.shape[:batch_ndim] + (1,) * bounded_ndim + t.shape[batch_ndim:])

        A_l_pos = A_lower.clamp(min=0)
        A_l_neg = A_lower.clamp(max=0)
        A_u_pos = A_upper.clamp(min=0)
        A_u_neg = A_upper.clamp(max=0)

        # Sign decomposition on alpha coefficients.
        new_A_lower = A_l_pos * bc(self.params.alpha_lower) + A_l_neg * bc(self.params.alpha_upper)
        new_A_upper = A_u_pos * bc(self.params.alpha_upper) + A_u_neg * bc(self.params.alpha_lower)

        # Bias contribution: sign decomposition on beta, summed over trailing node dims.
        sum_dims = tuple(range(-node_ndim, 0)) if node_ndim > 0 else ()
        delta_bias_lower = A_l_pos * bc(self.params.beta_lower) + A_l_neg * bc(self.params.beta_upper)
        delta_bias_upper = A_u_pos * bc(self.params.beta_upper) + A_u_neg * bc(self.params.beta_lower)
        if sum_dims:
            delta_bias_lower = delta_bias_lower.sum(dim=sum_dims)
            delta_bias_upper = delta_bias_upper.sum(dim=sum_dims)

        return BackwardContributions(
            a_terms={self.input_node: (new_A_lower, new_A_upper)},
            bias_lower=delta_bias_lower,
            bias_upper=delta_bias_upper,
        )


class _ElementwiseBackwardLBP(BackwardLBPStrategy):
    """Base for element-wise nonlinear backward LBP strategies."""

    def _get_input_and_bounds(
        self,
        node: fx.Node,
        tape: BackwardTape,
    ) -> tuple[fx.Node, IntervalBounds]:
        """Get the input fx.Node and its concrete interval bounds.

        Parameters
        ----------
        node : fx.Node
            The current node being processed.
        tape : BackwardTape
            The backward tape for resolving arguments and concretizing bounds.

        Returns
        -------
        tuple[fx.Node, IntervalBounds]
            The input node and its concretized interval bounds.
        """
        input_node = node.args[0]
        bounds = tape.concretize_at(input_node)
        return input_node, bounds


class BackwardLBPRelu(_ElementwiseBackwardLBP):
    """Backward LBP strategy for ReLU."""

    def build_relaxation(self, node: fx.Node, tape: BackwardTape) -> ElementwiseBackwardRelaxation:
        input_node, bounds = self._get_input_and_bounds(node, tape)
        params = compute_relu_relaxation(bounds, adaptive=False)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPSigmoid(_ElementwiseBackwardLBP):
    """Backward LBP strategy for sigmoid."""

    def build_relaxation(self, node: fx.Node, tape: BackwardTape) -> ElementwiseBackwardRelaxation:
        input_node, bounds = self._get_input_and_bounds(node, tape)
        params = compute_sigmoid_relaxation(bounds)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPTanh(_ElementwiseBackwardLBP):
    """Backward LBP strategy for tanh."""

    def build_relaxation(self, node: fx.Node, tape: BackwardTape) -> ElementwiseBackwardRelaxation:
        input_node, bounds = self._get_input_and_bounds(node, tape)
        params = compute_tanh_relaxation(bounds)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPExp(_ElementwiseBackwardLBP):
    """Backward LBP strategy for exp."""

    def build_relaxation(self, node: fx.Node, tape: BackwardTape) -> ElementwiseBackwardRelaxation:
        input_node, bounds = self._get_input_and_bounds(node, tape)
        params = compute_exp_relaxation(bounds)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPLog(_ElementwiseBackwardLBP):
    """Backward LBP strategy for log."""

    def build_relaxation(self, node: fx.Node, tape: BackwardTape) -> ElementwiseBackwardRelaxation:
        input_node, bounds = self._get_input_and_bounds(node, tape)
        params = compute_log_relaxation(bounds)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPSqrt(_ElementwiseBackwardLBP):
    """Backward LBP strategy for sqrt."""

    def build_relaxation(self, node: fx.Node, tape: BackwardTape) -> ElementwiseBackwardRelaxation:
        input_node, bounds = self._get_input_and_bounds(node, tape)
        params = compute_sqrt_relaxation(bounds)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPReciprocal(_ElementwiseBackwardLBP):
    """Backward LBP strategy for reciprocal."""

    def build_relaxation(self, node: fx.Node, tape: BackwardTape) -> ElementwiseBackwardRelaxation:
        input_node, bounds = self._get_input_and_bounds(node, tape)
        params = compute_reciprocal_relaxation(bounds)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPAbs(_ElementwiseBackwardLBP):
    """Backward LBP strategy for abs."""

    def build_relaxation(self, node: fx.Node, tape: BackwardTape) -> ElementwiseBackwardRelaxation:
        input_node, bounds = self._get_input_and_bounds(node, tape)
        params = compute_abs_relaxation(bounds)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPSin(_ElementwiseBackwardLBP):
    """Backward LBP strategy for sin."""

    def build_relaxation(self, node: fx.Node, tape: BackwardTape) -> ElementwiseBackwardRelaxation:
        input_node, bounds = self._get_input_and_bounds(node, tape)
        params = compute_sin_relaxation(bounds)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPCos(_ElementwiseBackwardLBP):
    """Backward LBP strategy for cos."""

    def build_relaxation(self, node: fx.Node, tape: BackwardTape) -> ElementwiseBackwardRelaxation:
        input_node, bounds = self._get_input_and_bounds(node, tape)
        params = compute_cos_relaxation(bounds)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPTan(_ElementwiseBackwardLBP):
    """Backward LBP strategy for tan."""

    def build_relaxation(self, node: fx.Node, tape: BackwardTape) -> ElementwiseBackwardRelaxation:
        input_node, bounds = self._get_input_and_bounds(node, tape)
        params = compute_tan_relaxation(bounds)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPClamp(BackwardLBPStrategy):
    """Backward LBP strategy for clamp.

    Unlike other element-wise strategies, clamp requires resolving the
    ``min`` and ``max`` arguments from the tape before computing the relaxation.
    """

    def build_relaxation(self, node: fx.Node, tape: BackwardTape) -> ElementwiseBackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        input_node = node.args[0]
        min_val = args[1] if len(args) > 1 else kwargs.get("min")
        max_val = args[2] if len(args) > 2 else kwargs.get("max")
        bounds = tape.concretize_at(input_node)
        params = compute_clamp_relaxation(bounds, min_val, max_val)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)
