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

from ...linear_operators import LinearOperator
from ..linear_relaxations.alpha_resolvers import (
    resolve_abs_alpha,
    resolve_clamp_alphas,
    resolve_cos_alpha,
    resolve_exp_alpha,
    resolve_log_alpha,
    resolve_reciprocal_alphas,
    resolve_relu_alpha,
    resolve_sigmoid_alphas,
    resolve_sin_alpha,
    resolve_sqrt_alpha,
    resolve_tan_alpha,
    resolve_tanh_alphas,
)
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
from .base import (
    BackwardContributions,
    BackwardLBPStrategy,
    BackwardRelaxation,
    IntermediateBoundsProvider,
    _wrap_a_term_tensors,
)

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
        A_lower: LinearOperator,
        A_upper: LinearOperator,
        batch_ndim: int,
    ) -> BackwardContributions:
        """Propagate A-matrices backward through this element-wise relaxation.

        Uses sign decomposition: where A > 0, use same-side relaxation;
        where A < 0, use opposite-side relaxation.
        """
        output_shape = A_lower.output_shape

        node_ndim = self.params.alpha_lower.ndim - batch_ndim
        bounded_ndim = A_lower.output_ndim + A_lower.input_ndim - self.params.alpha_lower.ndim

        def bc(t: torch.Tensor) -> torch.Tensor:
            """Broadcast ``(*batch, *node)`` to ``(*batch, *bounded_out, *node)``."""
            return t.reshape(t.shape[:batch_ndim] + (1,) * bounded_ndim + t.shape[batch_ndim:])

        A_l_pos = A_lower.clamp_min(0).to_dense().tensor
        A_l_neg = A_lower.clamp_max(0).to_dense().tensor
        A_u_pos = A_upper.clamp_min(0).to_dense().tensor
        A_u_neg = A_upper.clamp_max(0).to_dense().tensor

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
            a_terms=_wrap_a_term_tensors({self.input_node: (new_A_lower, new_A_upper)}, len(output_shape)),
            bias_lower=delta_bias_lower,
            bias_upper=delta_bias_upper,
        )


class _ElementwiseBackwardLBP(BackwardLBPStrategy):
    """Base for element-wise nonlinear backward LBP strategies."""

    def _get_input_and_bounds(
        self,
        node: fx.Node,
        bounds: IntermediateBoundsProvider,
    ) -> tuple[fx.Node, IntervalBounds]:
        """Get the input fx.Node and its concrete interval bounds.

        Parameters
        ----------
        node : fx.Node
            The current node being processed.
        bounds : IntermediateBoundsProvider
            Callable returning interval bounds for a predecessor node.

        Returns
        -------
        tuple[fx.Node, IntervalBounds]
            The input node and its concretized interval bounds.
        """
        input_node = node.args[0]
        return input_node, bounds(input_node)


class BackwardLBPRelu(_ElementwiseBackwardLBP):
    """Backward LBP strategy for ReLU."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> ElementwiseBackwardRelaxation:
        input_node, input_bounds = self._get_input_and_bounds(node, bounds)
        alpha = resolve_relu_alpha(tape.alpha_provider, node, input_bounds)
        params = compute_relu_relaxation(input_bounds, adaptive=False, alpha_relu_lower=alpha)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPSigmoid(_ElementwiseBackwardLBP):
    """Backward LBP strategy for sigmoid."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> ElementwiseBackwardRelaxation:
        input_node, input_bounds = self._get_input_and_bounds(node, bounds)
        alpha_lo, alpha_up = resolve_sigmoid_alphas(tape.alpha_provider, node, input_bounds)
        params = compute_sigmoid_relaxation(
            input_bounds,
            alpha_sigmoid_tangent_lower=alpha_lo,
            alpha_sigmoid_tangent_upper=alpha_up,
        )
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPTanh(_ElementwiseBackwardLBP):
    """Backward LBP strategy for tanh."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> ElementwiseBackwardRelaxation:
        input_node, input_bounds = self._get_input_and_bounds(node, bounds)
        alpha_lo, alpha_up = resolve_tanh_alphas(tape.alpha_provider, node, input_bounds)
        params = compute_tanh_relaxation(
            input_bounds,
            alpha_tanh_tangent_lower=alpha_lo,
            alpha_tanh_tangent_upper=alpha_up,
        )
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPExp(_ElementwiseBackwardLBP):
    """Backward LBP strategy for exp."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> ElementwiseBackwardRelaxation:
        input_node, input_bounds = self._get_input_and_bounds(node, bounds)
        alpha = resolve_exp_alpha(tape.alpha_provider, node, input_bounds)
        params = compute_exp_relaxation(input_bounds, alpha_exp_tangent_lower=alpha)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPLog(_ElementwiseBackwardLBP):
    """Backward LBP strategy for log."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> ElementwiseBackwardRelaxation:
        input_node, input_bounds = self._get_input_and_bounds(node, bounds)
        alpha = resolve_log_alpha(tape.alpha_provider, node, input_bounds)
        params = compute_log_relaxation(input_bounds, alpha_log_tangent_upper=alpha)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPSqrt(_ElementwiseBackwardLBP):
    """Backward LBP strategy for sqrt."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> ElementwiseBackwardRelaxation:
        input_node, input_bounds = self._get_input_and_bounds(node, bounds)
        alpha = resolve_sqrt_alpha(tape.alpha_provider, node, input_bounds)
        params = compute_sqrt_relaxation(input_bounds, alpha_sqrt_tangent_upper=alpha)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPReciprocal(_ElementwiseBackwardLBP):
    """Backward LBP strategy for reciprocal."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> ElementwiseBackwardRelaxation:
        input_node, input_bounds = self._get_input_and_bounds(node, bounds)
        alpha_lo, alpha_up = resolve_reciprocal_alphas(tape.alpha_provider, node, input_bounds)
        params = compute_reciprocal_relaxation(
            input_bounds,
            alpha_reciprocal_tangent_lower=alpha_lo,
            alpha_reciprocal_tangent_upper=alpha_up,
        )
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPAbs(_ElementwiseBackwardLBP):
    """Backward LBP strategy for abs."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> ElementwiseBackwardRelaxation:
        input_node, input_bounds = self._get_input_and_bounds(node, bounds)
        alpha = resolve_abs_alpha(tape.alpha_provider, node, input_bounds)
        params = compute_abs_relaxation(input_bounds, alpha_abs_lower=alpha)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPSin(_ElementwiseBackwardLBP):
    """Backward LBP strategy for sin."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> ElementwiseBackwardRelaxation:
        input_node, input_bounds = self._get_input_and_bounds(node, bounds)
        alpha = resolve_sin_alpha(tape.alpha_provider, node, input_bounds)
        params = compute_sin_relaxation(input_bounds, alpha_sin_tangent_frac=alpha)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPCos(_ElementwiseBackwardLBP):
    """Backward LBP strategy for cos."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> ElementwiseBackwardRelaxation:
        input_node, input_bounds = self._get_input_and_bounds(node, bounds)
        alpha = resolve_cos_alpha(tape.alpha_provider, node, input_bounds)
        params = compute_cos_relaxation(input_bounds, alpha_cos_tangent_frac=alpha)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPTan(_ElementwiseBackwardLBP):
    """Backward LBP strategy for tan."""

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> ElementwiseBackwardRelaxation:
        input_node, input_bounds = self._get_input_and_bounds(node, bounds)
        alpha = resolve_tan_alpha(tape.alpha_provider, node, input_bounds)
        params = compute_tan_relaxation(input_bounds, alpha_tan_tangent_frac=alpha)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


class BackwardLBPClamp(BackwardLBPStrategy):
    """Backward LBP strategy for clamp.

    Unlike other element-wise strategies, clamp requires resolving the
    ``min`` and ``max`` arguments from the tape before computing the relaxation.
    """

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> ElementwiseBackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        input_node = node.args[0]
        min_val = args[1] if len(args) > 1 else kwargs.get("min")
        max_val = args[2] if len(args) > 2 else kwargs.get("max")
        input_bounds = bounds(input_node)
        alpha_cm_lower, alpha_cmx_upper = resolve_clamp_alphas(tape.alpha_provider, node, input_bounds)
        params = compute_clamp_relaxation(
            input_bounds,
            min_val,
            max_val,
            alpha_clamp_crosses_min_lower=alpha_cm_lower,
            alpha_clamp_crosses_max_upper=alpha_cmx_upper,
        )
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)
