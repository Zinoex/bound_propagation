"""Backward LBP element-wise strategies — driven by :class:`ElementwiseSpec`.

Each strategy concretizes the predecessor's bounds via the
:class:`IntermediateBoundsProvider`, calls the spec's ``compute_*_relaxation``,
and wraps the slopes/biases in :class:`ElementwiseBackwardRelaxation` for the
sign-decomposition backward pass driven by the tape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.fx as fx
from beartype.typing import final

from ...linear_operators import LinearOperator
from ..linear_relaxations.elementwise import ElementwiseParams
from ..linear_relaxations.elementwise_specs import (
    ABS_SPEC,
    CLAMP_SPEC,
    COS_SPEC,
    EXP_SPEC,
    LOG_SPEC,
    POW_SPEC,
    RECIPROCAL_SPEC,
    RELU_SPEC,
    SIGMOID_SPEC,
    SIN_SPEC,
    SQRT_SPEC,
    TAN_SPEC,
    TANH_SPEC,
    ElementwiseSpec,
)
from .base import (
    BackwardContributions,
    BackwardLBPStrategy,
    BackwardRelaxation,
    IntermediateBoundsProvider,
    _wrap_a_term_tensors,
)

if TYPE_CHECKING:
    from .tape import BackwardTape


@final
@dataclass
class ElementwiseBackwardRelaxation(BackwardRelaxation):
    """Backward relaxation for an element-wise nonlinear operation.

    Holds the per-element slopes/biases ``(α_L, β_L, α_U, β_U)`` produced by
    the spec's ``compute_*_relaxation``. The backward pass applies sign
    decomposition on the running ``A`` against ``α_L`` / ``α_U`` (with the
    matching β contributions accumulated into the bias).
    """

    params: ElementwiseParams
    input_node: fx.Node

    def predecessor_nodes(self) -> list[fx.Node]:
        return [self.input_node]

    def backward_through(
        self,
        A_lower: LinearOperator,
        A_upper: LinearOperator,
        batch_ndim: int,
    ) -> BackwardContributions:
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

        # Bias contribution: same sign decomposition on beta, summed over trailing node dims.
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


class BackwardLBPElementwise(BackwardLBPStrategy):
    """Generic backward-LBP strategy for element-wise nonlinear ops.

    Configured by an :class:`ElementwiseSpec` — see
    :mod:`propagation.linear_relaxations.elementwise_specs`.
    """

    def __init__(self, spec: ElementwiseSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ElementwiseSpec:
        return self._spec

    def build_relaxation(
        self, node: fx.Node, tape: BackwardTape, bounds: IntermediateBoundsProvider
    ) -> ElementwiseBackwardRelaxation:
        args, kwargs = tape.resolve_args(node)
        input_node = node.args[0]
        if not isinstance(input_node, fx.Node):
            raise TypeError(f"BackwardLBPElementwise[{self._spec.name}] expects a single fx.Node input")
        input_bounds = bounds(input_node)
        params = self._spec.build_params(input_bounds, node, tape.alpha_provider, args, kwargs)
        return ElementwiseBackwardRelaxation(params=params, input_node=input_node)


# ---------------------------------------------------------------------------
# Per-op strategy classes. Each is a 3-line wrapper that bakes its spec into
# the generic strategy.
# ---------------------------------------------------------------------------


class BackwardLBPRelu(BackwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(RELU_SPEC)


class BackwardLBPSigmoid(BackwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(SIGMOID_SPEC)


class BackwardLBPTanh(BackwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(TANH_SPEC)


class BackwardLBPExp(BackwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(EXP_SPEC)


class BackwardLBPLog(BackwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(LOG_SPEC)


class BackwardLBPSqrt(BackwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(SQRT_SPEC)


class BackwardLBPReciprocal(BackwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(RECIPROCAL_SPEC)


class BackwardLBPAbs(BackwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(ABS_SPEC)


class BackwardLBPSin(BackwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(SIN_SPEC)


class BackwardLBPCos(BackwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(COS_SPEC)


class BackwardLBPTan(BackwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(TAN_SPEC)


class BackwardLBPPow(BackwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(POW_SPEC)


class BackwardLBPClamp(BackwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(CLAMP_SPEC)
