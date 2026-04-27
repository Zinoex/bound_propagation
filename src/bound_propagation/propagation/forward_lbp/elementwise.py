"""Forward LBP element-wise strategies — driven by :class:`ElementwiseSpec`.

Every element-wise activation wires the same forward composition: concretize
the input ``LinearBounds`` to an interval, compute slopes/biases via the
spec's ``compute_*_relaxation``, then re-compose with the input via signed
clamping in :class:`ElementwiseForwardRelaxation`. The 13 supported ops differ
only in which compute function and α-resolver feed the spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from ...bounds import LinearBounds
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
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch.fx as fx

    from ..context import PropagationContext


@final
@dataclass
class ElementwiseForwardRelaxation:
    """Element-wise linear relaxation ``y = α x + β`` composed forward.

    Stores four element-wise tensors (same shape as x / y):

        y_lower >= alpha_lower * x + beta_lower
        y_upper <= alpha_upper * x + beta_upper

    The abstract dimension convention for ``LinearBounds`` linear terms is
    ``(*batch_dims, *output_dims, *input_dims)``. ``alpha`` / ``beta`` live in
    ``(*batch_dims, *output_dims)``; forward composition appends the input
    trailing axes via broadcasting.
    """

    params: ElementwiseParams

    def forward(self, input_bounds: LinearBounds) -> LinearBounds:
        """Compose ``y = α x + β`` with ``x = W x0 + b`` → ``y = W' x0 + b'``.

        Sign decomposition on α: positive coefficients pair with the same-side
        running operator, negatives with the opposite side. Uses
        :class:`LinearOperator` algebra (``.scale``/``.add``) so structured
        operators stay structured when the operations support it.
        """
        al_pos = self.params.alpha_lower.clamp(min=0)
        al_neg = self.params.alpha_lower.clamp(max=0)
        au_pos = self.params.alpha_upper.clamp(min=0)
        au_neg = self.params.alpha_upper.clamp(max=0)

        linear_lower_ops = [
            lower_op.scale(al_pos).add(upper_op.scale(al_neg))
            for lower_op, upper_op in zip(input_bounds.linear_lowers_op, input_bounds.linear_uppers_op, strict=True)
        ]
        bias_lower = al_pos * input_bounds.bias_lower + al_neg * input_bounds.bias_upper + self.params.beta_lower

        linear_upper_ops = [
            upper_op.scale(au_pos).add(lower_op.scale(au_neg))
            for lower_op, upper_op in zip(input_bounds.linear_lowers_op, input_bounds.linear_uppers_op, strict=True)
        ]
        bias_upper = au_pos * input_bounds.bias_upper + au_neg * input_bounds.bias_lower + self.params.beta_upper

        return LinearBounds(
            regions=input_bounds.regions,
            linear_lower=linear_lower_ops or None,
            bias_lower=bias_lower,
            linear_upper=linear_upper_ops or None,
            bias_upper=bias_upper,
            input_ids=input_bounds.input_ids or None,
        )


class ForwardLBPElementwise(ForwardLBPStrategy):
    """Generic forward-LBP strategy for element-wise nonlinear ops.

    Configured by an :class:`ElementwiseSpec` — see
    :mod:`propagation.linear_relaxations.elementwise_specs`.
    """

    def __init__(self, spec: ElementwiseSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ElementwiseSpec:
        return self._spec

    def propagate_forward(self, node: fx.Node, ctx: PropagationContext) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]
        if not isinstance(bounds, LinearBounds):
            raise TypeError(f"ForwardLBPElementwise[{self._spec.name}] requires input to be LinearBounds")

        concrete_bounds = bounds.concretize()
        params = self._spec.build_params(concrete_bounds, node, ctx.alpha_provider, args, kwargs)
        return ElementwiseForwardRelaxation(params=params).forward(bounds)


# ---------------------------------------------------------------------------
# Per-op strategy classes. Each is a 3-line wrapper that bakes its spec into
# the generic strategy. Kept as classes (not instances) so call sites can
# instantiate with ``ForwardLBPRelu()`` per the existing public surface.
# ---------------------------------------------------------------------------


class ForwardLBPRelu(ForwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(RELU_SPEC)


class ForwardLBPAbs(ForwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(ABS_SPEC)


class ForwardLBPClamp(ForwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(CLAMP_SPEC)


class ForwardLBPCos(ForwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(COS_SPEC)


class ForwardLBPExp(ForwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(EXP_SPEC)


class ForwardLBPLog(ForwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(LOG_SPEC)


class ForwardLBPReciprocal(ForwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(RECIPROCAL_SPEC)


class ForwardLBPSigmoid(ForwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(SIGMOID_SPEC)


class ForwardLBPSin(ForwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(SIN_SPEC)


class ForwardLBPSqrt(ForwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(SQRT_SPEC)


class ForwardLBPTan(ForwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(TAN_SPEC)


class ForwardLBPTanh(ForwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(TANH_SPEC)


class ForwardLBPPow(ForwardLBPElementwise):
    def __init__(self) -> None:
        super().__init__(POW_SPEC)
