from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    import torch.fx as fx

    from ..context import PropagationContext


class IBPClamp(ForwardIBPStrategy):
    """IBP strategy for clamp."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPClamp requires input to be IntervalBounds")

        clamp_min = args[1] if len(args) > 1 else kwargs.get("min", None)
        clamp_max = args[2] if len(args) > 2 else kwargs.get("max", None)

        lower = torch.clamp(x_bounds.lower, min=clamp_min, max=clamp_max)
        upper = torch.clamp(x_bounds.upper, min=clamp_min, max=clamp_max)

        return IntervalBounds(lower, upper, batch_ndim=x_bounds.batch_ndim)


class IBPAbs(ForwardIBPStrategy):
    """IBP strategy for abs: abs([a, b]) accounts for sign-crossing."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPAbs requires input to be IntervalBounds")

        abs_lower = x_bounds.lower.abs()
        abs_upper = x_bounds.upper.abs()

        zero = torch.zeros_like(abs_lower)
        lower = torch.where(
            (x_bounds.lower < 0) & (x_bounds.upper > 0),
            zero,
            torch.min(abs_lower, abs_upper),
        )
        upper = torch.max(abs_lower, abs_upper)

        return IntervalBounds(lower, upper, batch_ndim=x_bounds.batch_ndim)


class IBPCbrt(ForwardIBPStrategy):
    """IBP strategy for cbrt (monotonic)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPCbrt requires input to be IntervalBounds")

        lower = torch.copysign(torch.pow(x_bounds.lower.abs(), 1 / 3), x_bounds.lower)
        upper = torch.copysign(torch.pow(x_bounds.upper.abs(), 1 / 3), x_bounds.upper)

        return IntervalBounds(lower, upper, batch_ndim=x_bounds.batch_ndim)


class IBPCos(ForwardIBPStrategy):
    """IBP strategy for cos with peak/trough analysis."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPCos requires input to be IntervalBounds")

        two_pi = 2 * torch.pi
        pi = torch.pi

        includes_peak = torch.floor(x_bounds.upper / two_pi) >= torch.ceil(x_bounds.lower / two_pi)

        includes_trough = torch.floor((x_bounds.upper - pi) / two_pi) >= torch.ceil((x_bounds.lower - pi) / two_pi)

        cos_lower = torch.cos(x_bounds.lower)
        cos_upper = torch.cos(x_bounds.upper)

        lower = torch.where(
            includes_trough,
            torch.tensor(-1.0, device=x_bounds.lower.device, dtype=x_bounds.lower.dtype),
            torch.min(cos_lower, cos_upper),
        )
        upper = torch.where(
            includes_peak,
            torch.tensor(1.0, device=x_bounds.upper.device, dtype=x_bounds.upper.dtype),
            torch.max(cos_lower, cos_upper),
        )

        return IntervalBounds(lower, upper, batch_ndim=x_bounds.batch_ndim)


class IBPExp(ForwardIBPStrategy):
    """IBP strategy for exp (monotone): exp([a, b]) = [exp(a), exp(b)]."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPExp requires input to be IntervalBounds")

        return IntervalBounds(torch.exp(x_bounds.lower), torch.exp(x_bounds.upper), batch_ndim=x_bounds.batch_ndim)


class IBPLog(ForwardIBPStrategy):
    """IBP strategy for log (monotone for positive inputs)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPLog requires input to be IntervalBounds")

        if torch.any(x_bounds.lower <= 0):
            raise ValueError("log requires positive input bounds")

        return IntervalBounds(torch.log(x_bounds.lower), torch.log(x_bounds.upper), batch_ndim=x_bounds.batch_ndim)


class IBPPow(ForwardIBPStrategy):
    """IBP strategy for integer power."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPPow requires input to be IntervalBounds")

        power = args[1] if len(args) > 1 else kwargs.get("exponent", 1)

        if isinstance(power, torch.Tensor) and torch.is_floating_point(power):
            raise TypeError("IBPPow requires power to be an integer tensor")

        lower, upper = x_bounds.lower, x_bounds.upper

        if isinstance(power, torch.Tensor):
            return self._tensor_power(lower, upper, power, x_bounds.batch_ndim)

        return self._scalar_power(lower, upper, power, x_bounds.batch_ndim)

    @staticmethod
    def _scalar_power(lower: torch.Tensor, upper: torch.Tensor, power: int, batch_ndim: int) -> IntervalBounds:
        if power == 0:
            ones = torch.ones_like(lower)
            return IntervalBounds(ones, ones, batch_ndim=batch_ndim)

        if power % 2 == 0:
            lower_act = torch.pow(lower, power)
            upper_act = torch.pow(upper, power)
            lower_out = torch.where((lower < 0) & (upper > 0), 0.0, torch.min(lower_act, upper_act))
            upper_out = torch.max(lower_act, upper_act)
            return IntervalBounds(lower_out, upper_out, batch_ndim=batch_ndim)

        return IntervalBounds(torch.pow(lower, power), torch.pow(upper, power), batch_ndim=batch_ndim)

    @staticmethod
    def _tensor_power(lower: torch.Tensor, upper: torch.Tensor, power: torch.Tensor, batch_ndim: int) -> IntervalBounds:
        if torch.all(power == 0):
            ones = torch.ones_like(lower)
            return IntervalBounds(ones, ones, batch_ndim=batch_ndim)

        lower_act = torch.pow(lower, power)
        upper_act = torch.pow(upper, power)

        lower_out = torch.zeros_like(lower)
        upper_out = torch.zeros_like(upper)

        power_expanded = power.expand_as(lower)
        zero_power = power_expanded == 0
        even = ((power_expanded % 2) == 0) & (~zero_power)
        odd = (~even) & (~zero_power)
        crossing = (lower < 0) & (upper > 0)

        # power == 0
        lower_out[zero_power] = 1.0
        upper_out[zero_power] = 1.0

        # Even powers
        crossing_even = crossing & even
        not_crossing_even = (~crossing) & even
        lower_out[crossing_even] = 0.0
        lower_out[not_crossing_even] = torch.min(lower_act[not_crossing_even], upper_act[not_crossing_even])
        upper_out[even] = torch.max(lower_act[even], upper_act[even])

        # Odd powers
        lower_out[odd] = lower_act[odd]
        upper_out[odd] = upper_act[odd]

        return IntervalBounds(lower_out, upper_out, batch_ndim=batch_ndim)


class IBPReciprocal(ForwardIBPStrategy):
    """IBP strategy for reciprocal: 1/[a, b]."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPReciprocal requires input to be IntervalBounds")

        unbounded_mask = (x_bounds.lower <= 0) & (x_bounds.upper >= 0)

        lower = 1 / x_bounds.upper
        lower = torch.where(unbounded_mask, float("-inf"), lower)

        upper = 1 / x_bounds.lower
        upper = torch.where(unbounded_mask, float("inf"), upper)

        return IntervalBounds(lower, upper, batch_ndim=x_bounds.batch_ndim)


class IBPRelu(ForwardIBPStrategy):
    """IBP strategy for relu: relu([a, b]) = [max(0, a), max(0, b)]."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPRelu requires input to be IntervalBounds")

        lower = torch.clamp(x_bounds.lower, min=0.0)
        upper = torch.clamp(x_bounds.upper, min=0.0)

        return IntervalBounds(lower, upper, batch_ndim=x_bounds.batch_ndim)


class IBPSigmoid(ForwardIBPStrategy):
    """IBP strategy for sigmoid (monotone)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPSigmoid requires input to be IntervalBounds")

        return IntervalBounds(
            torch.sigmoid(x_bounds.lower), torch.sigmoid(x_bounds.upper), batch_ndim=x_bounds.batch_ndim
        )


class IBPSin(ForwardIBPStrategy):
    """IBP strategy for sin with peak/trough analysis."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPSin requires input to be IntervalBounds")

        two_pi = 2 * torch.pi
        pi = torch.pi
        pi_over_2 = pi / 2
        three_pi_over_2 = 3 * pi / 2

        includes_peak = torch.floor((x_bounds.upper - pi_over_2) / two_pi) >= torch.ceil(
            (x_bounds.lower - pi_over_2) / two_pi
        )

        includes_trough = torch.floor((x_bounds.upper - three_pi_over_2) / two_pi) >= torch.ceil(
            (x_bounds.lower - three_pi_over_2) / two_pi
        )

        sin_lower = torch.sin(x_bounds.lower)
        sin_upper = torch.sin(x_bounds.upper)

        lower = torch.where(
            includes_trough,
            torch.tensor(-1.0, device=x_bounds.lower.device, dtype=x_bounds.lower.dtype),
            torch.min(sin_lower, sin_upper),
        )
        upper = torch.where(
            includes_peak,
            torch.tensor(1.0, device=x_bounds.upper.device, dtype=x_bounds.upper.dtype),
            torch.max(sin_lower, sin_upper),
        )

        return IntervalBounds(lower, upper, batch_ndim=x_bounds.batch_ndim)


class IBPSqrt(ForwardIBPStrategy):
    """IBP strategy for sqrt (monotone for non-negative inputs)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPSqrt requires input to be IntervalBounds")

        if torch.any(x_bounds.lower < 0):
            raise ValueError("sqrt requires non-negative input bounds")

        return IntervalBounds(torch.sqrt(x_bounds.lower), torch.sqrt(x_bounds.upper), batch_ndim=x_bounds.batch_ndim)


class IBPTan(ForwardIBPStrategy):
    """IBP strategy for tan with asymptote detection."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPTan requires input to be IntervalBounds")

        lower = torch.tan(x_bounds.lower)
        upper = torch.tan(x_bounds.upper)

        pi_over_2 = torch.pi / 2
        eps = torch.finfo(x_bounds.lower.dtype).eps * 8
        k_min = torch.ceil((x_bounds.lower - pi_over_2 - eps) / torch.pi)
        k_max = torch.floor((x_bounds.upper - pi_over_2 + eps) / torch.pi)

        contains_asymptote = k_min <= k_max
        lower[contains_asymptote] = float("-inf")
        upper[contains_asymptote] = float("inf")

        return IntervalBounds(lower, upper, batch_ndim=x_bounds.batch_ndim)


class IBPTanh(ForwardIBPStrategy):
    """IBP strategy for tanh (monotone)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPTanh requires input to be IntervalBounds")

        return IntervalBounds(torch.tanh(x_bounds.lower), torch.tanh(x_bounds.upper), batch_ndim=x_bounds.batch_ndim)
