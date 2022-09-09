import abc
import logging
from typing import Tuple

import numpy as np
import torch
from torch import nn

from . import IntervalBounds
from .saturation import Clamp
from .bivariate import Div, VectorSub
from .reshape import Flip
from .activation import BoundSigmoid, BoundActivation, assert_bound_order, bisection
from .linear import ElementWiseLinear
from .util import proj_grad_to_range_, clip_param_to_range_

logger = logging.getLogger(__name__)


class Erf(nn.Module):
    def forward(self, x):
        return torch.special.erf(x)


class BoundErf(BoundSigmoid):
    def derivative(self, x):
        return (2 / np.sqrt(np.pi)) * torch.exp(-x.pow(2))


class StandardNormalPDF(nn.Module):
    # While this class could be made as a sequential with pow(2), linear, exponential, linear,
    # by explicitly bounding a bell curve, the bounds will be tighter. Furthermore, BoundBellCurve
    # will be useful for other bell curve-shaped functions (in particular other distributions).

    def forward(self, x):
        return (1 / np.sqrt(2 * np.pi)) * torch.exp(-0.5 * x.pow(2))


def bell_curve_regimes(lower: torch.Tensor, upper: torch.Tensor, top_point) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero_width = torch.isclose(lower, upper, rtol=0.0, atol=1e-8) & torch.full_like(top_point, True, dtype=torch.bool)
    l = (~zero_width) & (upper <= top_point)
    u = (~zero_width) & (top_point <= lower)
    lu = (~zero_width) & (lower < top_point) & (top_point < upper)

    return zero_width, l, u, lu


class BoundBellCurve(BoundActivation, abc.ABC):
    midpoint = 0.0
    # - For some bell curves (non-Gaussian), the inflection points are unknown, but we do know that there are two and
    # we may compute upper and lower bounds for each inflection point (e.g. using bisection).
    # - Also note that inflection points may be tensors to allow data parallel computation of linear bounds for
    # multiple different bell curves.
    lower_inflection = (-1.0, -1.0)
    upper_inflection = (1.0, 1.0)

    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    def clear_relaxation(self):
        super().clear_relaxation()
        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    @abc.abstractmethod
    def derivative(self, x, mask=None):
        raise NotImplementedError()

    def func(self, x, mask=None):
        return self(x)

    @assert_bound_order
    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower, preactivation.upper

        zero_width, l, u, lu = bell_curve_regimes(lower, upper, self.midpoint)
        ones = torch.ones_like(zero_width, dtype=lower.dtype)
        lower, upper = ones * lower, ones * upper

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        lower_act, upper_act = self(lower), self(upper)
        lower_prime, upper_prime = self.derivative(lower), self.derivative(upper)
        slope = (upper_act - lower_act) / (upper - lower)
        lower_inflection = self.lower_inflection[0] * ones, self.lower_inflection[1] * ones
        upper_inflection = self.upper_inflection[0] * ones, self.upper_inflection[1] * ones
        mid = self.midpoint * ones

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, torch.min(lower_act[zero_width], upper_act[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, torch.max(lower_act[zero_width], upper_act[zero_width])

        def add_linear(alpha, beta, mask, a, x, y):
            alpha[mask] = a
            beta[mask] = y - a * x

        ###############
        # Lower bound #
        ###############
        lower_is_closer = ((lower - self.midpoint).abs() <= (upper - self.midpoint).abs())
        optimize_upper = (lu & lower_is_closer) | u
        optimize_lower = (lu & (~lower_is_closer)) | l

        # Fix lower input
        # - If we can take the direct slope
        direct_cond = ((lower < self.lower_inflection[1]) & (lower_prime >= slope)) | (lower >= self.lower_inflection[1])
        direct = optimize_lower & direct_cond
        add_linear(self.alpha_lower, self.beta_lower, mask=direct, a=slope[direct], x=upper[direct], y=upper_act[direct])

        # - Else
        indirect = optimize_lower & (~direct_cond)
        # - Bound right input such that the tangent lines are true upper bounds
        right = self.lower_left_tangent_point(indirect, lower, upper, lower_inflection)

        d = (lower[indirect] + right) / 2
        add_linear(self.alpha_lower, self.beta_lower, mask=indirect, a=self.derivative(d, mask=indirect), x=d, y=self.func(d, mask=indirect))

        # Allow parameterization
        # Save mask
        unstable_lower = [indirect]
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        unstable_d_lower = [d.detach().clone().requires_grad_()]
        # Save ranges to clip (aka. PGD)
        unstable_range_lower = [(lower[indirect], right)]

        # Fix upper input
        # - If we can take the direct slope
        direct_cond = ((upper > self.upper_inflection[0]) & (upper_prime <= slope)) | (upper <= self.upper_inflection[0])
        direct = optimize_upper & direct_cond
        add_linear(self.alpha_lower, self.beta_lower, mask=direct, a=slope[direct], x=lower[direct], y=lower_act[direct])

        # - Else
        indirect = optimize_upper & (~direct_cond)
        # - Bound left input such that the tangent lines are true upper bounds
        left = self.lower_right_tangent_point(indirect, lower, upper, upper_inflection)

        d = (left + upper[indirect]) / 2
        add_linear(self.alpha_lower, self.beta_lower, mask=indirect, a=self.derivative(d, mask=indirect), x=d, y=self.func(d, mask=indirect))

        # Allow parameterization
        # Save mask
        unstable_lower.append(indirect)
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        unstable_d_lower.append(d.detach().clone().requires_grad_())
        # Save ranges to clip (aka. PGD)
        unstable_range_lower.append((left, upper[indirect]))

        self.unstable_lower = unstable_lower
        self.unstable_d_lower = unstable_d_lower
        self.unstable_range_lower = unstable_range_lower

        ################
        # Lower regime #
        ################
        # Upper bound
        # - If we can take the direct slope
        direct_cond = ((upper > self.lower_inflection[0]) & (upper_prime >= slope)) | (upper <= self.lower_inflection[0])
        direct = l & direct_cond
        add_linear(self.alpha_upper, self.beta_upper, mask=direct, a=slope[direct], x=upper[direct], y=upper_act[direct])

        # - Else
        indirect = l & (~direct_cond)
        # - Bound left input such that the tangent lines are true upper bounds
        left = self.upper_right_tangent_point(indirect, lower, upper, mid, lower_inflection)

        d = (left + upper[indirect]) / 2
        add_linear(self.alpha_upper, self.beta_upper, mask=indirect, a=self.derivative(d, mask=indirect), x=d, y=self.func(d, mask=indirect))

        # Allow parameterization
        # Save mask
        unstable_upper = [indirect]
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        unstable_d_upper = [d.detach().clone().requires_grad_()]
        # Save ranges to clip (aka. PGD)
        unstable_range_upper = [(left, upper[indirect])]

        ################
        # Upper regime #
        ################
        # Upper bound
        # - If we can take the direct slope
        direct_cond = ((lower < self.upper_inflection[1]) & (lower_prime <= slope)) | (lower >= self.upper_inflection[1])
        direct = u & direct_cond
        add_linear(self.alpha_upper, self.beta_upper, mask=direct, a=slope[direct], x=lower[direct], y=lower_act[direct])

        # - Else
        indirect = u & (~direct_cond)
        # - Bound right input such that the tangent lines are true upper bounds
        right = self.upper_left_tangent_point(indirect, lower, upper, mid, upper_inflection)

        d = (lower[indirect] + right) / 2
        add_linear(self.alpha_upper, self.beta_upper, mask=indirect, a=self.derivative(d, mask=indirect), x=d, y=self.func(d, mask=indirect))

        # Allow parameterization
        # Save mask
        unstable_upper.append(indirect)
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        unstable_d_upper.append(d.detach().clone().requires_grad_())
        # Save ranges to clip (aka. PGD)
        unstable_range_upper.append((lower[indirect], right))

        #####################
        # Crossing midpoint #
        #####################
        # Upper bound
        # - Bound left and right inputs such that the tangent lines are true upper bounds
        left = self.upper_right_tangent_point(lu, lower, upper, mid, lower_inflection)
        right = self.upper_left_tangent_point(lu, lower, upper, mid, upper_inflection)

        d = (left + right) / 2
        add_linear(self.alpha_upper, self.beta_upper, mask=lu, a=self.derivative(d, mask=lu), x=d, y=self.func(d, mask=lu))

        # Allow parameterization
        # Save mask
        unstable_upper.append(lu)
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        unstable_d_upper.append(d.detach().clone().requires_grad_())
        # Save ranges to clip (aka. PGD)
        unstable_range_upper.append((left, right))

        self.unstable_upper = unstable_upper
        self.unstable_d_upper = unstable_d_upper
        self.unstable_range_upper = unstable_range_upper

    def upper_right_tangent_point(self, mask, lower, upper, mid, lower_inflection):
        over_inflection = lower >= lower_inflection[1]
        direct, indirect = mask & over_inflection, mask & (~over_inflection)
        over_inflection = over_inflection[mask]

        d_out = torch.zeros_like(lower[mask])
        d_out[over_inflection] = lower[direct]

        lower, upper = lower[indirect], upper[indirect]
        lower_act = self.func(lower, mask=indirect)

        def f(d: torch.Tensor) -> torch.Tensor:
            a_slope = (self.func(d, mask=indirect) - lower_act) / (d - lower)
            a_derivative = self.derivative(d, mask=indirect)
            return a_slope - a_derivative

        # Bisection will return left and right bounds for d s.t. f(d) is zero
        # Derivative of right bound will over-approximate the slope - hence a true bound
        bisection_lower = lower_inflection[1][indirect]
        bisection_upper = mid[indirect]
        _, d = bisection(bisection_lower, bisection_upper, f)
        d_out[~over_inflection] = d

        return d_out

    def upper_left_tangent_point(self, mask, lower, upper, mid, upper_inflection):
        under_inflection = upper <= upper_inflection[0]
        direct, indirect = mask & under_inflection, mask & (~under_inflection)
        under_inflection = under_inflection[mask]

        d_out = torch.zeros_like(lower[mask])
        d_out[under_inflection] = upper[direct]

        lower, upper = lower[indirect], upper[indirect]
        upper_act = self.func(upper, mask=indirect)

        def f(d: torch.Tensor) -> torch.Tensor:
            a_slope = (upper_act - self.func(d, mask=indirect)) / (upper - d)
            a_derivative = self.derivative(d, mask=indirect)
            return a_slope - a_derivative

        # Bisection will return left and right bounds for d s.t. f(d) is zero
        # Derivative of left bound will over-approximate the slope - hence a true bound
        bisection_lower = mid[indirect]
        bisection_upper = upper_inflection[0][indirect]
        d, _ = bisection(bisection_lower, bisection_upper, f)
        d_out[~under_inflection] = d

        return d_out

    def lower_left_tangent_point(self, mask, lower, upper, lower_inflection):
        under_inflection = upper <= lower_inflection[0]
        direct, indirect = mask & under_inflection, mask & (~under_inflection)
        under_inflection = under_inflection[mask]

        d_out = torch.zeros_like(lower[mask])
        d_out[under_inflection] = upper[direct]

        lower, upper = lower[indirect], upper[indirect]
        upper_act = self.func(upper, mask=indirect)

        def f(d: torch.Tensor) -> torch.Tensor:
            a_slope = (upper_act - self.func(d, mask=indirect)) / (upper - d)
            a_derivative = self.derivative(d, mask=indirect)
            return a_derivative - a_slope

        # Bisection will return left and right bounds for d s.t. f(d) is zero
        # Derivative of left bound will over-approximate the slope - hence a true bound
        bisection_lower = lower
        bisection_upper = lower_inflection[0][indirect]
        d, _ = bisection(bisection_lower, bisection_upper, f)
        d_out[~under_inflection] = d

        return d_out

    def lower_right_tangent_point(self, mask, lower, upper, upper_inflection):
        over_inflection = lower >= upper_inflection[1]
        direct, indirect = mask & over_inflection, mask & (~over_inflection)
        over_inflection = over_inflection[mask]

        d_out = torch.zeros_like(lower[mask])
        d_out[over_inflection] = lower[direct]

        lower, upper = lower[indirect], upper[indirect]
        lower_act = self.func(lower, mask=indirect)

        def f(d: torch.Tensor) -> torch.Tensor:
            a_slope = (self.func(d, mask=indirect) - lower_act) / (d - lower)
            a_derivative = self.derivative(d, mask=indirect)
            return a_derivative - a_slope

        # Bisection will return left and right bounds for d s.t. f(d) is zero
        # Derivative of right bound will over-approximate the slope - hence a true bound
        bisection_lower = upper_inflection[1][indirect]
        bisection_upper = upper
        _, d = bisection(bisection_lower, bisection_upper, f)
        d_out[~over_inflection] = d

        return d_out

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        if save_input_bounds:
            self.input_bounds = bounds

        zero_width, l, u, lu = bell_curve_regimes(bounds.lower, bounds.upper, self.midpoint)

        lower = torch.zeros_like(zero_width, dtype=bounds.lower.dtype)
        upper = torch.zeros_like(zero_width, dtype=bounds.lower.dtype)

        lower_act = self(bounds.lower)
        upper_act = self(bounds.upper)

        lower[zero_width] = torch.min(lower_act[zero_width], upper_act[zero_width])
        upper[zero_width] = torch.max(lower_act[zero_width], upper_act[zero_width])

        lower[l] = lower_act[l]
        upper[l] = upper_act[l]

        lower[u] = upper_act[u]
        upper[u] = lower_act[u]

        lower[lu] = torch.min(lower_act[lu], upper_act[lu])
        upper[lu] = (self(torch.as_tensor(self.midpoint, device=lower.device, dtype=upper.dtype)) * torch.ones_like(lower_act))[lu]

        return IntervalBounds(bounds.region, lower, upper)

    def parameterize_alpha_beta(self, alpha_lower, alpha_upper, beta_lower, beta_upper):
        if self.unstable_lower is None or self.unstable_upper is None:
            logger.warning('Bell curve bound not parameterized but expected to')

        # Use implicit parameterization (i.e. store d [point where touching the curve], and not alpha)
        def add_linear(alpha, beta, mask, x):
            a = self.derivative(x, mask=mask)
            y = self.func(x, mask=mask)

            alpha[mask] = a
            beta[mask] = y - a * x

        add_linear(alpha_lower, beta_lower, mask=self.unstable_lower[0], x=self.unstable_d_lower[0])
        add_linear(alpha_lower, beta_lower, mask=self.unstable_lower[1], x=self.unstable_d_lower[1])

        add_linear(alpha_upper, beta_upper, mask=self.unstable_upper[0], x=self.unstable_d_upper[0])
        add_linear(alpha_upper, beta_upper, mask=self.unstable_upper[1], x=self.unstable_d_upper[1])
        add_linear(alpha_upper, beta_upper, mask=self.unstable_upper[2], x=self.unstable_d_upper[2])

        return alpha_lower, alpha_upper, beta_lower, beta_upper

    def bound_parameters(self):
        if self.unstable_lower is None or self.unstable_upper is None:
            logger.warning('Bell curve bound not parameterized but expected to')

        yield from self.unstable_d_lower
        yield from self.unstable_d_upper

    def clip_params(self):
        clip_param_to_range_(self.unstable_d_lower[0], self.unstable_range_lower[0])
        clip_param_to_range_(self.unstable_d_lower[1], self.unstable_range_lower[1])

        clip_param_to_range_(self.unstable_d_upper[0], self.unstable_range_upper[0])
        clip_param_to_range_(self.unstable_d_upper[1], self.unstable_range_upper[1])
        clip_param_to_range_(self.unstable_d_upper[2], self.unstable_range_upper[2])

    def project_grads(self):
        proj_grad_to_range_(self.unstable_d_lower[0], self.unstable_range_lower[0])
        proj_grad_to_range_(self.unstable_d_lower[1], self.unstable_range_lower[1])

        proj_grad_to_range_(self.unstable_d_upper[0], self.unstable_range_upper[0])
        proj_grad_to_range_(self.unstable_d_upper[1], self.unstable_range_upper[1])
        proj_grad_to_range_(self.unstable_d_upper[2], self.unstable_range_upper[2])


class BoundStandardNormalPDF(BoundBellCurve):
    midpoint = 0.0
    lower_infliction = (-1.0, -1.0)
    upper_infliction = (1.0, 1.0)

    def derivative(self, x):
        return (1 / np.sqrt(2 * np.pi)) * (-torch.exp(-0.5 * x.pow(2)) * x)


class StandardNormalCDF(nn.Sequential):
    def __init__(self):
        super().__init__(
            ElementWiseLinear(1 / np.sqrt(2)),
            Erf(),
            ElementWiseLinear(0.5, 0.5)
        )


class NormalPDF(nn.Sequential):
    def __init__(self, loc, scale):
        inv_scale = 1 / scale

        super().__init__(
            ElementWiseLinear(inv_scale, -loc * inv_scale),
            StandardNormalPDF(),
            ElementWiseLinear(inv_scale)
        )


class NormalCDF(nn.Sequential):
    def __init__(self, loc, scale):
        inv_scale = 1 / (scale * np.sqrt(2))

        super().__init__(
            ElementWiseLinear(inv_scale, -loc * inv_scale),
            Erf(),
            ElementWiseLinear(0.5, 0.5)
        )


class TruncatedGaussianTwoSidedExpectation(nn.Sequential):
    def __init__(self, loc, scale, epsilon=1e-8):
        inv_scale = 1 / scale

        super().__init__(
            ElementWiseLinear(inv_scale, -loc * inv_scale),
            Div(
                nn.Sequential(StandardNormalPDF(), VectorSub()),
                nn.Sequential(Flip(), StandardNormalCDF(), VectorSub(), Clamp(min=epsilon)),
            ),
            ElementWiseLinear(scale, loc)
        )


class TruncatedGaussianLowerTailExpectation(nn.Sequential):
    def __init__(self, loc, scale):
        inv_scale = 1 / scale

        super().__init__(
            ElementWiseLinear(inv_scale, -loc * inv_scale),
            Div(
                StandardNormalPDF(),
                nn.Sequential(StandardNormalCDF(), ElementWiseLinear(-1.0, 1.0)),
            ),
            ElementWiseLinear(scale, loc)
        )


class TruncatedGaussianUpperTailExpectation(nn.Sequential):
    def __init__(self, loc, scale):
        inv_scale = 1 / scale

        super().__init__(
            ElementWiseLinear(inv_scale, -loc * inv_scale),
            Div(
                StandardNormalPDF(),
                nn.Sequential(StandardNormalCDF())
            ),
            ElementWiseLinear(-scale, loc)
        )
