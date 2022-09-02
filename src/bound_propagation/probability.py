import logging
from typing import Tuple

import numpy as np
import torch
from torch import nn

from .saturation import Clamp
from .bivariate import Div, VectorSub
from .reshape import Flip
from .polynomial import Pow
from .activation import BoundSigmoid, BoundActivation, assert_bound_order, Exp
from .linear import ElementWiseLinear

logger = logging.getLogger(__name__)


class Erf(nn.Module):
    def forward(self, x):
        return torch.special.erf(x)


class BoundErf(BoundSigmoid):
    def derivative(self, x):
        return (2 / np.sqrt(np.pi)) * torch.exp(-x.pow(2))


class StandardNormalPDF(nn.Sequential):
    def __init__(self):
        super().__init__(
            Pow(2),
            ElementWiseLinear(-0.5),
            Exp(),
            ElementWiseLinear(1 / np.sqrt(2 * np.pi))
        )


# class StandardNormalPDF(nn.Module):
#     def forward(self, x):
#         return (1 / np.sqrt(2 * np.pi)) * torch.exp(-0.5 * x.pow(2))


def standard_normal_regimes(lower: torch.Tensor, upper: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero_width = torch.isclose(lower, upper, rtol=0.0, atol=1e-8)
    ip = (~zero_width) & (-1 <= lower) & (upper <= 1)  # Infliction points
    n = (~zero_width) & (upper <= -1)
    p = (~zero_width) & (1 <= lower)

    nip = (~zero_width) & (lower < -1) & (upper <= 1)
    ipp = (~zero_width) & (-1 <= lower) & (1 < upper)
    full_width = (~zero_width) & (lower < -1) & (1 < upper)

    return zero_width, n, p, ip, nip, ipp, full_width


class BoundStandardNormalPDF(BoundActivation):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    def clear_relaxation(self):
        super().clear_relaxation()
        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    def derivative(self, x):
        return (1 / np.sqrt(2 * np.pi)) * (-torch.exp(-0.5 * x.pow(2)) * x)

    @assert_bound_order
    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower, preactivation.upper

        zero_width, n, p, ip, nip, ipp, full_width = standard_normal_regimes(lower, upper)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        lower_act, upper_act = self(lower), self(upper)
        lower_prime, upper_prime = self.derivative(lower), self.derivative(upper)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, torch.min(lower_act[zero_width], upper_act[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, torch.max(lower_act[zero_width], upper_act[zero_width])

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self(d)
        d_prime = self.derivative(d)

        slope = (upper_act - lower_act) / (upper - lower)

        def add_linear(alpha, beta, mask, a, x, y, a_mask=True):
            if a_mask:
                a = a[mask]

            alpha[mask] = a
            beta[mask] = y[mask] - a * x[mask]

        ###############
        # Upper curve #
        ###############
        upper_curve = increasing_upper_curve | decreasing_upper_curve | crossing_peak

        # Lower bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_lower, self.beta_lower, mask=upper_curve, a=slope, x=upper, y=upper_act)

        # Upper bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_upper, self.beta_upper, mask=upper_curve, a=d_prime, x=d, y=d_act)

        # Allow parameterization
        # Save mask
        self.unstable_upper = upper_curve
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        self.unstable_d_upper = d[upper_curve].detach().clone().requires_grad_()
        # Save ranges to clip (aka. PGD)
        self.unstable_range_upper = lower[upper_curve], upper[upper_curve]

        # ##########################
        # Increasing full region #
        ##########################
        # Upper bound #
        # If tangent to upper is below lower, then take direct slope between lower and upper
        direct = increasing_full_region & (slope <= upper_prime)
        add_linear(self.alpha_upper, self.beta_upper, mask=direct, a=slope, x=lower, y=lower_act)

        # Else use bisection to find upper bound on slope.
        implicit = increasing_full_region & (slope > upper_prime)

        if torch.any(implicit):
            implicit_lower, implicit_upper = lower[implicit], upper[implicit]
            implicit_lower_act = self(implicit_lower)

            def f_upper(d: torch.Tensor) -> torch.Tensor:
                a_slope = (self(d) - implicit_lower_act) / (d - implicit_lower)
                a_derivative = self.derivative(d)
                return a_slope - a_derivative

            # Bisection will return left and right bounds for d s.t. f_upper(d) is zero
            # Derivative of left bound will over-approximate the slope - hence a true bound
            bisection_lower = implicit_lower + self.period / 4 - torch.remainder(implicit_lower - self.zero_increasing, self.period / 4)
            d_upper, _ = bisection(bisection_lower, implicit_upper, f_upper)
            # Slope has to attach to (lower, sigma(lower))
            add_linear(self.alpha_upper, self.beta_upper, mask=implicit, a=self.derivative(d_upper), x=lower, y=lower_act, a_mask=False)

        # Lower bound #
        # If tangent to lower is above upper, then take direct slope between lower and upper
        direct = increasing_full_region & (slope <= lower_prime)
        add_linear(self.alpha_lower, self.beta_lower, mask=direct, a=slope, x=upper, y=upper_act)

        # Else use bisection to find upper bound on slope.
        implicit = increasing_full_region & (slope > lower_prime)

        if torch.any(implicit):
            implicit_lower, implicit_upper = lower[implicit], upper[implicit]
            implicit_upper_act = self(implicit_upper)

            def f_lower(d: torch.Tensor) -> torch.Tensor:
                a_slope = (implicit_upper_act - self(d)) / (implicit_upper - d)
                a_derivative = self.derivative(d)
                return a_derivative - a_slope

            # Bisection will return left and right bounds for d s.t. f_lower(d) is zero
            # Derivative of right bound will over-approximate the slope - hence a true bound
            bisection_upper = implicit_upper - torch.remainder(implicit_upper - self.zero_increasing, self.period / 4)
            _, d_lower = bisection(implicit_lower, bisection_upper, f_lower)
            # Slope has to attach to (upper, sigma(upper))
            add_linear(self.alpha_lower, self.beta_lower, mask=implicit, a=self.derivative(d_lower), x=upper, y=upper_act, a_mask=False)

        ##########################
        # Decreasing full region #
        ##########################
        # Upper bound #
        # If tangent to lower is below upper, then take direct slope between lower and upper
        direct = decreasing_full_region & (slope >= lower_prime)
        add_linear(self.alpha_upper, self.beta_upper, mask=direct, a=slope, x=lower, y=lower_act)

        # Else use bisection to find upper bound on slope.
        implicit = decreasing_full_region & (slope < lower_prime)

        if torch.any(implicit):
            implicit_lower, implicit_upper = lower[implicit], upper[implicit]
            implicit_lower_act = self(implicit_lower)

            def f_upper(d: torch.Tensor) -> torch.Tensor:
                a_slope = (implicit_lower_act - self(d)) / (implicit_lower - d)
                a_derivative = self.derivative(d)
                return a_derivative - a_slope

            # Bisection will return left and right bounds for d s.t. f_upper(d) is zero
            # Derivative of right bound will over-approximate the slope - hence a true bound
            bisection_upper = implicit_upper - torch.remainder(implicit_upper - self.zero_increasing, self.period / 4)
            _, d_upper = bisection(implicit_lower, bisection_upper, f_upper)
            # Slope has to attach to (lower, sigma(lower))
            add_linear(self.alpha_upper, self.beta_upper, mask=implicit, a=self.derivative(d_upper), x=upper, y=upper_act, a_mask=False)

        # Lower bound #
        # If tangent to upper is above lower, then take direct slope between lower and upper
        direct = decreasing_full_region & (slope >= upper_prime)
        add_linear(self.alpha_lower, self.beta_lower, mask=direct, a=slope, x=lower, y=lower_act)

        # Else use bisection to find upper bound on slope.
        implicit = decreasing_full_region & (slope < upper_prime)

        if torch.any(implicit):
            implicit_lower, implicit_upper = lower[implicit], upper[implicit]
            implicit_upper_act = self(implicit_upper)

            def f_lower(d: torch.Tensor) -> torch.Tensor:
                a_slope = (self(d) - implicit_upper_act) / (d - implicit_upper)
                a_derivative = self.derivative(d)
                return a_slope - a_derivative

            # Bisection will return left and right bounds for d s.t. f_lower(d) is zero
            # Derivative of left bound will over-approximate the slope - hence a true bound
            bisection_lower = implicit_lower + self.period / 4 - torch.remainder(implicit_lower - self.zero_increasing, self.period / 4)
            d_lower, _ = bisection(bisection_lower, implicit_upper, f_lower)
            # Slope has to attach to (upper, sigma(upper))
            add_linear(self.alpha_lower, self.beta_lower, mask=implicit, a=self.derivative(d_lower), x=lower, y=lower_act, a_mask=False)

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        if save_input_bounds:
            self.input_bounds = bounds

        zero_width, half_period, (increasing, _), (decreasing, _), crossing_peak, crossing_trough = \
            sine_like_regimes(bounds.lower, bounds.upper, period=self.period, zero_increasing=self.zero_increasing)

        lower = torch.zeros_like(bounds.lower)
        upper = torch.zeros_like(bounds.upper)

        lower_act = self.module(bounds.lower)
        upper_act = self.module(bounds.upper)

        lower[zero_width] = torch.min(lower_act[zero_width], upper_act[zero_width])
        upper[zero_width] = torch.max(lower_act[zero_width], upper_act[zero_width])

        lower[half_period] = -1
        upper[half_period] = 1

        lower[increasing] = lower_act[increasing]
        upper[increasing] = upper_act[increasing]

        lower[decreasing] = upper_act[decreasing]
        upper[decreasing] = lower_act[decreasing]

        lower[crossing_peak] = torch.min(lower_act[crossing_peak], upper_act[crossing_peak])
        upper[crossing_peak] = 1

        lower[crossing_trough] = -1
        upper[crossing_trough] = torch.max(lower_act[crossing_trough], upper_act[crossing_trough])

        return IntervalBounds(bounds.region, lower, upper)

    def parameterize_alpha_beta(self, alpha_lower, alpha_upper, beta_lower, beta_upper):
        if self.unstable_lower is None or self.unstable_upper is None:
            logger.warning('Sin/cos bound not parameterized but expected to')

        # Use implicit parameterization (i.e. store d [point where touching the curve], and not alpha)
        def add_linear(alpha, beta, mask, x):
            a = self.derivative(x)
            y = self(x)

            alpha[mask] = a
            beta[mask] = y - a * x

        add_linear(alpha_lower, beta_lower, mask=self.unstable_lower, x=self.unstable_d_lower)
        add_linear(alpha_upper, beta_upper, mask=self.unstable_upper, x=self.unstable_d_upper)

        return alpha_lower, alpha_upper, beta_lower, beta_upper

    def bound_parameters(self):
        if self.unstable_lower is None or self.unstable_upper is None:
            logger.warning('Sin/cos bound not parameterized but expected to')

        yield self.unstable_d_lower
        yield self.unstable_d_upper

    def clip_params(self):
        self.unstable_d_lower.data.clamp(min=self.unstable_range_lower[0], max=self.unstable_range_lower[1])
        self.unstable_d_upper.data.clamp(min=self.unstable_range_upper[0], max=self.unstable_range_upper[1])


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
