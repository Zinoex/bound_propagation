import abc
import logging
from functools import wraps
from typing import Tuple

import torch
from torch import nn
import numpy as np

from .general import BoundModule
from .bounds import LinearBounds, IntervalBounds
from .util import TensorFunction

logger = logging.getLogger(__name__)


def assert_bound_order(func, position=0, keyword='preactivation'):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(args) > position:
            bounds = args[position]
        else:
            bounds = kwargs[keyword]

        assert torch.isnan(bounds.lower).any() or torch.isnan(bounds.upper).any() or \
               torch.all(bounds.lower <= bounds.upper + 1e-6)

        return func(self, *args, **kwargs)

    return wrapper


@torch.jit.script
def crown_backward_act_jit(W_tilde: torch.Tensor, alpha: Tuple[torch.Tensor, torch.Tensor], beta: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    _lambda = torch.where(W_tilde < 0, alpha[0].unsqueeze(-2), alpha[1].unsqueeze(-2))
    _delta = torch.where(W_tilde < 0, beta[0].unsqueeze(-2), beta[1].unsqueeze(-2))

    bias = torch.sum(W_tilde * _delta, dim=-1)
    W_tilde = W_tilde * _lambda

    return W_tilde, bias


class BoundActivation(BoundModule, abc.ABC):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.input_bounds = None

        self.alpha_lower, self.beta_lower = None, None
        self.alpha_upper, self.beta_upper = None, None

        self.bounded = False
        self.size = None

    @abc.abstractmethod
    def alpha_beta(self, preactivation):
        raise NotImplementedError()

    @property
    def need_relaxation(self):
        return not self.bounded

    def set_relaxation(self, linear_bounds):
        interval_bounds = linear_bounds.concretize()
        interval_bounds = IntervalBounds(
            linear_bounds.region,
            torch.max(interval_bounds.lower, self.input_bounds.lower),
            torch.min(interval_bounds.upper, self.input_bounds.upper)
        )

        self.alpha_beta(preactivation=interval_bounds)
        self.bounded = True

    def backward_relaxation(self, region):
        assert self.size is not None

        linear_bounds = self.initial_linear_bounds(region, self.size)
        return linear_bounds, self

    def clear_relaxation(self):
        self.input_bounds = None

        self.alpha_lower, self.beta_lower = None, None
        self.alpha_upper, self.beta_upper = None, None

        self.bounded = False

    def crown_backward(self, linear_bounds, optimize):
        assert self.bounded

        alpha_lower, alpha_upper = self.alpha_lower.detach().clone(), self.alpha_upper.detach().clone()
        beta_lower, beta_upper = self.beta_lower.detach().clone(), self.beta_upper.detach().clone()

        if optimize:
            alpha_lower, alpha_upper, beta_lower, beta_upper = \
                self.parameterize_alpha_beta(alpha_lower, alpha_upper, beta_lower, beta_upper)

        # NOTE: The order of alpha and beta are deliberately reverse - this is not a mistake!
        if linear_bounds.lower is None:
            lower = None
        else:
            alpha = alpha_upper, alpha_lower
            beta = beta_upper, beta_lower

            lower = crown_backward_act_jit(linear_bounds.lower[0], alpha, beta)
            lower = (lower[0], lower[1] + linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            alpha = alpha_lower, alpha_upper
            beta = beta_lower, beta_upper
            upper = crown_backward_act_jit(linear_bounds.upper[0], alpha, beta)
            upper = (upper[0], upper[1] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        if save_input_bounds:
            self.input_bounds = bounds

        return IntervalBounds(bounds.region, self.module(bounds.lower), self.module(bounds.upper))

    def propagate_size(self, in_size):
        self.size = in_size
        return in_size

    @abc.abstractmethod
    def parameterize_alpha_beta(self, alpha_lower, alpha_upper, beta_lower, beta_upper):
        raise NotImplementedError()


def regimes(lower: torch.Tensor, upper: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zero_width = torch.isclose(lower, upper, rtol=0.0, atol=1e-8)
    n = (~zero_width) & (upper <= 0)
    p = (~zero_width) & (0 <= lower)
    np = (~zero_width) & (lower < 0) & (0 < upper)

    return zero_width, n, p, np


class BoundReLU(BoundActivation):
    def __init__(self, module, factory, adaptive_relu=True, **kwargs):
        super().__init__(module, factory)
        self.adaptive_relu = adaptive_relu

        self.unstable_lower, self.unstable_slope_lower = None, None

    def clear_relaxation(self):
        super().clear_relaxation()
        self.unstable_lower, self.unstable_slope_lower = None, None

    @assert_bound_order
    def alpha_beta(self, preactivation):
        """
        See the source below for information on what adaptive ReLU means and how regimes of the input
        are exploited to compute bounds.

        @misc{zhang2018efficient,
              title={Efficient Neural Network Robustness Certification with General Activation Functions},
              author={Huan Zhang and Tsui-Wei Weng and Pin-Yu Chen and Cho-Jui Hsieh and Luca Daniel},
              year={2018},
              eprint={1811.00866},
              archivePrefix={arXiv},
              primaryClass={cs.LG}
        }

        :param self:
        :param preactivation:
        """
        lower, upper = preactivation.lower, preactivation.upper
        zero_width, n, p, np = regimes(lower, upper)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, self(lower[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, self(upper[zero_width])

        self.alpha_lower[n], self.beta_lower[n] = 0, 0
        self.alpha_upper[n], self.beta_upper[n] = 0, 0

        self.alpha_lower[p], self.beta_lower[p] = 1, 0
        self.alpha_upper[p], self.beta_upper[p] = 1, 0

        lower, upper = lower[np], upper[np]

        z = upper / (upper - lower)
        if self.adaptive_relu:
            # Utilize that bool->float conversion is true=1 and false=0
            a = (upper >= torch.abs(lower)).to(lower.dtype)
        else:
            a = z

        self.alpha_lower[np], self.beta_lower[np] = a, 0
        self.alpha_upper[np], self.beta_upper[np] = z, -lower * z

        # Allow parameterization
        # Save mask
        self.unstable_lower = np
        # Optimization variables [0, 1] - detach, clone, and require grad to perform back prop and optimization
        # Be sure that slope != 0 as that would result in zero gradient
        self.unstable_slope_lower = z.detach().clone().requires_grad_()

    def parameterize_alpha_beta(self, alpha_lower, alpha_upper, beta_lower, beta_upper):
        if self.unstable_lower is not None:
            alpha_lower[self.unstable_lower] = self.unstable_slope_lower
        else:
            logger.warning('ReLU bound not parameterized but expected to')

        return alpha_lower, alpha_upper, beta_lower, beta_upper

    def bound_parameters(self):
        if self.unstable_lower is not None:
            yield self.unstable_slope_lower
        else:
            logger.warning('ReLU bound not parameterized but expected to')

    def clip_params(self):
        self.unstable_slope_lower.data.clamp_(min=0, max=1)


class BoundSigmoid(BoundActivation):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory)

        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    def clear_relaxation(self):
        super().clear_relaxation()
        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    def derivative(self, x):
        return torch.sigmoid(x) * (1 - torch.sigmoid(x))

    @assert_bound_order
    def alpha_beta(self, preactivation):
        """
            Function to compute upper and lower bounds for S-shaped activation functions.
            Assumptions:
            - :LB < :UB
            - :derivative is the derivative of :func
            - :func is S-/sigmoid shaped

            See the source below for information on how regimes of the input are exploited to compute bounds.

            @misc{zhang2018efficient,
                  title={Efficient Neural Network Robustness Certification with General Activation Functions},
                  author={Huan Zhang and Tsui-Wei Weng and Pin-Yu Chen and Cho-Jui Hsieh and Luca Daniel},
                  year={2018},
                  eprint={1811.00866},
                  archivePrefix={arXiv},
                  primaryClass={cs.LG}
            }

        :param self:
        :param preactivation:
        """
        lower, upper = preactivation.lower, preactivation.upper
        zero_width, n, p, np = regimes(lower, upper)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, self(lower[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, self(upper[zero_width])

        lower_act, upper_act = self(lower), self(upper)
        lower_prime, upper_prime = self.derivative(lower), self.derivative(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self(d)
        d_prime = self.derivative(d)

        slope = (upper_act - lower_act) / (upper - lower)

        def add_linear(alpha, beta, mask, a, x, y, a_mask=True):
            if a_mask:
                a = a[mask]

            alpha[mask] = a
            beta[mask] = y[mask] - a * x[mask]

        ###################
        # Negative regime #
        ###################
        # Upper bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_upper, self.beta_upper, mask=n, a=slope, x=upper, y=upper_act)

        # Lower bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_lower, self.beta_lower, mask=n, a=d_prime, x=d, y=d_act)

        # Allow parameterization
        # Save mask
        self.unstable_lower = n
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        self.unstable_d_lower = d[n].detach().clone().requires_grad_()
        # Save ranges to clip (aka. PGD)
        self.unstable_range_lower = lower[n], upper[n]

        ###################
        # Positive regime #
        ###################
        # Lower bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_lower, self.beta_lower, mask=p, a=slope, x=lower, y=lower_act)

        # Upper bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_upper, self.beta_upper, mask=p, a=d_prime, x=d, y=d_act)

        # Allow parameterization
        # Save mask
        self.unstable_upper = p
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        self.unstable_d_upper = d[p].detach().clone().requires_grad_()
        # Save ranges to clip (aka. PGD)
        self.unstable_range_upper = lower[p], upper[p]

        #################
        # Crossing zero #
        #################
        # Upper bound #
        # If tangent to upper is below lower, then take direct slope between lower and upper
        direct = np & (slope <= upper_prime)
        add_linear(self.alpha_upper, self.beta_upper, mask=direct, a=slope, x=lower, y=lower_act)

        # Else use bisection to find upper bound on slope.
        implicit = np & (slope > upper_prime)

        if torch.any(implicit):
            implicit_lower, implicit_upper = lower[implicit], upper[implicit]
            implicit_lower_act = self(implicit_lower)

            def f_upper(d: torch.Tensor) -> torch.Tensor:
                a_slope = (self(d) - implicit_lower_act) / (d - implicit_lower)
                a_derivative = self.derivative(d)
                return a_slope - a_derivative

            # Bisection will return left and right bounds for d s.t. f_upper(d) is zero
            # Derivative of left bound will over-approximate the slope - hence a true bound
            d_upper, _ = bisection(torch.zeros_like(implicit_upper), implicit_upper, f_upper)
            # Slope has to attach to (lower, sigma(lower))
            add_linear(self.alpha_upper, self.beta_upper, mask=implicit, a=self.derivative(d_upper), x=lower, y=lower_act, a_mask=False)

        # Lower bound #
        # If tangent to lower is above upper, then take direct slope between lower and upper
        direct = np & (slope <= lower_prime)
        add_linear(self.alpha_lower, self.beta_lower, mask=direct, a=slope, x=upper, y=upper_act)

        # Else use bisection to find upper bound on slope.
        implicit = np & (slope > lower_prime)

        if torch.any(implicit):
            implicit_lower, implicit_upper = lower[implicit], upper[implicit]
            implicit_upper_act = self(implicit_upper)

            def f_lower(d: torch.Tensor) -> torch.Tensor:
                a_slope = (implicit_upper_act - self(d)) / (implicit_upper - d)
                a_derivative = self.derivative(d)
                return a_derivative - a_slope

            # Bisection will return left and right bounds for d s.t. f_lower(d) is zero
            # Derivative of right bound will over-approximate the slope - hence a true bound
            _, d_lower = bisection(implicit_lower, torch.zeros_like(implicit_lower), f_lower)
            # Slope has to attach to (upper, sigma(upper))
            add_linear(self.alpha_lower, self.beta_lower, mask=implicit, a=self.derivative(d_lower), x=upper, y=upper_act, a_mask=False)

    def parameterize_alpha_beta(self, alpha_lower, alpha_upper, beta_lower, beta_upper):
        if self.unstable_lower is None or self.unstable_upper is None:
            logger.warning('Sigmoid/tanh bound not parameterized but expected to')

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
            logger.warning('Sigmoid/tanh bound not parameterized but expected to')

        yield self.unstable_d_lower
        yield self.unstable_d_upper

    def clip_params(self):
        self.unstable_d_lower.data.clamp_(min=self.unstable_range_lower[0], max=self.unstable_range_lower[1])
        self.unstable_d_upper.data.clamp_(min=self.unstable_range_upper[0], max=self.unstable_range_upper[1])


def bisection(l: torch.Tensor, u: torch.Tensor, f: TensorFunction, num_iter: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    l, u = l.detach().clone(), u.detach().clone()
    midpoint = (l + u) / 2

    for _ in range(num_iter):
        y = f(midpoint)

        msk = y <= 0
        l[msk] = midpoint[msk]
        u[~msk] = midpoint[~msk]

        midpoint = (l + u) / 2

    return l, u


class BoundTanh(BoundSigmoid):
    # Tanh is actually just a scaled sigmoid, so let's reuse that code
    def derivative(self, x):
        return 1 - torch.tanh(x) ** 2


class Erf(nn.Module):
    def forward(self, x):
        return torch.special.erf(x)


class BoundErf(BoundSigmoid):
    def derivative(self, x):
        return (2 / np.sqrt(np.pi)) * torch.exp(-x.pow(2))


class BoundIdentity(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds, optimize):
        return linear_bounds

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        return bounds

    def propagate_size(self, in_size):
        return in_size


class Exp(nn.Module):
    def forward(self, x):
        return x.exp()


class BoundExp(BoundActivation):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory)

        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None

    def clear_relaxation(self):
        super().clear_relaxation()
        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None

    def derivative(self, x):
        return x.exp()

    @assert_bound_order
    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower, preactivation.upper
        zero_width, n, p, np = regimes(lower, upper)
        all = (n | p | np)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, self(lower[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, self(upper[zero_width])

        lower_act, upper_act = self(lower), self(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self(d)
        d_prime = self.derivative(d)

        slope = (upper_act - lower_act) / (upper - lower)

        def add_linear(alpha, beta, a, x, y, a_mask=True):
            if a_mask:
                a = a[all]

            alpha[all] = a
            beta[all] = y[all] - a * x[all]

        # Upper bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_upper, self.beta_upper, a=slope, x=lower, y=lower_act)

        # Lower bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_lower, self.beta_lower, a=d_prime, x=d, y=d_act)

        # Allow parameterization
        # Save mask
        self.unstable_lower = all
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        self.unstable_d_lower = d[all].detach().clone().requires_grad_()
        # Save ranges to clip (aka. PGD)
        self.unstable_range_lower = lower[all], upper[all]

    def parameterize_alpha_beta(self, alpha_lower, alpha_upper, beta_lower, beta_upper):
        if self.unstable_lower is None:
            logger.warning('Exp not parameterized but expected to')

        # Use implicit parameterization (i.e. store d [point where touching the curve], and not alpha)
        a = self.derivative(self.unstable_d_lower)
        y = self(self.unstable_d_lower)

        alpha_lower[self.unstable_lower] = a
        beta_lower[self.unstable_lower] = y - a * self.unstable_d_lower

        return alpha_lower, alpha_upper, beta_lower, beta_upper

    def bound_parameters(self):
        if self.unstable_lower is None:
            logger.warning('Exp bound not parameterized but expected to')

        yield self.unstable_d_lower

    def clip_params(self):
        self.unstable_d_lower.data.clamp_(min=self.unstable_range_lower[0], max=self.unstable_range_lower[1])


class Log(nn.Module):
    def forward(self, x):
        # Assumption: x > 0 (as log is only defined for x > 0)
        return x.log()


class BoundLog(BoundActivation):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory)

        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    def clear_relaxation(self):
        super().clear_relaxation()
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    def derivative(self, x):
        # As we already assume x > 0, this is well-defined
        return 1 / x

    @assert_bound_order
    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower, preactivation.upper
        assert torch.all(lower > 0)

        zero_width, n, p, np = regimes(lower, upper)
        all = (n | p | np)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, self(lower[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, self(upper[zero_width])

        lower_act, upper_act = self(lower), self(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self(d)
        d_prime = self.derivative(d)

        slope = (upper_act - lower_act) / (upper - lower)

        def add_linear(alpha, beta, a, x, y, a_mask=True):
            if a_mask:
                a = a[all]

            alpha[all] = a
            beta[all] = y[all] - a * x[all]

        # Lower bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_lower, self.beta_lower, a=slope, x=lower, y=lower_act)

        # Upper bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_upper, self.beta_upper, a=d_prime, x=d, y=d_act)

        # Allow parameterization
        # Save mask
        self.unstable_upper = all
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        self.unstable_d_upper = d[all].detach().clone().requires_grad_()
        # Save ranges to clip (aka. PGD)
        self.unstable_range_upper = lower[all], upper[all]

    def parameterize_alpha_beta(self, alpha_lower, alpha_upper, beta_lower, beta_upper):
        if self.unstable_upper is None:
            logger.warning('Log not parameterized but expected to')

        # Use implicit parameterization (i.e. store d [point where touching the curve], and not alpha)
        a = self.derivative(self.unstable_d_upper)
        y = self(self.unstable_d_upper)

        alpha_upper[self.unstable_upper] = a
        beta_upper[self.unstable_upper] = y - a * self.unstable_d_upper

        return alpha_lower, alpha_upper, beta_lower, beta_upper

    def bound_parameters(self):
        if self.unstable_upper is None:
            logger.warning('Log bound not parameterized but expected to')

        yield self.unstable_d_upper

    def clip_params(self):
        self.unstable_d_upper.data.clamp_(min=self.unstable_range_upper[0], max=self.unstable_range_upper[1])


class Reciprocal(nn.Module):
    def forward(self, x):
        return 1 / x


class BoundReciprocal(BoundActivation):
    # Only negative or positive. If crossing zero then no linear relaxation can exist as it goes to +- \infty

    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory)

        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    def clear_relaxation(self):
        super().clear_relaxation()
        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    def derivative(self, x):
        return -1 / (x ** 2)

    @assert_bound_order
    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower, preactivation.upper
        zero_width, n, p, np = regimes(lower, upper)
        assert not torch.any(np), 'Input to reciprocal cannot cross zero as no linear bound can exist'

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, self(lower[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, self(upper[zero_width])

        lower_act, upper_act = self(lower), self(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self(d)
        d_prime = self.derivative(d)

        slope = (upper_act - lower_act) / (upper - lower)

        def add_linear(alpha, beta, mask, a, x, y, a_mask=True):
            if a_mask:
                a = a[mask]

            alpha[mask] = a
            beta[mask] = y[mask] - a * x[mask]

        ###################
        # Negative regime #
        ###################
        # Lower bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_lower, self.beta_lower, mask=n, a=slope, x=upper, y=upper_act)

        # Upper bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_upper, self.alpha_upper, mask=n, a=d_prime, x=d, y=d_act)

        # Allow parameterization
        # Save mask
        self.unstable_upper = n
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        self.unstable_d_upper = d[n].detach().clone().requires_grad_()
        # Save ranges to clip (aka. PGD)
        self.unstable_range_upper = lower[n], upper[n]

        ###################
        # Positive regime #
        ###################
        # Upper bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_upper, self.beta_upper, mask=p, a=slope, x=lower, y=lower_act)

        # Lower bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_lower, self.beta_lower, mask=p, a=d_prime, x=d, y=d_act)

        # Allow parameterization
        # Save mask
        self.unstable_lower = p
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        self.unstable_d_lower = d[p].detach().clone().requires_grad_()
        # Save ranges to clip (aka. PGD)
        self.unstable_range_lower = lower[p], upper[p]

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        if save_input_bounds:
            self.input_bounds = bounds

        lower_act = self.module(bounds.lower)
        upper_act = self.module(bounds.upper)

        lower = torch.min(lower_act, upper_act)
        upper = torch.max(lower_act, upper_act)

        return IntervalBounds(bounds.region, lower, upper)

    def parameterize_alpha_beta(self, alpha_lower, alpha_upper, beta_lower, beta_upper):
        if self.unstable_lower is None or self.unstable_upper is None:
            logger.warning('Reciprocal bound not parameterized but expected to')

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
            logger.warning('Reciprocal bound not parameterized but expected to')

        yield self.unstable_d_lower
        yield self.unstable_d_upper

    def clip_params(self):
        self.unstable_d_lower.data.clamp_(min=self.unstable_range_lower[0], max=self.unstable_range_lower[1])
        self.unstable_d_upper.data.clamp_(min=self.unstable_range_upper[0], max=self.unstable_range_upper[1])


class Sin(nn.Module):
    def forward(self, x):
        return x.sin()


def sine_like_regimes(lower, upper, period, zero_increasing):
    zero_width = torch.isclose(lower, upper, rtol=0.0, atol=1e-8)

    half_period = upper - lower >= period / 2

    zero_shifted_lower = lower - zero_increasing
    zero_shifted_upper = upper - zero_increasing

    shift = torch.div(zero_shifted_lower, period, rounding_mode='floor') * period
    shifted_lower = zero_shifted_lower - shift
    shifted_upper = zero_shifted_upper - shift

    increasing_region1 = (shifted_upper <= (1 / 4) * period)
    increasing_region2 = (shifted_lower >= (3 / 4) * period) & (shifted_upper <= (5 / 4) * period)
    increasing_region3 = (shifted_lower >= (7 / 4) * period)
    increasing = (~zero_width) & (~half_period) & (increasing_region1 | increasing_region2 | increasing_region3)

    increasing_region1 = (shifted_lower >= (3 / 4) * period) & (shifted_upper <= period)
    increasing_region2 = (shifted_lower >= (7 / 4) * period)
    increasing_lower_curve = increasing & (increasing_region1 | increasing_region2)

    increasing_region1 = (shifted_upper <= (1 / 4) * period)
    increasing_region2 = (shifted_lower >= period) & (shifted_upper <= (5 / 4) * period)
    increasing_upper_curve = increasing & (increasing_region1 | increasing_region2)

    increasing_full_region = increasing & (~increasing_lower_curve) & (~increasing_upper_curve)
    increasing = increasing, (increasing_lower_curve, increasing_upper_curve, increasing_full_region)

    decreasing_region1 = (shifted_lower >= (1 / 4) * period) & (shifted_upper <= (3 / 4) * period)
    decreasing_region2 = (shifted_lower >= (5 / 4) * period) & (shifted_upper <= (7 / 4) * period)
    decreasing = (~zero_width) & (~half_period) & (decreasing_region1 | decreasing_region2)

    decreasing_region1 = (shifted_lower >= (2 / 4) * period) & (shifted_upper <= (3 / 4) * period)
    decreasing_region2 = (shifted_lower >= (6 / 4) * period) & (shifted_upper <= (7 / 4) * period)
    decreasing_lower_curve = decreasing & (decreasing_region1 | decreasing_region2)

    decreasing_region1 = (shifted_lower >= (1 / 4) * period) & (shifted_upper <= (2 / 4) * period)
    decreasing_region2 = (shifted_lower >= (5 / 4) * period) & (shifted_upper <= (6 / 4) * period)
    decreasing_upper_curve = decreasing & (decreasing_region1 | decreasing_region2)

    decreasing_full_region = decreasing & (~decreasing_lower_curve) & (~decreasing_upper_curve)
    decreasing = decreasing, (decreasing_lower_curve, decreasing_upper_curve, decreasing_full_region)

    crossing_peak_region1 = (shifted_lower < (1 / 4) * period) & (shifted_upper > (1 / 4) * period)
    crossing_peak_region2 = (shifted_lower < (5 / 4) * period) & (shifted_upper > (5 / 4) * period)
    crossing_peak = (~zero_width) & (~half_period) & (crossing_peak_region1 | crossing_peak_region2)

    crossing_trough_region1 = (shifted_lower < (3 / 4) * period) & (shifted_upper > (3 / 4) * period)
    crossing_trough_region2 = (shifted_lower < (7 / 4) * period) & (shifted_upper > (7 / 4) * period)
    crossing_trough = (~zero_width) & (~half_period) & (crossing_trough_region1 | crossing_trough_region2)

    return zero_width, half_period, increasing, decreasing, crossing_peak, crossing_trough


class BoundSin(BoundActivation):
    period = 2 * np.pi
    zero_increasing = 0

    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory)

        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    def clear_relaxation(self):
        super().clear_relaxation()
        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    def derivative(self, x):
        return x.cos()

    @assert_bound_order
    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower, preactivation.upper

        zero_width, half_period, (_, increasing), (_, decreasing), crossing_peak, crossing_trough = \
            sine_like_regimes(lower, upper, period=self.period, zero_increasing=self.zero_increasing)
        increasing_lower_curve, increasing_upper_curve, increasing_full_region = increasing
        decreasing_lower_curve, decreasing_upper_curve, decreasing_full_region = decreasing

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, self(lower[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, self(upper[zero_width])

        lower_act, upper_act = self(lower), self(upper)
        lower_prime, upper_prime = self.derivative(lower), self.derivative(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self(d)
        d_prime = self.derivative(d)

        slope = (upper_act - lower_act) / (upper - lower)

        ones = torch.ones_like(lower)
        zeros = torch.zeros_like(lower)

        def add_linear(alpha, beta, mask, a, x, y, a_mask=True):
            if a_mask:
                a = a[mask]

            alpha[mask] = a
            beta[mask] = y[mask] - a * x[mask]

        ##################
        # >= Half period #
        ##################
        # Lower bound
        # - Flat line = -1
        add_linear(self.alpha_lower, self.beta_lower, mask=half_period, a=zeros, x=zeros, y=-ones)

        # Upper bound
        # - Flat line = +1
        add_linear(self.alpha_upper, self.beta_upper, mask=half_period, a=zeros, x=zeros, y=ones)

        ###############
        # Lower curve #
        ###############
        lower_curve = increasing_lower_curve | decreasing_lower_curve | crossing_trough

        # Upper bound
        # - Exact slope between lower and upper
        add_linear(self.alpha_upper, self.beta_upper, mask=lower_curve, a=slope, x=lower, y=lower_act)

        # Lower bound
        # - d = (lower + upper) / 2 for midpoint
        # - Slope is sigma'(d) and it has to cross through sigma(d)
        add_linear(self.alpha_lower, self.beta_lower, mask=lower_curve, a=d_prime, x=d, y=d_act)

        # Allow parameterization
        # Save mask
        self.unstable_lower = lower_curve
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        self.unstable_d_lower = d[lower_curve].detach().clone().requires_grad_()
        # Save ranges to clip (aka. PGD)
        self.unstable_range_lower = lower[lower_curve], upper[lower_curve]

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


class Cos(nn.Module):
    def forward(self, x):
        return x.cos()


class BoundCos(BoundSin):
    period = 2 * np.pi
    zero_increasing = -np.pi / 2

    def derivative(self, x):
        return -x.sin()
