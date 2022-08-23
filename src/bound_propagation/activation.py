import abc
import logging
from functools import wraps
from typing import Tuple

import torch

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
        self.alpha_beta(preactivation=interval_bounds)
        self.bounded = True

    def backward_relaxation(self, region):
        assert self.size is not None

        linear_bounds = self.initial_linear_bounds(region, self.size)
        return linear_bounds, self

    def clear_relaxation(self):
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
    def ibp_forward(self, bounds, save_relaxation=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

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

        self.unstable_lower, self._unstable_slope_lower, self.initial_unstable_slope_lower = None, None, None

    @property
    def unstable_slope_lower(self):
        return self._unstable_slope_lower

    @unstable_slope_lower.setter
    def unstable_slope_lower(self, value):
        self._unstable_slope_lower = value
        self.initial_unstable_slope_lower = value

    def clear_relaxation(self):
        super().clear_relaxation()
        self.unstable_lower, self._unstable_slope_lower, self.initial_unstable_slope_lower = None, None, None

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

        self.unstable_lower = np
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

    def reset_params(self):
        self._unstable_slope_lower = self.initial_unstable_slope_lower

    def clip_params(self):
        self._unstable_slope_lower.data.clamp_(min=0, max=1)


class BoundSigmoid(BoundActivation):
    def __init__(self, module, factory, adaptive_relu=True, **kwargs):
        super().__init__(module, factory)
        self.adaptive_relu = adaptive_relu

        self.unstable_lower, self._unstable_d_lower, self.initial_unstable_d_lower = None, None, None
        self.unstable_upper, self._unstable_d_upper, self.initial_unstable_d_upper = None, None, None
        self.unstable_range_lower, self.unstable_range_upper = None, None

    @property
    def unstable_d_lower(self):
        return self._unstable_d_lower

    @unstable_d_lower.setter
    def unstable_d_lower(self, value):
        self._unstable_d_lower = value
        self.initial_unstable_d_lower = value

    @property
    def unstable_d_upper(self):
        return self._unstable_d_upper

    @unstable_d_upper.setter
    def unstable_d_upper(self, value):
        self._unstable_d_upper = value
        self.initial_unstable_d_upper = value

    def clear_relaxation(self):
        super().clear_relaxation()
        self.unstable_lower, self._unstable_d_lower, self.initial_unstable_d_lower = None, None, None
        self.unstable_upper, self._unstable_d_upper, self.initial_unstable_d_upper = None, None, None
        self.unstable_range_lower, self.unstable_range_upper = None, None

    def func(self, x):
        return torch.sigmoid(x)

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

        lower_act, upper_act = self.func(lower), self.func(upper)
        lower_prime, upper_prime = self.derivative(lower), self.derivative(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self.func(d)
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
        self.unstable_lower = n
        self.unstable_d_lower = d[n].detach().clone().requires_grad_()
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
        self.unstable_upper = p
        self.unstable_d_upper = d[p].detach().clone().requires_grad_()
        self.unstable_range_upper = lower[p], upper[p]

        #################
        # Crossing zero #
        #################
        # Upper bound #
        # If tangent to upper is below lower, then take direct slope between lower and upper
        direct_upper = np & (slope <= upper_prime)
        add_linear(self.alpha_upper, self.beta_upper, mask=direct_upper, a=slope, x=lower, y=lower_act)

        # Else use bisection to find upper bound on slope.
        implicit_upper = np & (slope > upper_prime)

        def f_upper(d: torch.Tensor) -> torch.Tensor:
            a_slope = (self.func(d) - self.func(lower[implicit_upper])) / (d - lower[implicit_upper])
            a_derivative = self.derivative(d)
            return a_slope - a_derivative

        # Bisection will return left and right bounds for d s.t. f_upper(d) is zero
        # Derivative of left bound will over-approximate the slope - hence a true bound
        d_upper, _ = bisection(torch.zeros_like(upper[implicit_upper]), upper[implicit_upper], f_upper)
        # Slope has to attach to (lower, sigma(lower))
        add_linear(self.alpha_upper, self.beta_upper, mask=implicit_upper, a=self.derivative(d_upper), x=lower, y=lower_act, a_mask=False)

        # Lower bound #
        # If tangent to lower is above upper, then take direct slope between lower and upper
        direct_lower = np & (slope <= lower_prime)
        add_linear(self.alpha_lower, self.beta_lower, mask=direct_lower, a=slope, x=upper, y=upper_act)

        # Else use bisection to find upper bound on slope.
        implicit_lower = np & (slope > lower_prime)

        def f_lower(d: torch.Tensor) -> torch.Tensor:
            a_slope = (self.func(upper[implicit_lower]) - self.func(d)) / (upper[implicit_lower] - d)
            a_derivative = self.derivative(d)
            return a_derivative - a_slope

        # Bisection will return left and right bounds for d s.t. f_lower(d) is zero
        # Derivative of right bound will over-approximate the slope - hence a true bound
        _, d_lower = bisection(lower[implicit_lower], torch.zeros_like(lower[implicit_lower]), f_lower)
        # Slope has to attach to (upper, sigma(upper))
        add_linear(self.alpha_lower, self.beta_lower, mask=implicit_lower, a=self.derivative(d_lower), x=upper, y=upper_act, a_mask=False)

    def parameterize_alpha_beta(self, alpha_lower, alpha_upper, beta_lower, beta_upper):
        if self.unstable_lower is None or self.unstable_upper is None:
            logger.warning('Sigmoid/tanh bound not parameterized but expected to')

        def add_linear(alpha, beta, mask, x):
            a = self.derivative(x)
            y = self.func(x)

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

    def reset_params(self):
        self._unstable_d_lower = self.initial_unstable_d_lower
        self._unstable_d_upper = self.initial_unstable_d_upper

    def clip_params(self):
        self._unstable_d_lower.data.clamp_(min=self.unstable_range_lower[0], max=self.unstable_range_lower[1])
        self._unstable_d_upper.data.clamp_(min=self.unstable_range_upper[0], max=self.unstable_range_upper[1])


def bisection(l: torch.Tensor, h: torch.Tensor, f: TensorFunction, num_iter: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    midpoint = (l + h) / 2

    for _ in range(num_iter):
        y = f(midpoint)

        msk = y <= 0
        l[msk] = midpoint[msk]
        h[~msk] = midpoint[~msk]

        midpoint = (l + h) / 2

    return l, h


class BoundTanh(BoundSigmoid):
    def func(self, x):
        return torch.tanh(x)

    def derivative(self, x):
        return 1 - torch.tanh(x) ** 2


class BoundIdentity(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds, optimize):
        return linear_bounds

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False):
        return bounds

    def propagate_size(self, in_size):
        return in_size
