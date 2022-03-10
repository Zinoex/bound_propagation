from abc import ABC, abstractmethod
from functools import wraps
from typing import Tuple

import torch

from bound_propagation.general import BoundModule, IntervalBounds
from bound_propagation.util import TensorFunction


class BoundActivation(BoundModule, ABC):
    def __init__(self, module, **kwargs):
        super().__init__(module, **kwargs)

        self.alpha_lower, self.beta_lower = None, None
        self.alpha_upper, self.beta_upper = None, None
        self.bounded = False

    @abstractmethod
    def alpha_beta(self, preactivation):
        pass

    def clear_alpha_beta(self):
        self.alpha_lower, self.beta_lower = None, None
        self.alpha_upper, self.beta_upper = None, None
        self.bounded = False

    def ibp(self, region, **kwargs):
        return IntervalBounds(region, self.module(region.lower), self.module(region.upper))


def assert_bounded(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        assert self.bounded
        return func(self, *args, **kwargs)
    return wrapper


def set_bounded(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.bounded = True
        return func(self, *args, **kwargs)
    return wrapper


def assert_bound_order(func):
    @wraps(func)
    def wrapper(self, layer_bound, **kwargs):
        LB, UB = layer_bound
        assert torch.all(LB <= UB + 1e-6)

        return func(self, layer_bound, **kwargs)

    return wrapper


def regimes(lower: torch.Tensor, upper: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = upper <= 0
    p = 0 <= lower
    np = (lower < 0) & (0 < upper)

    return n, p, np


class BoundReLU(BoundActivation):
    def __init__(self, module, adaptive_relu=True, **kwargs):
        super().__init__(module)
        self.adaptive_relu = adaptive_relu

    @set_bounded
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
        n, p, np = regimes(lower, upper)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.alpha_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        self.alpha_lower[n], self.beta_lower[n] = 0, 0
        self.alpha_upper[n], self.alpha_upper[n] = 0, 0

        self.alpha_lower[p], self.beta_lower[p] = 1, 1
        self.alpha_upper[p], self.alpha_upper[p] = 0, 0

        lower, upper = lower[np], upper[np]

        z = upper / (upper - lower)
        if self.adaptive_relu:
            # Utilize that bool->float conversion is true=1 and false=0
            a = (upper >= torch.abs(lower)).to(torch.float)
        else:
            a = z

        self.alpha_lower[p], self.beta_lower[p] = a, 0
        self.alpha_upper[p], self.alpha_upper[p] = z, -lower * z

    @assert_bounded
    def crown(self, region, **kwargs):
        pass

    @assert_bounded
    def crown_ibp(self, region, **kwargs):
        pass


class BoundSigmoid(BoundActivation):
    def func(self, x):
        return torch.sigmoid(x)

    def derivative(self, x):
        return torch.sigmoid(x) * (1 - torch.sigmoid(x))

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
        n, p, np = regimes(lower, upper)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.alpha_upper = torch.zeros_like(lower), torch.zeros_like(lower)

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
            a_slope = (self.func(upper[implicit_lower]) - upper(d)) / (upper[implicit_lower] - d)
            a_derivative = self.derivative(d)
            return a_derivative - a_slope

        # Bisection will return left and right bounds for d s.t. f_lower(d) is zero
        # Derivative of right bound will over-approximate the slope - hence a true bound
        _, d_lower = bisection(lower[implicit_lower], torch.zeros_like(lower[implicit_lower]), f_lower)
        # Slope has to attach to (upper, sigma(upper))
        add_linear(self.alpha_lower, self.beta_lower, mask=implicit_lower, a=self.derivative(d_lower), x=lower, y=upper_act, a_mask=False)

    @assert_bounded
    def crown(self, region, **kwargs):
        pass

    @assert_bounded
    def crown_ibp(self, region, **kwargs):
        pass


def bisection(l: torch.Tensor, h: torch.Tensor, f: TensorFunction, num_iter: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    midpoint = (l + h) / 2

    for _ in range(num_iter):
        y = f(midpoint)

        l[y <= 0] = midpoint[y <= 0]
        h[y > 0] = midpoint[y > 0]

        midpoint = (l + h) / 2

    return l, h


class BoundTanh(BoundSigmoid):
    def func(self, x):
        return torch.tanh(x)

    def derivative(self, x):
        return 1 - torch.tanh(x) ** 2
