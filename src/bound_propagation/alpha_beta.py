from functools import wraps
from typing import Tuple, List, Optional, Callable

import torch
from torch import nn

from .util import LayerBounds, add_method, AlphaBetas, AlphaBeta, TensorFunction, LayerBound


def alpha_beta(class_or_obj):
    types = {
        nn.Sequential: alpha_beta_sequential,
        nn.Linear: alpha_beta_linear,
        nn.ReLU: alpha_beta_relu,
        nn.Sigmoid: alpha_beta_sigmoid,
        nn.Tanh: alpha_beta_tanh
    }

    for layer_type, alpha_beta_fun in types.items():
        if isinstance(class_or_obj, layer_type) or \
                (isinstance(class_or_obj, type) and issubclass(class_or_obj, layer_type)):
            return alpha_beta_fun(class_or_obj)

    raise NotImplementedError('Selected type of layer not supported')


def add_alpha_beta_submodules(model: nn.Sequential):
    for module in model:
        if not hasattr(module, 'alpha_beta'):
            # Decorator also adds the method inplace.
            alpha_beta(module)


def alpha_beta_sequential(class_or_obj):
    def _alpha_beta(self: nn.Sequential, layer_bounds: LayerBounds, **kwargs) -> AlphaBetas:
        add_alpha_beta_submodules(self)

        alpha_betas = []

        for module, layer_bound in zip(self, layer_bounds):
            ab = module.alpha_beta(layer_bound, **kwargs)
            alpha_betas.append(ab)

        return alpha_betas

    add_method(class_or_obj, 'alpha_beta', _alpha_beta)
    return class_or_obj


def regimes(LB: torch.Tensor, UB: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = UB <= 0
    p = 0 <= LB
    np = (LB < 0) & (0 < UB)

    return n, p, np


def assert_bound_order(func):
    @wraps(func)
    def wrapper(self, layer_bound: LayerBound, **kwargs):
        LB, UB = layer_bound
        assert torch.all(LB <= UB + 1e-6)

        return func(self, layer_bound, **kwargs)

    return wrapper


def alpha_beta_relu(class_or_obj):
    @assert_bound_order
    def alpha_beta(self: nn.ReLU, layer_bound: LayerBound, adaptive_relu: bool = True, **kwargs) -> AlphaBeta:
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
        :param layer_bound:
        :param adaptive_relu:
        :param kwargs:
        :return:
        """
        LB, UB = layer_bound
        n, p, np = regimes(LB, UB)

        alpha_lower_k = torch.zeros_like(LB)
        alpha_upper_k = torch.zeros_like(LB)
        beta_lower_k = torch.zeros_like(LB)
        beta_upper_k = torch.zeros_like(LB)

        alpha_lower_k[n] = 0
        alpha_upper_k[n] = 0
        beta_lower_k[n] = 0
        beta_upper_k[n] = 0

        alpha_lower_k[p] = 1
        alpha_upper_k[p] = 1
        beta_lower_k[p] = 0
        beta_upper_k[p] = 0

        LB, UB = LB[np], UB[np]

        z = UB / (UB - LB)
        if adaptive_relu:
            # Utilize that bool->float conversion is true=1 and false=0
            a = (UB >= torch.abs(LB)).to(torch.float)
        else:
            a = z

        alpha_lower_k[np] = a
        alpha_upper_k[np] = z
        beta_lower_k[np] = 0
        beta_upper_k[np] = -LB * z

        return (alpha_lower_k, alpha_upper_k), (beta_lower_k, beta_upper_k)

    add_method(class_or_obj, 'alpha_beta', alpha_beta)
    return class_or_obj


def alpha_beta_sigmoid(class_or_obj):
    def derivative(d):
        return torch.sigmoid(d) * (1 - torch.sigmoid(d))

    @assert_bound_order
    def alpha_beta(self: nn.Sigmoid, layer_bound: LayerBound, **kwargs) -> AlphaBeta:
        return alpha_beta_general(layer_bound, torch.sigmoid, derivative)

    add_method(class_or_obj, 'alpha_beta', alpha_beta)
    return class_or_obj


def alpha_beta_tanh(class_or_obj):
    def derivative(d):
        return 1 - torch.tanh(d) ** 2

    @assert_bound_order
    def alpha_beta(self: nn.Tanh, layer_bound: LayerBound, **kwargs) -> AlphaBeta:
        return alpha_beta_general(layer_bound, torch.tanh, derivative)

    add_method(class_or_obj, 'alpha_beta', alpha_beta)
    return class_or_obj


def alpha_beta_general(layer_bound: LayerBound, func: TensorFunction, derivative: TensorFunction) -> AlphaBeta:
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

    :param layer_bound:
    :param func:
    :param derivative:
    :return:
    """
    LB, UB = layer_bound
    n, p, np = regimes(LB, UB)

    alpha_lower = torch.zeros_like(LB)
    alpha_upper = torch.zeros_like(LB)
    beta_lower = torch.zeros_like(LB)
    beta_upper = torch.zeros_like(LB)

    LB_act, UB_act = func(LB), func(UB)
    LB_prime, UB_prime = derivative(LB), derivative(UB)

    d = (LB + UB) * 0.5  # Let d be the midpoint of the two bounds
    d_act = func(d)
    d_prime = derivative(d)

    slope = (UB_act - LB_act) / (UB - LB)

    def add_linear(alpha, beta, mask, a, x, y, a_mask=True):
        if a_mask:
            a = a[mask]

        alpha[mask] = a
        beta[mask] = y[mask] - a * x[mask]

    ###################
    # Negative regime #
    ###################
    # Upper bound
    # - Exact slope between LB and UB
    add_linear(alpha_upper, beta_upper, mask=n, a=slope, x=UB, y=UB_act)

    # Lower bound
    # - d = (LB + UB) / 2 for midpoint
    # - Slope is sigma'(d) and it has to cross through sigma(d)
    add_linear(alpha_lower, beta_lower, mask=n, a=d_prime, x=d, y=d_act)

    ###################
    # Positive regime #
    ###################
    # Lower bound
    # - Exact slope between LB and UB
    add_linear(alpha_lower, beta_lower, mask=p, a=slope, x=LB, y=LB_act)

    # Upper bound
    # - d = (LB + UB) / 2 for midpoint
    # - Slope is sigma'(d) and it has to cross through sigma(d)
    add_linear(alpha_upper, beta_upper, mask=p, a=d_prime, x=d, y=d_act)

    #################
    # Crossing zero #
    #################
    # Upper bound #
    # If tangent to UB is below LB, then take direct slope between LB and UB
    direct_upper = np & (slope <= UB_prime)
    add_linear(alpha_upper, beta_upper, mask=direct_upper, a=slope, x=LB, y=LB_act)

    # Else use bisection to find upper bound on slope.
    implicit_upper = np & (slope > UB_prime)

    def f_upper(d: torch.Tensor) -> torch.Tensor:
        a_slope = (func(d) - func(LB[implicit_upper])) / (d - LB[implicit_upper])
        a_derivative = derivative(d)
        return a_slope - a_derivative

    # Bisection will return left and right bounds for d s.t. f_upper(d) is zero
    # Derivative of left bound will overapproximate the slope - hence a true bound
    d_upper, _ = bisection(torch.zeros_like(UB[implicit_upper]), UB[implicit_upper], f_upper)
    # Slope has to attach to (LB, sigma(LB))
    add_linear(alpha_upper, beta_upper, mask=implicit_upper, a=derivative(d_upper), x=LB, y=LB_act, a_mask=False)

    # Lower bound #
    # If tangent to LB is above UB, then take direct slope between LB and UB
    direct_lower = np & (slope <= LB_prime)
    add_linear(alpha_lower, beta_lower, mask=direct_lower, a=slope, x=UB, y=UB_act)

    # Else use bisection to find upper bound on slope.
    implicit_lower = np & (slope > LB_prime)

    def f_lower(d: torch.Tensor) -> torch.Tensor:
        a_slope = (func(UB[implicit_lower]) - func(d)) / (UB[implicit_lower] - d)
        a_derivative = derivative(d)
        return a_derivative - a_slope

    # Bisection will return left and right bounds for d s.t. f_lower(d) is zero
    # Derivative of right bound will overapproximate the slope - hence a true bound
    _, d_lower = bisection(LB[implicit_lower], torch.zeros_like(LB[implicit_lower]), f_lower)
    # Slope has to attach to (UB, sigma(UB))
    add_linear(alpha_lower, beta_lower, mask=implicit_lower, a=derivative(d_lower), x=UB, y=UB_act, a_mask=False)

    return (alpha_lower, alpha_upper), (beta_lower, beta_upper)


def bisection(l: torch.Tensor, h: torch.Tensor, f: TensorFunction, num_iter: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    midpoint = (l + h) / 2

    for _ in range(num_iter):
        y = f(midpoint)

        l[y <= 0] = midpoint[y <= 0]
        h[y > 0] = midpoint[y > 0]

        midpoint = (l + h) / 2

    return l, h


def alpha_beta_linear(class_or_obj):
    def alpha_beta(self: nn.Linear, layer_bound: LayerBound, **kwargs) -> AlphaBeta:
        return (None, None), (None, None)

    add_method(class_or_obj, 'alpha_beta', alpha_beta)
    return class_or_obj
