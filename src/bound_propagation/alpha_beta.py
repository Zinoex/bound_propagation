from typing import Tuple, List, Optional, Callable

import torch
from torch import nn

from .util import LayerBounds, add_method, AlphaBetas, AlphaBeta, TensorFunction


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


def alpha_beta_sequential(class_or_obj):
    def _alpha_beta(model: nn.Sequential, layer_bounds: LayerBounds, **kwargs) -> AlphaBetas:
        alpha_betas = []

        for module, (LB, UB) in zip(model, layer_bounds):
            if not hasattr(module, 'alpha_beta'):
                # Decorator also adds the method inplace.
                alpha_beta(module)

            assert torch.all(LB <= UB + 1e-6)

            ab = module.alpha_beta(LB, UB, **kwargs)

            alpha_betas.append(ab)

        return alpha_betas

    add_method(class_or_obj, 'alpha_beta', _alpha_beta)
    return class_or_obj


def regimes(LB: torch.Tensor, UB: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = UB <= 0
    p = 0 <= LB
    np = (LB < 0) & (0 < UB)

    return n, p, np


def alpha_beta_relu(class_or_obj):
    def alpha_beta(module: nn.ReLU, LB: torch.Tensor, UB: torch.Tensor, adaptive_relu: bool = True, **kwargs) -> AlphaBeta:
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

        :param module:
        :param LB:
        :param UB:
        :param adaptive_relu:
        :param kwargs:
        :return:
        """

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

    def alpha_beta(module: nn.Sigmoid, LB: torch.Tensor, UB: torch.Tensor, **kwargs) -> AlphaBeta:
        return alpha_beta_general(LB, UB, torch.sigmoid, derivative)

    add_method(class_or_obj, 'alpha_beta', alpha_beta)
    return class_or_obj


def alpha_beta_tanh(class_or_obj):
    def derivative(d):
        return 1 - torch.tanh(d) ** 2

    def alpha_beta(module: nn.Tanh, LB: torch.Tensor, UB: torch.Tensor, **kwargs) -> AlphaBeta:
        return alpha_beta_general(LB, UB, torch.tanh, derivative)

    add_method(class_or_obj, 'alpha_beta', alpha_beta)
    return class_or_obj


def alpha_beta_general(LB: torch.Tensor, UB: torch.Tensor, func: TensorFunction, derivative: TensorFunction) -> AlphaBeta:
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

    :param LB:
    :param UB:
    :param func:
    :param derivative:
    :return:
    """

    n, p, np = regimes(LB, UB)

    alpha_lower_k = torch.zeros_like(LB)
    alpha_upper_k = torch.zeros_like(LB)
    beta_lower_k = torch.zeros_like(LB)
    beta_upper_k = torch.zeros_like(LB)

    LB_act, UB_act = func(LB), func(UB)
    LB_prime, UB_prime = derivative(LB), derivative(UB)

    d = (LB + UB) * 0.5  # Let d be the midpoint of the two bounds
    d_act = func(d)
    d_prime = derivative(d)

    slope = (UB_act - LB_act) / (UB - LB)

    ###################
    # Negative regime #
    ###################
    # Upper bound
    # - Exact slope between LB and UB
    alpha_upper_k[n] = slope[n]
    beta_upper_k[n] = UB_act[n] - alpha_upper_k[n] * UB[n]

    # Lower bound
    # - d = (LB + UB) / 2 for midpoint
    # - Slope is sigma'(d) and it has to cross through sigma(d)
    alpha_lower_k[n] = d_prime[n]
    beta_lower_k[n] = d_act[n] - alpha_lower_k[n] * d[n]

    ###################
    # Positive regime #
    ###################
    # Lower bound
    # - Exact slope between LB and UB
    alpha_lower_k[p] = slope[p]
    beta_lower_k[p] = LB_act[p] - alpha_lower_k[p] * LB[p]

    # Upper bound
    # - d = (LB + UB) / 2 for midpoint
    # - Slope is sigma'(d) and it has to cross through sigma(d)
    alpha_upper_k[p] = d_prime[p]
    beta_upper_k[p] = d_act[p] - alpha_upper_k[p] * d[p]

    #################
    # Crossing zero #
    #################
    # Upper bound #
    UB_prime_at_LB = UB_prime * (LB - UB) + UB_act

    # If tangent to UB is below LB, then take direct slope between LB and UB
    direct_upper = np & (UB_prime_at_LB <= 0)
    alpha_upper_k[direct_upper] = slope[direct_upper]

    # Else use bisection to find upper bound on slope.
    implicit_upper = np & (UB_prime_at_LB > 0)

    def f_upper(d: torch.Tensor) -> torch.Tensor:
        a_slope = (func(d) - func(LB[implicit_upper])) / (d - LB[implicit_upper])
        a_derivative = derivative(d)
        return a_slope - a_derivative

    # Bisection will return left and right bounds for d s.t. f_upper(d) is zero
    # Derivative of left bound will overapproximate the slope - hence a true bound
    d_upper, _ = bisection(torch.zeros_like(UB[implicit_upper]), UB[implicit_upper], f_upper)

    # Slope has to attach to (LB, sigma(LB))
    alpha_upper_k[implicit_upper] = derivative(d_upper)
    beta_upper_k[np] = LB_act[np] - alpha_upper_k[np] * LB[np]

    # Lower bound #
    LB_prime_at_UB = LB_prime * (UB - LB) + LB_act

    # If tangent to LB is above UB, then take direct slope between LB and UB
    direct_lower = np & (LB_prime_at_UB >= 0)
    alpha_lower_k[direct_lower] = slope[direct_lower]

    # Else use bisection to find upper bound on slope.
    implicit_lower = np & (LB_prime_at_UB < 0)

    def f_lower(d: torch.Tensor) -> torch.Tensor:
        a_slope = (func(UB[implicit_lower]) - func(d)) / (UB[implicit_lower] - d)
        a_derivative = derivative(d)
        return a_derivative - a_slope

    # Bisection will return left and right bounds for d s.t. f_lower(d) is zero
    # Derivative of right bound will overapproximate the slope - hence a true bound
    _, d_lower = bisection(LB[implicit_lower], torch.zeros_like(LB[implicit_lower]), f_lower)

    alpha_lower_k[implicit_lower] = derivative(d_lower)
    beta_lower_k[np] = UB_act[np] - alpha_lower_k[np] * UB[np]

    return (alpha_lower_k, alpha_upper_k), (beta_lower_k, beta_upper_k)


def bisection(l: torch.Tensor, h: torch.Tensor, f: TensorFunction, num_iter: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    midpoint = (l + h) / 2

    for _ in range(num_iter):
        y = f(midpoint)

        l[y <= 0] = midpoint[y <= 0]
        h[y > 0] = midpoint[y > 0]

        midpoint = (l + h) / 2

    return l, h


def alpha_beta_linear(class_or_obj):
    def alpha_beta(module: nn.Linear, LB: torch.Tensor, UB: torch.Tensor, **kwargs) -> AlphaBeta:
        return (None, None), (None, None)

    add_method(class_or_obj, 'alpha_beta', alpha_beta)
    return class_or_obj
