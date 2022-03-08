from typing import Callable, Tuple, List, Union, Optional

import torch
from torch import nn

from .alpha_beta import alpha_beta, add_alpha_beta_submodules
from .util import add_method, LinearBounds, IntervalBounds, LayerBounds, AlphaBetas, LinearBound, AlphaBeta, WeightBias


def crown(model: nn.Sequential):
    def crown_linear(self: nn.Sequential, lower: torch.Tensor, upper: torch.Tensor, **kwargs) -> LinearBounds:
        add_alpha_beta_submodules(self)
        subnetwork_crown(self)

        alpha_betas = []

        for i in range(len(self)):
            # We do this wrt. next layer because if the next layer is linear, we can skip this iteration
            b = self[i].subnetwork_crown(self[:i], alpha_betas, (lower, upper))
            alpha_beta = self[i].alpha_beta(b)
            alpha_betas.append(alpha_beta)

        batch_size = lower.size(0)
        out_size = output_size(lower.size(-1), self)
        bounds = linear_bounds(self, alpha_betas, batch_size, out_size, lower.device, **kwargs)

        return bounds

    add_method(model, 'crown_linear', crown_linear)

    def crown_interval(self: nn.Sequential, lower: torch.Tensor, upper: torch.Tensor, **kwargs) -> IntervalBounds:
        bounds = crown_linear(self, lower, upper, **kwargs)
        return interval_bounds(bounds, (lower, upper))

    add_method(model, 'crown_interval', crown_interval)

    model = alpha_beta(model)

    return model


def subnetwork_crown(model):
    if isinstance(model, nn.Sequential):
        return subnetwork_crown_sequential(model)
    elif isinstance(model, nn.Linear):
        return subnetwork_crown_linear(model)
    else:
        return subnetwork_crown_activation(model)


def subnetwork_crown_sequential(model: nn.Sequential):
    for module in model:
        subnetwork_crown(module)


def subnetwork_crown_linear(class_or_obj):
    def subnetwork_crown(self: nn.Linear, model: nn.Sequential, layer_bounds: LayerBounds, input_bounds: IntervalBounds) -> Optional[IntervalBounds]:
        return None

    add_method(class_or_obj, 'subnetwork_crown', subnetwork_crown)
    return class_or_obj


def subnetwork_crown_activation(class_or_obj):
    def subnetwork_crown(self: nn.Module, model: nn.Sequential, alpha_betas: AlphaBetas, input_bounds: IntervalBounds) -> IntervalBounds:
        device = input_bounds[0].device
        batch_size = input_bounds[0].size(0)
        out_size = output_size(input_bounds[0].size(-1), model)
        bounds = linear_bounds(model, alpha_betas, batch_size, out_size, device)

        return interval_bounds(bounds, input_bounds)

    add_method(class_or_obj, 'subnetwork_crown', subnetwork_crown)
    return class_or_obj


def output_size(input_size: int, model: nn.Sequential) -> int:
    out_size = input_size

    for module in model:
        if isinstance(module, nn.Linear):
            out_size = module.out_features

    return out_size


def interval_bounds(bounds: LinearBounds, input_bounds: IntervalBounds) -> IntervalBounds:
    lower, upper = input_bounds
    lower, upper = lower.unsqueeze(-1), upper.unsqueeze(-1)

    # We can do this instead of finding the Q-norm, as we only deal with perturbation over a hyperrectangular input,
    # and not an arbitrary B_p(epsilon) ball
    # This is essentially:
    # - min_x Omega_0 @ x + Omega_accumulator
    # - max_x Gamma_0 @ x + Gamma_accumulator

    mid = (lower + upper) / 2
    diff = (upper - lower) / 2

    Omega, Gamma = bounds

    if Omega is not None:
        Omega_0, Omega_accumulator = Omega
        min_Omega_x = (Omega_0.matmul(mid) - Omega_0.abs().matmul(diff))[..., 0]
        Omega = min_Omega_x + Omega_accumulator

    if Gamma is not None:
        Gamma_0, Gamma_accumulator = Gamma
        max_Gamma_x = (Gamma_0.matmul(mid) + Gamma_0.abs().matmul(diff))[..., 0]
        Gamma = max_Gamma_x + Gamma_accumulator

    return Omega, Gamma


def linear_bounds(model: nn.Sequential, alpha_betas: AlphaBetas, batch_size: int, out_size: int, device: torch.device,
                  bound_lower=True, bound_upper=True) -> LinearBounds:
    # Compute bounds as two iterations to reduce memory consumption by half
    lower = oneside_linear_bound(model, alpha_betas, batch_size, out_size, device, lower=True) if bound_lower else None
    upper = oneside_linear_bound(model, alpha_betas, batch_size, out_size, device, lower=False) if bound_upper else None

    return lower, upper


def oneside_linear_bound(model: nn.Sequential, alpha_betas: AlphaBetas, batch_size: int, out_size: int, device: torch.device, **kwargs) -> LinearBound:
    W_tilde = torch.eye(out_size, device=device).unsqueeze(0).expand(batch_size, out_size, out_size)
    acc = 0

    # List is necessary around zip to allow reversing
    for module, alpha_beta in reversed(list(zip(model, alpha_betas))):
        if not hasattr(module, 'crown_backward'):
            # Decorator also adds the method inplace.
            crown_backward(module)

        W_tilde, bias = module.crown_backward(W_tilde, alpha_beta, **kwargs)
        acc = acc + bias

    return W_tilde, acc


def crown_backward(class_or_obj):
    types = {
        nn.Linear: crown_backward_linear,
        nn.ReLU: crown_backward_activation,
        nn.Sigmoid: crown_backward_activation,
        nn.Tanh: crown_backward_activation
    }

    for layer_type, crown_backward_fun in types.items():
        if isinstance(class_or_obj, layer_type) or \
                (isinstance(class_or_obj, type) and issubclass(class_or_obj, layer_type)):
            return crown_backward_fun(class_or_obj)

    raise NotImplementedError('Selected type of layer not supported')


def crown_backward_linear(class_or_obj):
    def crown_backward(self: nn.Linear, W_tilde: torch.Tensor, alpha_beta: AlphaBeta, **kwargs) -> WeightBias:
        bias = self.bias
        weight = self.weight

        if bias is None:
            bias_acc = 0
        elif bias.dim() == 1:
            bias_acc = W_tilde.matmul(bias)
        else:
            # This allows stochastic dynamics to be treated as an nn.Linear
            bias = bias.view(bias.size(0), 1, bias.size(-1), 1)
            bias_acc = W_tilde.matmul(bias)[..., 0]

        if weight.dim() == 2:
            W_tilde = W_tilde.matmul(weight)
        else:
            if W_tilde.dim() == 3:
                W_tilde = W_tilde.unsqueeze(0)

            W_tilde = W_tilde.matmul(weight.unsqueeze(1))

        return W_tilde, bias_acc

    add_method(class_or_obj, 'crown_backward', crown_backward)
    return class_or_obj


def crown_backward_activation(class_or_obj):
    def crown_backward(self: nn.Module, W_tilde: torch.Tensor, alpha_beta: AlphaBeta, lower: bool = True, **kwargs) -> WeightBias:
        if lower:
            return act_lower(W_tilde, alpha_beta)
        else:
            return act_upper(W_tilde, alpha_beta)

    add_method(class_or_obj, 'crown_backward', crown_backward)
    return class_or_obj


def act_lower(Omega_tilde: torch.Tensor, alpha_beta: AlphaBeta) -> WeightBias:
    (al_k, au_k), (bl_k, bu_k) = alpha_beta

    bias = torch.sum(Omega_tilde * _theta(Omega_tilde, bl_k, bu_k), dim=-1)
    Omega_tilde = Omega_tilde * _omega(Omega_tilde, al_k, au_k)

    return Omega_tilde, bias


def _theta(Omega_tilde: torch.Tensor, beta_lower: torch.Tensor, beta_upper: torch.Tensor) -> torch.Tensor:
    return torch.where(Omega_tilde < 0, beta_upper.unsqueeze(-2), beta_lower.unsqueeze(-2))


def _omega(Omega_tilde: torch.Tensor, alpha_lower: torch.Tensor, alpha_upper: torch.Tensor) -> torch.Tensor:
    return torch.where(Omega_tilde < 0, alpha_upper.unsqueeze(-2), alpha_lower.unsqueeze(-2))


def act_upper(Gamma_tilde: torch.Tensor, alpha_beta: AlphaBeta) -> WeightBias:
    (al_k, au_k), (bl_k, bu_k) = alpha_beta

    bias = torch.sum(Gamma_tilde * _delta(Gamma_tilde, bl_k, bu_k), dim=-1)
    Gamma_tilde = Gamma_tilde * _lambda(Gamma_tilde, al_k, au_k)

    return Gamma_tilde, bias


def _delta(Gamma_tilde: torch.Tensor, beta_lower: torch.Tensor, beta_upper: torch.Tensor) -> torch.Tensor:
    return torch.where(Gamma_tilde < 0, beta_lower.unsqueeze(-2), beta_upper.unsqueeze(-2))


def _lambda(Gamma_tilde: torch.Tensor, alpha_lower: torch.Tensor, alpha_upper: torch.Tensor) -> torch.Tensor:
    return torch.where(Gamma_tilde < 0, alpha_lower.unsqueeze(-2), alpha_upper.unsqueeze(-2))
