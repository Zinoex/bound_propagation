from typing import Callable, Tuple, List, Union, Optional

import torch
from torch import nn

from .alpha_beta import alpha_beta
from .util import add_method, LinearBounds, IntervalBounds, LayerBounds


def crown(model: nn.Sequential):
    def crown_linear(self: nn.Sequential, lower: torch.Tensor, upper: torch.Tensor) -> LinearBounds:
        batch_size = lower.size(0)
        layer_bounds = [(lower, upper)]

        for i in range(1, len(self)):
            subnetwork = self[:i]
            b = subnetwork_crown(subnetwork, layer_bounds, batch_size, lower.device)
            layer_bounds.append(b)

        alpha_betas = self.alpha_beta(layer_bounds)
        bounds = linear_bounds(self, alpha_betas, batch_size, lower.device)

        return bounds

    add_method(model, 'crown_linear', crown_linear)

    def crown_interval(self: nn.Sequential, lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
        bounds = crown_linear(self, lower, upper)
        return interval_bounds(bounds, lower, upper)

    add_method(model, 'crown_interval', crown_interval)

    model = alpha_beta(model)

    return model


def subnetwork_crown(model: nn.Sequential, layer_bounds: LayerBounds, batch_size: int, device: torch.device) -> IntervalBounds:
    model = alpha_beta(model)

    alpha_betas = model.alpha_beta(layer_bounds)
    bounds = linear_bounds(model, alpha_betas, batch_size, device)

    return interval_bounds(bounds, *layer_bounds[0])


def interval_bounds(bounds: LinearBounds, lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
    (Omega_0, Omega_accumulator), (Gamma_0, Gamma_accumulator) = bounds

    lower, upper = lower.unsqueeze(-1), upper.unsqueeze(-1)

    # We can do this instead of finding the Q-norm, as we only deal with perturbation over a hyperrectangular input,
    # and not an arbitrary B_p(epsilon) ball
    # This is essentially:
    # - min_x Omega_0 @ x + Omega_accumulator
    # - max_x Gamma_0 @ x + Gamma_accumulator

    mid = (lower + upper) / 2
    diff = (upper - lower) / 2

    min_Omega_x = (torch.matmul(Omega_0, mid) - torch.matmul(torch.abs(Omega_0), diff))[..., 0]
    max_Gamma_x = (torch.matmul(Gamma_0, mid) + torch.matmul(torch.abs(Gamma_0), diff))[..., 0]

    return min_Omega_x + Omega_accumulator, max_Gamma_x + Gamma_accumulator


def linear_bounds(model, alpha_betas, batch_size, device):
    # Compute bounds as two iterations to reduce memory consumption by half
    return oneside_linear_bound(model, alpha_betas, batch_size, device, lower=True), \
           oneside_linear_bound(model, alpha_betas, batch_size, device, lower=False)


def oneside_linear_bound(model, alpha_betas, batch_size, device, **kwargs):
    out_size = output_size(model)

    W_tilde = torch.eye(out_size, device=device).unsqueeze(0).expand(batch_size, out_size, out_size)
    acc = 0

    # List is necessary around zip to allow reversing
    for module, alpha_beta in reversed(list(zip(model, alpha_betas))):
        if not hasattr(module, 'crown_backward'):
            # Decorator also adds the method inplace.
            crown_backward(module)

        W_tilde, bias = module.crown_backward(W_tilde, module, alpha_beta, **kwargs)
        acc = acc + bias

    return W_tilde, acc


def output_size(model: nn.Sequential) -> int:
    for module in reversed(model):
        if isinstance(module, nn.Linear):
            return module.out_features


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
    def crown_backward(self: nn.Linear, W_tilde, module, alpha_beta, **kwargs):
        W_tilde_new = torch.matmul(W_tilde, self.weight)

        if self.bias is None:
            bias_acc = 0
        else:
            bias_acc = torch.matmul(W_tilde, self.bias)

        return W_tilde_new, bias_acc

    add_method(class_or_obj, 'crown_backward', crown_backward)
    return class_or_obj


def crown_backward_activation(class_or_obj):
    def crown_backward(self: nn.Module, W_tilde, module, alpha_beta, lower=True, **kwargs):
        if lower:
            return act_lower(W_tilde, alpha_beta)
        else:
            return act_upper(W_tilde, alpha_beta)

    add_method(class_or_obj, 'crown_backward', crown_backward)
    return class_or_obj


def act_lower(Omega_tilde, alpha_beta):
    (al_k, au_k), (bl_k, bu_k) = alpha_beta

    bias = torch.sum(Omega_tilde * _theta(Omega_tilde, bl_k, bu_k), dim=-1)
    Omega_tilde = Omega_tilde * _omega(Omega_tilde, al_k, au_k)

    return Omega_tilde, bias


def _theta(Omega_tilde, beta_lower, beta_upper):
    return torch.where(Omega_tilde < 0, beta_upper.unsqueeze(-2), beta_lower.unsqueeze(-2))


def _omega(Omega_tilde, alpha_lower, alpha_upper):
    return torch.where(Omega_tilde < 0, alpha_upper.unsqueeze(-2), alpha_lower.unsqueeze(-2))


def act_upper(Gamma_tilde, alpha_beta):
    (al_k, au_k), (bl_k, bu_k) = alpha_beta

    bias = torch.sum(Gamma_tilde * _delta(Gamma_tilde, bl_k, bu_k), dim=-1)
    Gamma_tilde = Gamma_tilde * _lambda(Gamma_tilde, al_k, au_k)

    return Gamma_tilde, bias


def _delta(Gamma_tilde, beta_lower, beta_upper):
    return torch.where(Gamma_tilde < 0, beta_lower.unsqueeze(-2), beta_upper.unsqueeze(-2))


def _lambda(Gamma_tilde, alpha_lower, alpha_upper):
    return torch.where(Gamma_tilde < 0, alpha_lower.unsqueeze(-2), alpha_upper.unsqueeze(-2))
