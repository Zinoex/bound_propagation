import torch
from torch import nn

from .ibp import ibp
from .crown import linear_bounds, interval_bounds
from .util import add_method, LinearBounds, IntervalBounds


def crown_ibp(model):
    def crown_ibp_linear(self: nn.Sequential, lower: torch.Tensor, upper: torch.Tensor) -> LinearBounds:
        ibp(self)

        batch_size = lower.size(0)

        layer_bounds = self.ibp(lower, upper, pre=True)

        alpha_beta = self.alpha_beta(layer_bounds)
        bounds = linear_bounds(self, alpha_beta, batch_size, lower.device)

        return bounds

    add_method(model, 'crown_ibp_linear', crown_ibp_linear)

    def crown_ibp_interval(self: nn.Sequential, lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
        bounds = crown_ibp_linear(self, lower, upper)
        return interval_bounds(bounds, lower, upper)

    add_method(model, 'crown_ibp_interval', crown_ibp_interval)

    return model
