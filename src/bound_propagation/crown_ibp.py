import torch
from torch import nn

from .ibp import ibp
from .crown import linear_bounds, interval_bounds, output_size
from .util import add_method, LinearBounds, IntervalBounds


def crown_ibp(model):
    def crown_ibp_linear(self: nn.Sequential, lower: torch.Tensor, upper: torch.Tensor) -> LinearBounds:
        ibp(self)

        layer_bounds = self.ibp(lower, upper, pre=True)
        alpha_betas = self.alpha_beta(layer_bounds)

        batch_size = lower.size(0)
        out_size = output_size(lower.size(-1), self)
        bounds = linear_bounds(self, alpha_betas, batch_size, out_size, lower.device)

        return bounds

    add_method(model, 'crown_ibp_linear', crown_ibp_linear)

    def crown_ibp_interval(self: nn.Sequential, lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
        bounds = crown_ibp_linear(self, lower, upper)
        return interval_bounds(bounds, (lower, upper))

    add_method(model, 'crown_ibp_interval', crown_ibp_interval)

    return model
