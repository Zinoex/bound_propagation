import torch
from torch import nn

from .general import BoundModule
from .bounds import IntervalBounds, LinearBounds


class Cat(nn.Module):
    def __init__(self, subnetwork):
        super().__init__()

        self.subnetwork = subnetwork

    def forward(self, x):
        y = self.subnetwork(x)
        return torch.cat([x, y], dim=-1)


class BoundCat(BoundModule):
    def __init__(self, model, factory, **kwargs):
        super().__init__(model, factory)

        self.subnetwork = factory.build(model.subnetwork)
        self.in_size = None

    @property
    def need_relaxation(self):
        return self.subnetwork.need_relaxation

    def clear_relaxation(self):
        self.subnetwork.clear_relaxation()

    def backward_relaxation(self, region):
        assert self.subnetwork.need_relaxation

        return self.subnetwork.backward_relaxation(region)

    def crown_backward(self, linear_bounds, optimize):
        assert self.in_size is not None

        residual_linear_bounds = self.subnetwork.crown_backward(linear_bounds[..., self.in_size:], optimize)
        linear_bounds = linear_bounds[..., :self.in_size]

        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (linear_bounds.lower[0] + residual_linear_bounds.lower[0], residual_linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (linear_bounds.upper[0] + residual_linear_bounds.upper[0], residual_linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        residual_bounds = self.subnetwork.ibp_forward(bounds, save_relaxation=save_relaxation)

        # Order matters here! - must match forward on Cat class
        lower = torch.cat([bounds.lower, residual_bounds.lower], dim=-1)
        upper = torch.cat([bounds.upper, residual_bounds.upper], dim=-1)
        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        self.in_size = in_size
        out_size = self.subnetwork.propagate_size(in_size)

        return in_size + out_size

    def bound_parameters(self):
        yield from self.subnetwork.bound_parameters()

    def reset_params(self):
        self.subnetwork.reset_params()

    def clip_params(self):
        self.subnetwork.clip_params()
