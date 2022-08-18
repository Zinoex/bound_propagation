from torch import nn

from .general import BoundModule
from .bounds import IntervalBounds, LinearBounds


class Residual(nn.Module):
    def __init__(self, subnetwork):
        super().__init__()

        self.subnetwork = subnetwork

    def forward(self, x):
        return self.subnetwork(x) + x


class BoundResidual(BoundModule):
    def __init__(self, model, factory, **kwargs):
        super().__init__(model, factory)

        self.subnetwork = factory.build(model.subnetwork)

    @property
    def need_relaxation(self):
        return self.subnetwork.need_relaxation

    def clear_relaxation(self):
        self.subnetwork.clear_relaxation()

    def backward_relaxation(self, region):
        assert self.subnetwork.need_relaxation

        return self.subnetwork.backward_relaxation(region)

    def crown_backward(self, linear_bounds, optimize):
        residual_linear_bounds = self.subnetwork.crown_backward(linear_bounds, optimize)

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

        return IntervalBounds(bounds.region, bounds.lower + residual_bounds.lower, bounds.upper + residual_bounds.upper)

    def propagate_size(self, in_size):
        out_size = self.subnetwork.propagate_size(in_size)
        assert in_size == out_size

        return out_size

    def bound_parameters(self):
        yield from self.subnetwork.bound_parameters()

    def reset_params(self):
        self.subnetwork.reset_params()

    def clip_params(self):
        self.subnetwork.clip_params()
