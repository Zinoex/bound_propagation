import torch
from torch import nn

from .bounds import IntervalBounds, LinearBounds
from .general import BoundModule


class Parallel(nn.Module):
    def __init__(self, subnetwork1, subnetwork2, split_size=None):
        super().__init__()

        self.split_size = split_size
        self.subnetwork1 = subnetwork1
        self.subnetwork2 = subnetwork2

    def forward(self, x):
        if self.split_size:
            x1, x2 = x[..., :self.split_size], x[..., self.split_size:]
        else:
            x1, x2 = x, x

        return torch.cat([self.subnetwork1(x1), self.subnetwork2(x2)], dim=-1)


class BoundParallel(BoundModule):
    def __init__(self, model, factory, **kwargs):
        super().__init__(model, factory)

        self.out_size1 = None
        self.subnetwork1 = factory.build(model.subnetwork1)
        self.subnetwork2 = factory.build(model.subnetwork2)

    @property
    def need_relaxation(self):
        return self.subnetwork1.need_relaxation or self.subnetwork2.need_relaxation

    def clear_relaxation(self):
        self.subnetwork1.clear_relaxation()
        self.subnetwork2.clear_relaxation()

    def backward_relaxation(self, region):
        if self.subnetwork1.need_relaxation:
            linear_bounds, relaxation_module = self.subnetwork1.backward_relaxation(region)
            return self.padding(linear_bounds, order=0), relaxation_module
        else:
            assert self.subnetwork2.need_relaxation
            linear_bounds, relaxation_module = self.subnetwork2.backward_relaxation(region)
            return self.padding(linear_bounds, order=1), relaxation_module

    def padding(self, linear_bounds, order=0):
        if self.module.split_size is None:
            return linear_bounds

        if order == 0:
            lowerA = torch.cat([linear_bounds.lower[0], torch.zeros_like(linear_bounds.lower[0])], dim=-1)
            upperA = torch.cat([linear_bounds.upper[0], torch.zeros_like(linear_bounds.upper[0])], dim=-1)
        else:
            lowerA = torch.cat([torch.zeros_like(linear_bounds.lower[0]), linear_bounds.lower[0]], dim=-1)
            upperA = torch.cat([torch.zeros_like(linear_bounds.upper[0]), linear_bounds.upper[0]], dim=-1)

        return LinearBounds(
            linear_bounds.region,
            (lowerA, linear_bounds.lower[1]),
            (upperA, linear_bounds.upper[1])
        )

    def crown_backward(self, linear_bounds, optimize):
        assert self.out_size1 is not None

        residual_linear_bounds1 = self.subnetwork1.crown_backward(linear_bounds[..., :self.out_size1], optimize)
        residual_linear_bounds2 = self.subnetwork2.crown_backward(linear_bounds[..., self.out_size1:], optimize)

        if linear_bounds.lower is None:
            lower = None
        else:
            if self.module.split_size is not None:
                lowerA = torch.cat([residual_linear_bounds1.lower[0], residual_linear_bounds2.lower[0]], dim=-1)
            else:
                lowerA = residual_linear_bounds1.lower[0] + residual_linear_bounds2.lower[0]

            lower = (lowerA, residual_linear_bounds1.lower[1] + residual_linear_bounds2.lower[1] - linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            if self.module.split_size is not None:
                upperA = torch.cat([residual_linear_bounds1.upper[0], residual_linear_bounds2.upper[0]], dim=-1)
            else:
                upperA = residual_linear_bounds1.upper[0] + residual_linear_bounds2.upper[0]

            upper = (upperA, residual_linear_bounds1.upper[1] + residual_linear_bounds2.upper[1] - linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        if self.module.split_size is not None:
            bounds1, bounds2 = bounds[..., :self.module.split_size], bounds[..., self.module.split_size:]
        else:
            bounds1, bounds2 = bounds, bounds

        residual_bounds1 = self.subnetwork1.ibp_forward(bounds1, save_relaxation=save_relaxation)
        residual_bounds2 = self.subnetwork2.ibp_forward(bounds2, save_relaxation=save_relaxation)

        # Order matters here! - must match forward on Parallel class
        lower = torch.cat([residual_bounds1.lower, residual_bounds2.lower], dim=-1)
        upper = torch.cat([residual_bounds1.upper, residual_bounds2.upper], dim=-1)
        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        if self.module.split_size is not None:
            out_size1 = self.subnetwork1.propagate_size(self.module.split_size)
            out_size2 = self.subnetwork2.propagate_size(in_size - self.module.split_size)
        else:
            out_size1 = self.subnetwork1.propagate_size(in_size)
            out_size2 = self.subnetwork2.propagate_size(in_size)

        self.out_size1 = out_size1

        return out_size1 + out_size2

    def bound_parameters(self):
        yield from self.subnetwork1.bound_parameters()
        yield from self.subnetwork2.bound_parameters()

    def reset_params(self):
        self.subnetwork1.reset_params()
        self.subnetwork2.reset_params()

    def clip_params(self):
        self.subnetwork1.clip_params()
        self.subnetwork2.clip_params()
