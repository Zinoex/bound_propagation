import torch
from torch import nn

from .general import BoundModule
from .bounds import IntervalBounds, LinearBounds


class Add(nn.Module):
    def __init__(self, network1, network2):
        super().__init__()

        self.network1, self.network2 = network1, network2

    def forward(self, x):
        return self.network1(x) + self.network2(x)


class BoundAdd(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.bound_network1 = factory.build(module.network1)
        self.bound_network2 = factory.build(module.network2)

    @property
    def need_relaxation(self):
        return self.bound_network1.need_relaxation or self.bound_network2.need_relaxation

    def clear_relaxation(self):
        self.bound_network1.clear_relaxation()
        self.bound_network2.clear_relaxation()

    def backward_relaxation(self, region):
        if self.bound_network1.need_relaxation:
            return self.bound_network1.backward_relaxation(region)
        else:
            assert self.bound_network2.need_relaxation
            return self.bound_network2.backward_relaxation(region)

    def crown_backward(self, linear_bounds):
        linear_bounds1 = self.bound_network1.crown_backward(linear_bounds)
        linear_bounds2 = self.bound_network2.crown_backward(linear_bounds)

        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (linear_bounds1.lower[0] + linear_bounds2.lower[0], linear_bounds1.lower[1] + linear_bounds2.lower[1] - linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (linear_bounds1.upper[0] + linear_bounds2.upper[0], linear_bounds1.upper[1] + linear_bounds2.upper[1] - linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        bounds1 = self.bound_network1.ibp_forward(bounds, save_relaxation=save_relaxation)
        bounds2 = self.bound_network2.ibp_forward(bounds, save_relaxation=save_relaxation)

        return IntervalBounds(
            bounds.region,
            bounds1.lower + bounds2.lower,
            bounds1.upper + bounds2.upper
        )

    def propagate_size(self, in_size):
        out_size1 = self.bound_network1.propagate_size(in_size)
        out_size2 = self.bound_network2.propagate_size(in_size)

        assert out_size1 == out_size2

        return out_size1


class Sub(nn.Module):
    def __init__(self, network1, network2):
        super().__init__()

        self.network1, self.network2 = network1, network2

    def forward(self, x):
        return self.network1(x) - self.network2(x)


class BoundSub(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.bound_network1 = factory.build(module.network1)
        self.bound_network2 = factory.build(module.network2)

    @property
    def need_relaxation(self):
        return self.bound_network1.need_relaxation or self.bound_network2.need_relaxation

    def clear_relaxation(self):
        self.bound_network1.clear_relaxation()
        self.bound_network2.clear_relaxation()

    def backward_relaxation(self, region):
        if self.bound_network1.need_relaxation:
            return self.bound_network1.backward_relaxation(region)
        else:
            assert self.bound_network2.need_relaxation
            return self.bound_network2.backward_relaxation(region)

    def crown_backward(self, linear_bounds):
        assert linear_bounds.lower is not None and linear_bounds.upper is not None, 'bound_lower=False and bound_upper=False cannot be used with Sub'

        input_bounds = LinearBounds(
            linear_bounds.region,
            (linear_bounds.lower[0], torch.zeros_like(linear_bounds.lower[1])) if linear_bounds.lower is not None else None,
            (linear_bounds.upper[0], torch.zeros_like(linear_bounds.upper[1])) if linear_bounds.upper is not None else None,
        )

        linear_bounds1 = self.bound_network1.crown_backward(input_bounds)
        linear_bounds2 = self.bound_network2.crown_backward(input_bounds)

        lower = (linear_bounds1.lower[0] - linear_bounds2.upper[0], linear_bounds1.lower[1] - linear_bounds2.upper[1] + linear_bounds.lower[1])
        upper = (linear_bounds1.upper[0] - linear_bounds2.lower[0], linear_bounds1.upper[1] - linear_bounds2.lower[1] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False):
        bounds1 = self.bound_network1.ibp_forward(bounds, save_relaxation=save_relaxation)
        bounds2 = self.bound_network2.ibp_forward(bounds, save_relaxation=save_relaxation)

        return IntervalBounds(
            bounds.region,
            bounds1.lower - bounds2.upper,
            bounds1.upper - bounds2.lower
        )

    def propagate_size(self, in_size):
        out_size1 = self.bound_network1.propagate_size(in_size)
        out_size2 = self.bound_network2.propagate_size(in_size)

        assert out_size1 == out_size2

        return out_size1
