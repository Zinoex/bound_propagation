import torch
from torch import nn

from .general import BoundModule
from .bounds import IntervalBounds, LinearBounds


class Select(nn.Module):
    def __init__(self, indices):
        super().__init__()

        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, slice):
            if indices.step is None:
                r = range(indices.start, indices.stop)
            else:
                r = range(indices.start, indices.stop, indices.step)

            indices = [i for i in r]

        self.indices = indices

    def forward(self, x):
        return x[..., self.indices]


class BoundSelect(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory)

        self.in_size = None

    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds, optimize):
        assert self.in_size

        if linear_bounds.lower is None:
            lower = None
        else:
            lowerA = torch.zeros((*linear_bounds.lower[0].size()[:-1], self.in_size), device=linear_bounds.lower[0].device)

            view_size = *[1 for _ in range(linear_bounds.lower[0].dim() - 1)], -1
            indices = torch.tensor(self.module.indices, device=linear_bounds.lower[0].device).view(*view_size).expand_as(linear_bounds.lower[0])
            lowerA.scatter_add_(-1, indices, linear_bounds.lower[0])

            lower = (lowerA, linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upperA = torch.zeros((*linear_bounds.upper[0].size()[:-1], self.in_size), device=linear_bounds.upper[0].device)

            view_size = *[1 for _ in range(linear_bounds.upper[0].dim() - 1)], -1
            indices = torch.tensor(self.module.indices, device=linear_bounds.upper[0].device).view(*view_size).expand_as(linear_bounds.upper[0])
            upperA.scatter_add_(-1, indices, linear_bounds.upper[0])
            upper = (upperA, linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        return IntervalBounds(bounds.region, bounds.lower[..., self.module.indices], bounds.upper[..., self.module.indices])

    def propagate_size(self, in_size):
        self.in_size = in_size

        return len(self.module.indices)
