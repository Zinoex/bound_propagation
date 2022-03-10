import abc

from torch import nn


class HyperRectangle:
    def __init__(self, lower, upper):
        self.lower, self.upper = lower, upper

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    def __len__(self):
        return self.lower.size(0)

    @staticmethod
    def from_eps(x, eps):
        lower, upper = x - eps, x + eps
        return HyperRectangle(lower, upper)


class IntervalBounds(HyperRectangle):
    def __init__(self, region, lower, upper):
        super().__init__(lower, upper)
        self.region = region


class LinearBounds:
    def __init__(self, region, lower, upper):
        self.region = region
        self.lower, self.upper = lower, upper

    def concretize(self):
        center, diff = self.region.center, self.region.width / 2
        center, diff = center.unsqueeze(-2), diff.unsqueeze(-2)

        if self.lower is not None:
            slope, intercept = self.lower
            slope = slope.transpose(-1, -2)
            lower = center.matmul(slope) - diff.matmul(slope.abs()) + intercept.unsqueeze(-2)
            lower = lower.squeeze(-2)
        else:
            lower = None

        if self.upper is not None:
            slope, intercept = self.upper
            slope = slope.transpose(-1, -2)
            upper = center.matmul(slope) + diff.matmul(slope.abs()) + intercept.unsqueeze(-2)
            upper = upper.squeeze(-2)
        else:
            upper = None

        return IntervalBounds(self.region, lower, upper)


class BoundModule(nn.Module, abc.ABC):
    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module

    @abc.abstractmethod
    def crown(self, region, **kwargs):
        pass

    @abc.abstractmethod
    def crown_ibp(self, region, **kwargs):
        pass

    @abc.abstractmethod
    def ibp(self, region, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
