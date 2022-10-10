import abc

import torch


class IntervalBounds:
    def __init__(self, region, lower, upper):
        self.region = region
        self.lower, self.upper = lower, upper

    def __len__(self):
        return len(self.region)

    def __getitem__(self, item):
        return IntervalBounds(
            self.region[item],
            self.lower[item] if self.lower is not None else None,
            self.upper[item] if self.upper is not None else None
        )

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    @property
    def device(self):
        return self.region.device

    @property
    def dtype(self):
        return self.region.device

    def to(self, *args, **kwargs):
        return IntervalBounds(
            self.region.to(*args, **kwargs),
            self.lower.to(*args, **kwargs) if self.lower is not None else None,
            self.upper.to(*args, **kwargs) if self.upper is not None else None
        )

    def cpu(self):
        return self.to(torch.device('cpu'))

    def concretize(self):
        return self


class LinearBounds:
    def __init__(self, region, lower, upper):
        self.region = region
        self.lower, self.upper = lower, upper

    def concretize(self):
        lower, upper = self.region.concretize(self)
        return IntervalBounds(self.region, lower, upper)

    def __len__(self):
        return len(self.region)

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) >= 2 and idx[0] == Ellipsis:
            bias_idx = idx[:-1]
            region_idx = idx[:-2] + idx[-1:]
        else:
            bias_idx = idx
            region_idx = idx

        return LinearBounds(
            self.region[region_idx],
            (self.lower[0][idx], self.lower[1][bias_idx]) if self.lower is not None else None,
            (self.upper[0][idx], self.upper[1][bias_idx]) if self.upper is not None else None
        )

    @property
    def device(self):
        return self.region.device

    @property
    def dtype(self):
        return self.region.device

    def to(self, *args, **kwargs):
        return LinearBounds(
            self.region.to(*args, **kwargs),
            (self.lower[0].to(*args, **kwargs), self.lower[1].to(*args, **kwargs)) if self.lower is not None else None,
            (self.upper[0].to(*args, **kwargs), self.upper[1].to(*args, **kwargs)) if self.upper is not None else None
        )

    def cpu(self):
        return self.to(torch.device('cpu'))


class AbstractInputSet(abc.ABC):

    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError()

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def size(self, dim=None):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def device(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def dtype(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def to(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def cpu(self):
        return self.to(torch.device('cpu'))

    @abc.abstractmethod
    def concretize(self, linear_bounds):
        raise NotImplementedError()

    @abc.abstractmethod
    def bounding_hyperrect(self):
        raise NotImplementedError()


class HyperRectangle(AbstractInputSet):
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

    def size(self, dim=None):
        if dim is None:
            return self.lower.size()

        return self.lower.size(dim)

    @staticmethod
    def from_eps(x, eps):
        lower, upper = x - eps, x + eps
        return HyperRectangle(lower, upper)

    def __getitem__(self, item):
        return HyperRectangle(
            self.lower[item] if self.lower is not None else None,
            self.upper[item] if self.upper is not None else None
        )

    @property
    def device(self):
        return self.lower.device

    @property
    def dtype(self):
        return self.lower.dtype

    def to(self, *args, **kwargs):
        return HyperRectangle(
            self.lower.to(*args, **kwargs) if self.lower is not None else None,
            self.upper.to(*args, **kwargs) if self.upper is not None else None
        )

    def cpu(self):
        return self.to(torch.device('cpu'))

    def concretize(self, linear_bounds):
        center, diff = self.center, self.width / 2
        center, diff = center.unsqueeze(-2), diff.unsqueeze(-2)

        if linear_bounds.lower is not None:
            slope, intercept = linear_bounds.lower
            slope = slope.transpose(-1, -2)
            lower = center.matmul(slope) - diff.matmul(slope.abs())
            lower = lower.squeeze(-2) + intercept
        else:
            lower = None

        if linear_bounds.upper is not None:
            slope, intercept = linear_bounds.upper
            slope = slope.transpose(-1, -2)
            upper = center.matmul(slope) + diff.matmul(slope.abs())
            upper = upper.squeeze(-2) + intercept
        else:
            upper = None

        return lower, upper

    def bounding_hyperrect(self):
        return IntervalBounds(self, self.lower, self.upper)


class LpNormSet(AbstractInputSet):
    """
    Works for p > 1 (the dual of p = 1 is q = inf)
    """
    def __init__(self, mid, eps, p):
        self.mid, self.eps, self.p = mid, torch.as_tensor(eps), torch.as_tensor(p)

    def __getitem__(self, item):
        eps = self.eps
        if eps.dim() != 0:
            eps = eps[item]

        p = self.p
        if p.dim() != 0:
            p = p[item]

        return LpNormSet(self.mid[item], eps, p)

    def __len__(self):
        return self.mid.size(0)

    def size(self, dim=None):
        if dim is None:
            return self.mid.size()

        return self.mid.size(dim)

    @property
    def device(self):
        return self.mid.device

    @property
    def dtype(self):
        return self.mid.dtype

    def to(self, *args, **kwargs):
        return LpNormSet(self.mid.to(*args, **kwargs), self.eps.to(*args, **kwargs), self.p.to(*args, **kwargs))

    def cpu(self):
        return self.to(torch.device('cpu'))

    def concretize(self, linear_bounds):
        mid, eps = self.mid.unsqueeze(-2), self.eps
        q_norm = self.p / (1.0 - self.p)

        if linear_bounds.lower is not None:
            slope, intercept = linear_bounds.lower
            slope = slope.transpose(-1, -2)
            lower = mid.matmul(slope).squeeze(-2) - self.eps * slope.norm(q_norm, dim=-2) + intercept
        else:
            lower = None

        if linear_bounds.upper is not None:
            slope, intercept = linear_bounds.upper
            slope = slope.transpose(-1, -2)
            upper = mid.matmul(slope).squeeze(-2) + self.eps * slope.norm(q_norm, dim=-2) + intercept
        else:
            upper = None

        return lower, upper

    def bounding_hyperrect(self):
        return IntervalBounds(self, self.mid - self.eps, self.mid + self.eps)
