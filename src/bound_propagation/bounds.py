import torch


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

    def __getitem__(self, item):
        return HyperRectangle(
            self.lower[item] if self.lower is not None else None,
            self.upper[item] if self.upper is not None else None
        )

    def to(self, *args, **kwargs):
        return HyperRectangle(
            self.lower.to(*args, **kwargs) if self.lower is not None else None,
            self.upper.to(*args, **kwargs) if self.upper is not None else None
        )

    def cpu(self):
        return self.to(torch.device('cpu'))


class IntervalBounds(HyperRectangle):
    def __init__(self, region, lower, upper):
        super().__init__(lower, upper)
        self.region = region

    def __getitem__(self, item):
        return IntervalBounds(
            self.region[item],
            self.lower[item] if self.lower is not None else None,
            self.upper[item] if self.upper is not None else None
        )

    def to(self, *args, **kwargs):
        return IntervalBounds(
            self.region.to(*args, **kwargs),
            self.lower.to(*args, **kwargs) if self.lower is not None else None,
            self.upper.to(*args, **kwargs) if self.upper is not None else None
        )


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
            lower = center.matmul(slope) - diff.matmul(slope.abs())
            lower = lower.squeeze(-2) + intercept
        else:
            lower = None

        if self.upper is not None:
            slope, intercept = self.upper
            slope = slope.transpose(-1, -2)
            upper = center.matmul(slope) + diff.matmul(slope.abs())
            upper = upper.squeeze(-2) + intercept
        else:
            upper = None

        return IntervalBounds(self.region, lower, upper)

    def __len__(self):
        return self.lower.size(0)

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) >= 2 and idx[0] == Ellipsis:
            bias_idx = idx[:-1]
        else:
            bias_idx = idx

        return LinearBounds(
            self.region[bias_idx],
            (self.lower[0][idx], self.lower[1][bias_idx]) if self.lower is not None else None,
            (self.upper[0][idx], self.upper[1][bias_idx]) if self.upper is not None else None
        )

    def to(self, *args, **kwargs):
        return LinearBounds(
            self.region.to(*args, **kwargs),
            (self.lower[0].to(*args, **kwargs), self.lower[1].to(*args, **kwargs)) if self.lower is not None else None,
            (self.upper[0].to(*args, **kwargs), self.upper[1].to(*args, **kwargs)) if self.upper is not None else None
        )

    def cpu(self):
        return self.to(torch.device('cpu'))
