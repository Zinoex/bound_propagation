from typing import Tuple, Optional, Union

import torch
from torch import nn, Tensor

from bound_propagation import BoundActivation
from bound_propagation.activation import assert_bound_order


class Clamp(nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()

        assert min is not None or max is not None

        self.min = min
        self.max = max

    def forward(self, x):
        return x.clamp(min=self.min, max=self.max)


def regimes(lower: torch.Tensor, upper: torch.Tensor, min: Optional[Union[Tensor, float]], max: Optional[Union[Tensor, float]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_lower = torch.full_like(lower, False, dtype=torch.bool)
    flat_upper = torch.full_like(lower, False, dtype=torch.bool)
    slope = torch.full_like(lower, False, dtype=torch.bool)
    lower_bend = torch.full_like(lower, False, dtype=torch.bool)
    upper_bend = torch.full_like(lower, False, dtype=torch.bool)
    full_range = torch.full_like(lower, False, dtype=torch.bool)

    tautology = torch.full_like(lower, True, dtype=torch.bool)

    if min is not None:
        flat_lower |= upper <= min
        lower_bend |= (lower < min) & (upper > min) & (tautology if max is None else (upper < max))
    else:
        slope |= (upper <= max)

    if max is not None:
        flat_upper |= lower >= max
        upper_bend |= (lower < max) & (upper > max) & (tautology if min is None else (lower > min))
    else:
        slope |= (lower >= min)

    if min is not None and max is not None:
        full_range |= (lower < min) & (upper > max)
        slope |= (lower >= min) & (upper <= max)

    return flat_lower, flat_upper, slope, lower_bend, upper_bend, full_range


class BoundClamp(BoundActivation):
    def __init__(self, module, factory, adaptive_clamp=True, **kwargs):
        super().__init__(module, factory)
        self.adaptive_clamp = adaptive_clamp

    @assert_bound_order
    def alpha_beta(self, preactivation):
        """
        Adaptive is similar to :BoundReLU: with the adaptivity being applied to both bends

        :param self:
        :param preactivation:
        """
        lower, upper = preactivation.lower, preactivation.upper
        flat_lower, flat_upper, slope, lower_bend, upper_bend, full_range = regimes(lower, upper, self.module.min, self.module.max)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        # Flat lower
        self.alpha_lower[flat_lower], self.beta_lower[flat_lower] = 0, self.module.min or 0
        self.alpha_upper[flat_lower], self.beta_upper[flat_lower] = 0, self.module.min or 0

        # Flat upper
        self.alpha_lower[flat_upper], self.beta_lower[flat_upper] = 0, self.module.max or 0
        self.alpha_upper[flat_upper], self.beta_upper[flat_upper] = 0, self.module.max or 0

        # Slope
        self.alpha_lower[slope], self.beta_lower[slope] = 1, 0
        self.alpha_upper[slope], self.beta_upper[slope] = 1, 0

        z = (self(upper) - self(lower)) / (upper - lower)
        if self.adaptive_clamp:
            # Utilize that bool->float conversion is true=1 and false=0
            a = (upper >= torch.abs(lower)).to(lower.dtype)
        else:
            a = z

        # Lower bend
        if self.module.min is not None:
            self.alpha_lower[lower_bend], self.beta_lower[lower_bend] = a[lower_bend], self.module.min * (1 - a[lower_bend])
            self.alpha_upper[lower_bend], self.beta_upper[lower_bend] = z[lower_bend], self(lower[lower_bend]) - lower[lower_bend] * z[lower_bend]

        # Upper bend
        if self.module.max is not None:
            self.alpha_lower[upper_bend], self.beta_lower[upper_bend] = z[upper_bend], self(lower[upper_bend]) - lower[upper_bend] * z[upper_bend]
            self.alpha_upper[upper_bend], self.beta_upper[upper_bend] = a[upper_bend], self.module.max * (1 - a[upper_bend])

        # Full range
        if self.module.min is not None and self.module.max is not None:
            z_lower = (self(upper[full_range]) - self.module.min) / (upper[full_range] - self.module.min)
            self.alpha_lower[full_range], self.beta_lower[full_range] = z_lower, self(upper[full_range]) - upper[full_range] * z_lower
            z_upper = (self.module.max - self(lower[full_range])) / (self.module.max - lower[full_range])
            self.alpha_upper[full_range], self.beta_upper[full_range] = z_upper, self(lower[full_range]) - lower[full_range] * z_upper
