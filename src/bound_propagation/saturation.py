import logging
from typing import Tuple, Optional, Union

import torch
from torch import nn, Tensor

from bound_propagation import BoundActivation
from bound_propagation.activation import assert_bound_order

logger = logging.getLogger(__name__)


class Clamp(nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()

        assert min is not None or max is not None

        self.min = min
        self.max = max

    def forward(self, x):
        return x.clamp(min=self.min, max=self.max)


def regimes(lower: torch.Tensor, upper: torch.Tensor, min: Optional[Union[Tensor, float]],
            max: Optional[Union[Tensor, float]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_lower = torch.full_like(lower, False, dtype=torch.bool)
    flat_upper = torch.full_like(lower, False, dtype=torch.bool)
    slope = torch.full_like(lower, False, dtype=torch.bool)
    lower_bend = torch.full_like(lower, False, dtype=torch.bool)
    upper_bend = torch.full_like(lower, False, dtype=torch.bool)
    full_range = torch.full_like(lower, False, dtype=torch.bool)
    zero_width = torch.isclose(lower, upper, rtol=0.0, atol=1e-8)

    tautology = torch.full_like(lower, True, dtype=torch.bool)

    if min is not None:
        flat_lower |= (~zero_width) & (upper <= min)
        lower_bend |= (~zero_width) & (lower < min) & (upper > min) & (tautology if max is None else (upper < max))
    else:
        slope |= (~zero_width) & (upper <= max)

    if max is not None:
        flat_upper |= (~zero_width) & (lower >= max)
        upper_bend |= (~zero_width) & (lower < max) & (upper > max) & (tautology if min is None else (lower > min))
    else:
        slope |= (~zero_width) & (lower >= min)

    if min is not None and max is not None:
        full_range |= (~zero_width) & (lower < min) & (upper > max)
        slope |= (~zero_width) & (lower >= min) & (upper <= max)

    return zero_width, flat_lower, flat_upper, slope, lower_bend, upper_bend, full_range


class BoundClamp(BoundActivation):
    def __init__(self, module, factory, adaptive_clamp=True, **kwargs):
        super().__init__(module, factory)
        self.adaptive_clamp = adaptive_clamp

        self.unstable_lower, self.unstable_slope_lower = None, None
        self.unstable_upper, self.unstable_slope_upper = None, None

    def clear_relaxation(self):
        super().clear_relaxation()
        self.unstable_lower, self.unstable_slope_lower = None, None
        self.unstable_upper, self.unstable_slope_upper = None, None

    @assert_bound_order
    def alpha_beta(self, preactivation):
        """
        Adaptive is similar to :BoundReLU: with the adaptivity being applied to both bends

        :param self:
        :param preactivation:
        """
        lower, upper = preactivation.lower, preactivation.upper
        zero_width, flat_lower, flat_upper, slope, lower_bend, upper_bend, full_range = regimes(lower, upper,
                                                                                                self.module.min,
                                                                                                self.module.max)

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        act_lower, act_upper = self(lower), self(upper)

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        if zero_width.any():
            self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, act_lower[zero_width]
            self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, act_upper[zero_width]

        min, max = self.module.min, self.module.max
        if min is not None and torch.is_tensor(min):
            min = min.view(*[1 for _ in range(self.alpha_lower.dim() - 1)], -1).expand_as(self.alpha_lower)

        if max is not None and torch.is_tensor(max):
            max = max.view(*[1 for _ in range(self.alpha_lower.dim() - 1)], -1).expand_as(self.alpha_lower)

        # Flat lower
        flat_lower_min = min[flat_lower] if min is not None and torch.is_tensor(min) else min
        self.alpha_lower[flat_lower], self.beta_lower[flat_lower] = 0, 0 if min is None else flat_lower_min
        self.alpha_upper[flat_lower], self.beta_upper[flat_lower] = 0, 0 if min is None else flat_lower_min

        # Flat upper
        flat_upper_max = max[flat_upper] if max is not None and torch.is_tensor(max) else max
        self.alpha_lower[flat_upper], self.beta_lower[flat_upper] = 0, 0 if max is None else flat_upper_max
        self.alpha_upper[flat_upper], self.beta_upper[flat_upper] = 0, 0 if max is None else flat_upper_max

        # Slope
        self.alpha_lower[slope], self.beta_lower[slope] = 1, 0
        self.alpha_upper[slope], self.beta_upper[slope] = 1, 0

        z = (self(upper) - self(lower)) / (upper - lower)

        # Lower bend
        if min is not None:
            if self.adaptive_clamp:
                # Utilize that bool->float conversion is true=1 and false=0
                a = (upper - min >= min - lower).to(lower.dtype)
            else:
                a = z

            lower_bend_min = min[lower_bend] if torch.is_tensor(min) else min
            self.alpha_lower[lower_bend], self.beta_lower[lower_bend] = a[lower_bend], lower_bend_min * (
                        1 - a[lower_bend])
            self.alpha_upper[lower_bend], self.beta_upper[lower_bend] = z[lower_bend], act_lower[lower_bend] - lower[
                lower_bend] * z[lower_bend]

            self.unstable_lower = lower_bend
            self.unstable_slope_lower = z[lower_bend].detach().clone().requires_grad_()

        # Upper bend
        if max is not None:
            if self.adaptive_clamp:
                # Utilize that bool->float conversion is true=1 and false=0
                a = (upper - max <= max - lower).to(lower.dtype)
            else:
                a = z

            upper_bend_max = max[upper_bend] if torch.is_tensor(max) else max
            self.alpha_lower[upper_bend], self.beta_lower[upper_bend] = z[upper_bend], act_lower[upper_bend] - lower[
                upper_bend] * z[upper_bend]
            self.alpha_upper[upper_bend], self.beta_upper[upper_bend] = a[upper_bend], upper_bend_max * (
                        1 - a[upper_bend])

            self.unstable_upper = upper_bend
            self.unstable_slope_upper = z[upper_bend].detach().clone().requires_grad_()

        # Full range
        if self.module.min is not None and self.module.max is not None:
            full_range_min = min[full_range] if torch.is_tensor(min) else min
            full_range_max = max[full_range] if torch.is_tensor(max) else max

            z_lower = (act_upper[full_range] - full_range_min) / (upper[full_range] - full_range_max)
            self.alpha_lower[full_range], self.beta_lower[full_range] = z_lower, act_upper[full_range] - upper[
                full_range] * z_lower

            z_upper = (full_range_max - act_lower[full_range]) / (full_range_max - lower[full_range])
            self.alpha_upper[full_range], self.beta_upper[full_range] = z_upper, act_lower[full_range] - lower[
                full_range] * z_upper

    def parameterize_alpha_beta(self, alpha_lower, alpha_upper, beta_lower, beta_upper):
        if self.unstable_lower is None and self.unstable_upper is None:
            logger.warning('Clamp bound not parameterized but expected to')

        if self.unstable_lower is not None:
            min = self.module.min
            if torch.is_tensor(min):
                min = min.view(*[1 for _ in range(alpha_lower.dim() - 1)], -1).expand_as(alpha_lower)[self.unstable_lower]
            alpha_lower[self.unstable_lower], beta_lower[self.unstable_lower] = self.unstable_slope_lower, min * (1 - self.unstable_slope_lower)

        if self.unstable_upper is not None:
            max = self.module.max
            if torch.is_tensor(max):
                max = max.view(*[1 for _ in range(alpha_upper.dim() - 1)], -1).expand_as(alpha_upper)[self.unstable_upper]
            alpha_upper[self.unstable_upper], beta_upper[self.unstable_upper] = self.unstable_slope_upper, max * (1 - self.unstable_slope_upper)

        return alpha_lower, alpha_upper, beta_lower, beta_upper

    def bound_parameters(self):
        if self.unstable_lower is None and self.unstable_upper is None:
            logger.warning('Clamp bound not parameterized but expected to')

        if self.unstable_lower is not None:
            yield self.unstable_slope_lower

        if self.unstable_upper is not None:
            yield self.unstable_slope_upper

    def clip_params(self):
        if self.unstable_lower is not None:
            self.unstable_slope_lower.data.clamp_(min=0, max=1)

        if self.unstable_upper is not None:
            self.unstable_slope_upper.data.clamp_(min=0, max=1)
