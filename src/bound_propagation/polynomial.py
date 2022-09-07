import logging
from typing import Tuple, List

import torch
from torch import nn

from .parallel import Parallel
from .reshape import Select
from .bivariate import Mul
from .activation import assert_bound_order, regimes, bisection, BoundActivation
from .bounds import IntervalBounds, LinearBounds
from .general import BoundModule
from .util import clip_param_to_range_, proj_grad_to_range_

logger = logging.getLogger(__name__)


class Pow(nn.Module):
    def __init__(self, power):
        super().__init__()

        # assert power >= 1, 'Pow only supports integer powers'
        # assert all([isinstance(power, int) and power >= 2 for power in self.powers]), 'Univariate monomial only supports integer powers'

        self.power = power

    def forward(self, x):
        return x.pow(self.power)


class BoundPow(BoundActivation):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    def clear_relaxation(self):
        super().clear_relaxation()
        self.unstable_lower, self.unstable_d_lower, self.unstable_range_lower = None, None, None
        self.unstable_upper, self.unstable_d_upper, self.unstable_range_upper = None, None, None

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.alpha_beta(preactivation=bounds)
            self.bounded = True

        if save_input_bounds:
            self.input_bounds = bounds

        lower, upper = bounds.lower, bounds.upper
        lower_act, upper_act = self(lower), self(upper)

        lower_out, upper_out = torch.zeros_like(lower), torch.zeros_like(upper)

        # Even powers
        even = (torch.as_tensor(self.module.power, device=lower.device) % 2) == 0
        crossing = (lower < 0) & (upper > 0)

        crossing, not_crossing = (crossing & even), ((~crossing) & even)

        lower_out[crossing] = torch.zeros_like(lower[crossing])
        lower_out[not_crossing] = torch.min(lower_act[not_crossing], upper_act[not_crossing])

        upper_out[..., even] = torch.max(lower_act[..., even], upper_act[..., even])

        # Odd powers
        lower_out[..., ~even] = lower_act[..., ~even]
        upper_out[..., ~even] = upper_act[..., ~even]

        return IntervalBounds(bounds.region, lower_out, upper_out)

    def func(self, x, power=None):
        if power is None:
            power = torch.as_tensor(self.module.power, device=x.device)

        x = x.pow(power)
        return x

    def derivative(self, x, power=None):
        if power is None:
            power = torch.as_tensor(self.module.power, device=x.device)

        x = power * x.pow(power - 1)
        return x

    @assert_bound_order
    def alpha_beta(self, preactivation):
        lower, upper = preactivation.lower, preactivation.upper
        zero_width, n, p, np = regimes(lower, upper)
        power = torch.as_tensor(self.module.power, device=lower.device)
        even = power % 2 == 0
        odd = ~even

        self.alpha_lower, self.beta_lower = torch.zeros_like(lower), torch.zeros_like(lower)
        self.alpha_upper, self.beta_upper = torch.zeros_like(lower), torch.zeros_like(lower)

        lower_act, upper_act = self.func(lower), self.func(upper)
        lower_prime, upper_prime = self.derivative(lower), self.derivative(upper)

        d = (lower + upper) * 0.5  # Let d be the midpoint of the two bounds
        d_act = self.func(d)
        d_prime = self.derivative(d)

        slope = (upper_act - lower_act) / (upper - lower)

        def add_linear(alpha, beta, mask, a, x, y, a_mask=True):
            if a_mask:
                a = a[mask]

            alpha[mask] = a
            beta[mask] = y[mask] - a * x[mask]

        # Use upper and lower in the bias to account for a small numerical difference between lower and upper
        # which ought to be negligible, but may still be present due to torch.isclose.
        self.alpha_lower[zero_width], self.beta_lower[zero_width] = 0, torch.min(lower_act[zero_width], upper_act[zero_width])
        self.alpha_upper[zero_width], self.beta_upper[zero_width] = 0, torch.max(lower_act[zero_width], upper_act[zero_width])

        ########
        # Even #
        ########
        all_even = (n | p | np) & even

        add_linear(self.alpha_lower, self.beta_lower, mask=all_even, a=d_prime, x=d, y=d_act)
        add_linear(self.alpha_upper, self.beta_upper, mask=all_even, a=slope, x=lower, y=lower_act)

        # Allow parameterization
        # Save mask
        unstable_lower = [all_even]
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        unstable_d_lower = [d[all_even].detach().clone().requires_grad_()]
        # Save ranges to clip (aka. PGD)
        unstable_range_lower = [(lower[all_even], upper[all_even])]

        #########################
        # Odd - negative regime #
        #########################
        n_odd = n & odd

        add_linear(self.alpha_lower, self.beta_lower, mask=n_odd, a=slope, x=lower, y=lower_act)
        add_linear(self.alpha_upper, self.beta_upper, mask=n_odd, a=d_prime, x=d, y=d_act)

        # Allow parameterization
        # Save mask
        self.unstable_upper = n_odd
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        self.unstable_d_upper = d[n_odd].detach().clone().requires_grad_()
        # Save ranges to clip (aka. PGD)
        self.unstable_range_upper = lower[n_odd], upper[n_odd]

        #########################
        # Odd - positive regime #
        #########################
        p_odd = p & odd

        add_linear(self.alpha_lower, self.beta_lower, mask=p_odd, a=d_prime, x=d, y=d_act)
        add_linear(self.alpha_upper, self.beta_upper, mask=p_odd, a=slope, x=lower, y=lower_act)

        # Allow parameterization
        # Save mask
        unstable_lower.append(p_odd)
        # Optimization variables - detach, clone, and require grad to perform back prop and optimization
        unstable_d_lower.append(d[p_odd].detach().clone().requires_grad_())
        # Save ranges to clip (aka. PGD)
        unstable_range_lower.append((lower[p_odd], upper[p_odd]))

        self.unstable_lower = unstable_lower
        self.unstable_d_lower = unstable_d_lower
        self.unstable_range_lower = unstable_range_lower

        #######################
        # Odd - crossing zero #
        #######################
        np_odd = np & odd
        power = power.view(*[1 for _ in range(np.dim() - 1)], -1).expand_as(np_odd)

        # Upper bound #
        # If tangent to lower is below upper, then take direct slope between lower and upper
        direct = np_odd & (slope >= lower_prime)
        add_linear(self.alpha_upper, self.beta_upper, mask=direct, a=slope, x=upper, y=upper_act)

        # Else use bisection to find upper bound on slope.
        implicit = np_odd & (slope < lower_prime)

        if torch.any(implicit):
            implicit_power = power[implicit]
            implicit_lower, implicit_upper = lower[implicit], upper[implicit]
            implicit_upper_act = self.func(implicit_upper, implicit_power)

            def f_upper(d: torch.Tensor) -> torch.Tensor:
                a_slope = (implicit_upper_act - self.func(d, implicit_power)) / (implicit_upper - d)
                a_derivative = self.derivative(d, implicit_power)
                return a_slope - a_derivative

            # Bisection will return left and right bounds for d s.t. f_upper(d) is zero
            # Derivative of left bound will over-approximate the slope - hence a true bound
            d_upper, _ = bisection(implicit_lower, torch.zeros_like(implicit_lower), f_upper)
            # Slope has to attach to (lower, sigma(lower))
            add_linear(self.alpha_upper, self.beta_upper, mask=implicit, a=self.derivative(d_upper, implicit_power), x=upper, y=upper_act, a_mask=False)

        # Lower bound #
        # If tangent to upper is above lower, then take direct slope between lower and upper
        direct = np_odd & (slope >= upper_prime)
        add_linear(self.alpha_lower, self.beta_lower, mask=direct, a=slope, x=lower, y=lower_act)

        # Else use bisection to find upper bound on slope.
        implicit = np_odd & (slope < upper_prime)

        if torch.any(implicit):
            implicit_power = power[implicit]
            implicit_lower, implicit_upper = lower[implicit], upper[implicit]
            implicit_lower_act = self.func(implicit_lower, implicit_power)

            def f_lower(d: torch.Tensor) -> torch.Tensor:
                a_slope = (self.func(d, implicit_power) - implicit_lower_act) / (d - implicit_lower)
                a_derivative = self.derivative(d, implicit_power)
                return a_derivative - a_slope

            # Bisection will return left and right bounds for d s.t. f_lower(d) is zero
            # Derivative of right bound will over-approximate the slope - hence a true bound
            _, d_lower = bisection(torch.zeros_like(implicit_upper), implicit_upper, f_lower)
            # Slope has to attach to (upper, sigma(upper))
            add_linear(self.alpha_lower, self.beta_lower, mask=implicit, a=self.derivative(d_lower, implicit_power), x=lower, y=lower_act, a_mask=False)

    def parameterize_alpha_beta(self, alpha_lower, alpha_upper, beta_lower, beta_upper):
        if self.unstable_lower is None or self.unstable_upper is None:
            logger.warning('Polynomial bound not parameterized but expected to')

        power = torch.as_tensor(self.module.power, device=alpha_lower.device).view(*[1 for _ in range(alpha_lower.dim() - 1)], -1).expand_as(alpha_lower)

        # Use implicit parameterization (i.e. store d [point where touching the curve], and not alpha)
        def add_linear(alpha, beta, mask, x):
            p = power[mask]

            a = self.derivative(x, p)
            y = self.func(x, p)

            alpha[mask] = a
            beta[mask] = y - a * x

        add_linear(alpha_lower, beta_lower, mask=self.unstable_lower[0], x=self.unstable_d_lower[0])
        add_linear(alpha_lower, beta_lower, mask=self.unstable_lower[1], x=self.unstable_d_lower[1])
        add_linear(alpha_upper, beta_upper, mask=self.unstable_upper, x=self.unstable_d_upper)

        return alpha_lower, alpha_upper, beta_lower, beta_upper

    def bound_parameters(self):
        if self.unstable_lower is None or self.unstable_upper is None:
            logger.warning('Polynomial bound not parameterized but expected to')

        yield from self.unstable_d_lower
        yield self.unstable_d_upper

    def clip_params(self):
        clip_param_to_range_(self.unstable_d_lower[0], self.unstable_range_lower[0])
        clip_param_to_range_(self.unstable_d_lower[1], self.unstable_range_lower[1])
        clip_param_to_range_(self.unstable_d_upper, self.unstable_range_upper)

    def project_grads(self):
        proj_grad_to_range_(self.unstable_d_lower[0], self.unstable_range_lower[0])
        proj_grad_to_range_(self.unstable_d_lower[1], self.unstable_range_lower[1])
        proj_grad_to_range_(self.unstable_d_upper, self.unstable_range_upper)


class UnivariateMonomial(nn.Sequential):
    def __init__(self, powers):
        # Assume powers are of the structure [(index, power)] and this will be the output order too
        indices = [index for index, _ in powers]
        powers = [powers for _, powers in powers]

        super().__init__(
            Select(indices),
            Pow(powers)
        )


class MultivariateMonomial(nn.Sequential):
    def __init__(self, monomials):
        super().__init__()

        monomials = [self.coalesce_factors(monomial) for monomial in monomials]

        # Assume powers are of the structure [[(index, power)]]
        linear_terms = list(set([index for monomial in monomials for index, power in monomial if power == 1]))
        factors = list(set([(index, power) for monomial in monomials for index, power in monomial if power >= 2]))
        monomial_indices = []

        for monomial in monomials:
            monomial_index = []
            for index, power in monomial:
                if power == 1:
                    monomial_index.append(linear_terms.index(index))
                else:
                    assert isinstance(power, int) and power >= 2, 'Multivariate monomial only supports integer powers'

                    monomial_index.append(len(linear_terms) + factors.index((index, power)))

            monomial_indices.append(monomial_index)

        super().__init__(
            Parallel(Select(linear_terms), UnivariateMonomial(factors)),
            Parallel(*[self.construct_mul(monomial_index) for monomial_index in monomial_indices])
        )

    def construct_mul(self, monomial_index):
        module = Select(monomial_index[0])

        for index in monomial_index[1:]:
            module = Mul(module, Select(index))

        return module

    def coalesce_factors(self, monomial):
        powers = {}

        for index, power in monomial:
            powers[index] = powers.get(index, 0) + power

        return list(powers.items())


class Polynomials(nn.Module):
    pass

    # Combine Linear and MultivariateMonomial to form Polynomials
