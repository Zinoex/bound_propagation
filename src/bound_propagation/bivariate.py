import logging
from typing import Tuple

import torch
from torch import nn
from torch.nn import Identity

from . import Reciprocal
from .general import BoundModule
from .bounds import IntervalBounds, LinearBounds


logger = logging.getLogger(__name__)


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

    def crown_backward(self, linear_bounds, optimize):
        linear_bounds1 = self.bound_network1.crown_backward(linear_bounds, optimize)
        linear_bounds2 = self.bound_network2.crown_backward(linear_bounds, optimize)

        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (linear_bounds1.lower[0] + linear_bounds2.lower[0], linear_bounds1.lower[1] + linear_bounds2.lower[1] - linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (linear_bounds1.upper[0] + linear_bounds2.upper[0], linear_bounds1.upper[1] + linear_bounds2.upper[1] - linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        bounds1 = self.bound_network1.ibp_forward(bounds, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)
        bounds2 = self.bound_network2.ibp_forward(bounds, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)

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

    def bound_parameters(self):
        yield from self.bound_network1.bound_parameters()
        yield from self.bound_network2.bound_parameters()

    def clip_params(self):
        self.bound_network1.clip_params()
        self.bound_network2.clip_params()


class VectorAdd(nn.Module):
    def forward(self, x):
        assert x.size(-1) % 2 == 0
        half_size = x.size(-1) // 2

        return x[..., :half_size] + x[..., half_size:]


class BoundVectorAdd(BoundModule):
    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds, optimize):
        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (torch.cat(2 * [linear_bounds.lower[0]], dim=-1), linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (torch.cat(2 * [linear_bounds.upper[0]], dim=-1), linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        half_size = bounds.lower.size(-1) // 2

        return IntervalBounds(
            bounds.region,
            bounds.lower[..., :half_size] + bounds.lower[..., half_size:],
            bounds.upper[..., :half_size] + bounds.upper[..., half_size:]
        )

    def propagate_size(self, in_size):
        assert in_size % 2 == 0

        return in_size // 2


class Residual(Add):
    def __init__(self, subnetwork):
        super().__init__(Identity(), subnetwork)


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

    def crown_backward(self, linear_bounds, optimize):
        input_bounds = LinearBounds(
            linear_bounds.region,
            (linear_bounds.lower[0], torch.zeros_like(linear_bounds.lower[1])) if linear_bounds.lower is not None else None,
            (linear_bounds.upper[0], torch.zeros_like(linear_bounds.upper[1])) if linear_bounds.upper is not None else None,
        )
        linear_bounds1 = self.bound_network1.crown_backward(input_bounds, optimize)

        input_bounds = LinearBounds(
            linear_bounds.region,
            (-linear_bounds.lower[0], torch.zeros_like(linear_bounds.lower[1])) if linear_bounds.lower is not None else None,
            (-linear_bounds.upper[0], torch.zeros_like(linear_bounds.upper[1])) if linear_bounds.upper is not None else None,
        )
        linear_bounds2 = self.bound_network2.crown_backward(input_bounds, optimize)

        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (linear_bounds1.lower[0] + linear_bounds2.lower[0], linear_bounds1.lower[1] + linear_bounds2.lower[1] + linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (linear_bounds1.upper[0] + linear_bounds2.upper[0], linear_bounds1.upper[1] + linear_bounds2.upper[1] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        bounds1 = self.bound_network1.ibp_forward(bounds, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)
        bounds2 = self.bound_network2.ibp_forward(bounds, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)

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

    def bound_parameters(self):
        yield from self.bound_network1.bound_parameters()
        yield from self.bound_network2.bound_parameters()

    def clip_params(self):
        self.bound_network1.clip_params()
        self.bound_network2.clip_params()


class VectorSub(nn.Module):
    def forward(self, x):
        assert x.size(-1) % 2 == 0
        half_size = x.size(-1) // 2

        return x[..., :half_size] - x[..., half_size:]


class BoundVectorSub(BoundModule):
    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds, optimize):
        if linear_bounds.lower is None:
            lower = None
        else:
            lower = (torch.cat([linear_bounds.lower[0], -linear_bounds.lower[0]], dim=-1), linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = (torch.cat([linear_bounds.upper[0], -linear_bounds.upper[0]], dim=-1), linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        half_size = bounds.lower.size(-1) // 2

        return IntervalBounds(
            bounds.region,
            bounds.lower[..., :half_size] - bounds.upper[..., half_size:],
            bounds.upper[..., :half_size] - bounds.lower[..., half_size:]
        )

    def propagate_size(self, in_size):
        assert in_size % 2 == 0

        return in_size // 2


class VectorMul(nn.Module):
    def forward(self, x):
        assert x.size(-1) % 2 == 0
        half_size = x.size(-1) // 2

        return x[..., :half_size] * x[..., half_size:]


@torch.jit.script
def crown_backward_mul_jit(W_tilde: torch.Tensor, alpha_x: Tuple[torch.Tensor, torch.Tensor], alpha_y: Tuple[torch.Tensor, torch.Tensor], beta: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _delta = torch.where(W_tilde < 0, beta[0].unsqueeze(-2), beta[1].unsqueeze(-2))
    bias = torch.sum(W_tilde * _delta, dim=-1)

    _lambda = torch.where(W_tilde < 0, alpha_x[0].unsqueeze(-2), alpha_x[1].unsqueeze(-2))
    W_tilde_x = W_tilde * _lambda

    _lambda = torch.where(W_tilde < 0, alpha_y[0].unsqueeze(-2), alpha_y[1].unsqueeze(-2))
    W_tilde_y = W_tilde * _lambda

    return W_tilde_x, W_tilde_y, bias


class Mul(nn.Module):
    def __init__(self, network1, network2):
        super().__init__()

        self.network1, self.network2 = network1, network2

    def forward(self, x):
        return self.network1(x) * self.network2(x)


class BoundMul(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.bound_network1 = factory.build(module.network1)
        self.bound_network2 = factory.build(module.network2)

        self.input_bounds = None
        self.kappa_lower, self.kappa_upper = None, None

        self.bounded = False
        self.size = None

    def alpha_beta(self, optimize):
        lower_x, upper_x = self.input_bounds[0].lower, self.input_bounds[0].upper
        lower_y, upper_y = self.input_bounds[1].lower, self.input_bounds[1].upper

        # Lower bound
        first_corner = lower_x, lower_y
        second_corner = upper_x, upper_y

        kappa_lower = self.kappa_lower if optimize else 0.5

        d = first_corner[0] * kappa_lower + second_corner[0] * (1 - kappa_lower), first_corner[1] * kappa_lower + second_corner[1] * (1 - kappa_lower)
        d_prime = d[1], d[0]

        alpha_x_lower = d_prime[0]
        alpha_y_lower = d_prime[1]

        beta_corner = lower_x, upper_y
        beta_lower = beta_corner[0] * beta_corner[1] - d_prime[0] * beta_corner[0] - d_prime[1] * beta_corner[1]

        # Upper bound
        first_corner = lower_x, upper_y
        second_corner = upper_x, lower_y

        kappa_upper = self.kappa_upper if optimize else 0.5

        d = first_corner[0] * kappa_upper + second_corner[0] * (1 - kappa_upper), first_corner[1] * kappa_upper + second_corner[1] * (1 - kappa_upper)
        d_prime = d[1], d[0]

        alpha_x_upper = d_prime[0]
        alpha_y_upper = d_prime[1]

        beta_corner = lower_x, lower_y
        beta_upper = beta_corner[0] * beta_corner[1] - d_prime[0] * beta_corner[0] - d_prime[1] * beta_corner[1]

        return (alpha_x_lower, alpha_x_upper), (alpha_y_lower, alpha_y_upper), (beta_lower, beta_upper)

    def init_kappa(self):
        self.kappa_lower = torch.full_like(self.input_bounds[0].lower, 0.5).requires_grad_()
        self.kappa_upper = torch.full_like(self.input_bounds[0].lower, 0.5).requires_grad_()

    @property
    def need_relaxation(self):
        return self.bound_network1.need_relaxation or self.bound_network2.need_relaxation or not self.bounded

    def set_relaxation(self, linear_bounds):
        interval_bounds = linear_bounds.concretize()
        lower_x, lower_y = interval_bounds.lower.tensor_split(2, dim=-1)
        upper_x, upper_y = interval_bounds.upper.tensor_split(2, dim=-1)

        self.input_bounds = IntervalBounds(linear_bounds.region,
                                           torch.max(lower_x, self.input_bounds[0].lower),
                                           torch.min(upper_x, self.input_bounds[0].upper)),\
                            IntervalBounds(linear_bounds.region,
                                           torch.max(lower_y, self.input_bounds[1].lower),
                                           torch.min(upper_y, self.input_bounds[1].upper))

        self.init_kappa()
        self.bounded = True

    def backward_relaxation(self, region):
        if self.bound_network1.need_relaxation:
            return self.bound_network1.backward_relaxation(region)
        elif self.bound_network2.need_relaxation:
            return self.bound_network2.backward_relaxation(region)
        else:
            assert self.size is not None

            linear_bounds = self.initial_linear_bounds(region, self.size)
            linear_bounds1 = self.bound_network1.crown_backward(linear_bounds, False)
            linear_bounds2 = self.bound_network2.crown_backward(linear_bounds, False)

            lower = (torch.cat([linear_bounds1.lower[0], linear_bounds2.lower[0]], dim=-2), torch.cat([linear_bounds1.lower[1], linear_bounds2.lower[1]], dim=-1))
            upper = (torch.cat([linear_bounds1.upper[0], linear_bounds2.upper[0]], dim=-2), torch.cat([linear_bounds1.upper[1], linear_bounds2.upper[1]], dim=-1))

            return LinearBounds(linear_bounds.region, lower, upper), self

    def clear_relaxation(self):
        self.input_bounds = None
        self.kappa_lower, self.kappa_upper = None, None

        self.bounded = False

    def crown_backward(self, linear_bounds, optimize):
        assert self.bounded

        (alpha_x_lower, alpha_x_upper), (alpha_y_lower, alpha_y_upper), (beta_lower, beta_upper) = self.alpha_beta(optimize)

        # NOTE: The order of alpha and beta are deliberately reverse - this is not a mistake!
        if linear_bounds.lower is None:
            lower = None
        else:
            alpha_x = alpha_x_upper, alpha_x_lower
            alpha_y = alpha_y_upper, alpha_y_lower
            beta = beta_upper, beta_lower

            lower = crown_backward_mul_jit(linear_bounds.lower[0], alpha_x, alpha_y, beta)

        if linear_bounds.upper is None:
            upper = None
        else:
            alpha_x = alpha_x_lower, alpha_x_upper
            alpha_y = alpha_y_lower, alpha_y_upper
            beta = beta_lower, beta_upper

            upper = crown_backward_mul_jit(linear_bounds.upper[0], alpha_x, alpha_y, beta)

        input_bounds1 = LinearBounds(linear_bounds.region,
                                     (lower[0], torch.zeros_like(lower[2])) if lower is not None else None,
                                     (upper[0], torch.zeros_like(upper[2])) if upper is not None else None)
        input_bounds2 = LinearBounds(linear_bounds.region,
                                     (lower[1], torch.zeros_like(lower[2])) if lower is not None else None,
                                     (upper[1], torch.zeros_like(upper[2])) if upper is not None else None)

        linear_bounds1 = self.bound_network1.crown_backward(input_bounds1, optimize)
        linear_bounds2 = self.bound_network2.crown_backward(input_bounds2, optimize)

        if lower is None:
            lower = None
        else:
            lower = (linear_bounds1.lower[0] + linear_bounds2.lower[0], linear_bounds1.lower[1] + linear_bounds2.lower[1] + lower[2] + linear_bounds.lower[1])

        if upper is None:
            upper = None
        else:
            upper = (linear_bounds1.upper[0] + linear_bounds2.upper[0], linear_bounds1.upper[1] + linear_bounds2.upper[1] + upper[2] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        bounds1 = self.bound_network1.ibp_forward(bounds, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)
        bounds2 = self.bound_network2.ibp_forward(bounds, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)

        if save_relaxation:
            self.input_bounds = bounds1, bounds2
            self.init_kappa()
            self.bounded = True

        if save_input_bounds:
            self.input_bounds = bounds1, bounds2

        combinations = torch.stack([
            bounds1.lower * bounds2.lower,
            bounds1.lower * bounds2.upper,
            bounds1.upper * bounds2.lower,
            bounds1.upper * bounds2.upper
        ], dim=-1)

        return IntervalBounds(
            bounds.region,
            combinations.min(dim=-1).values,
            combinations.max(dim=-1).values
        )

    def propagate_size(self, in_size):
        out_size1 = self.bound_network1.propagate_size(in_size)
        out_size2 = self.bound_network2.propagate_size(in_size)

        assert out_size1 == out_size2

        self.size = out_size1
        return out_size1

    def bound_parameters(self):
        if self.kappa_lower is None or self.kappa_upper is None:
            logger.warning('VectorMul bound not parameterized but expected to')

        yield self.kappa_lower
        yield self.kappa_upper

    def clip_params(self):
        self.kappa_lower.data.clamp_(min=0.0, max=1.0)
        self.kappa_upper.data.clamp_(min=0.0, max=1.0)


class Div(Mul):
    def __init__(self, network1, network2):
        super().__init__(network1, nn.Sequential(network2, Reciprocal()))


class BoundVectorMul(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory, **kwargs)

        self.input_bounds = None
        self.kappa_lower, self.kappa_upper = None, None

        self.bounded = False
        self.size = None

    def alpha_beta(self, optimize):
        preactivation = self.input_bounds
        half_size = preactivation.lower.size(-1) // 2

        lower_x, upper_x = preactivation.lower[..., :half_size], preactivation.upper[..., :half_size]
        lower_y, upper_y = preactivation.lower[..., half_size:], preactivation.upper[..., half_size:]

        # Lower bound
        first_corner = lower_x, lower_y
        second_corner = upper_x, upper_y

        kappa_lower = self.kappa_lower if optimize else 0.5

        d = first_corner[0] * kappa_lower + second_corner[0] * (1 - kappa_lower), first_corner[1] * kappa_lower + second_corner[1] * (1 - kappa_lower)
        d_prime = d[1], d[0]

        alpha_x_lower = d_prime[0]
        alpha_y_lower = d_prime[1]

        beta_corner = lower_x, upper_y
        beta_lower = beta_corner[0] * beta_corner[1] - d_prime[0] * beta_corner[0] - d_prime[1] * beta_corner[1]

        # Upper bound
        first_corner = lower_x, upper_y
        second_corner = upper_x, lower_y

        kappa_upper = self.kappa_upper if optimize else 0.5

        d = first_corner[0] * kappa_upper + second_corner[0] * (1 - kappa_upper), first_corner[1] * kappa_upper + second_corner[1] * (1 - kappa_upper)
        d_prime = d[1], d[0]

        alpha_x_upper = d_prime[0]
        alpha_y_upper = d_prime[1]

        beta_corner = lower_x, lower_y
        beta_upper = beta_corner[0] * beta_corner[1] - d_prime[0] * beta_corner[0] - d_prime[1] * beta_corner[1]

        return (alpha_x_lower, alpha_x_upper), (alpha_y_lower, alpha_y_upper), (beta_lower, beta_upper)

    def init_kappa(self):
        half_size = self.input_bounds.lower.size(-1) // 2
        lower_x = self.input_bounds.lower[..., :half_size]

        self.kappa_lower = torch.full_like(lower_x, 0.5).requires_grad_()
        self.kappa_upper = torch.full_like(lower_x, 0.5).requires_grad_()

    @property
    def need_relaxation(self):
        return not self.bounded

    def set_relaxation(self, linear_bounds):
        interval_bounds = linear_bounds.concretize()
        self.input_bounds = IntervalBounds(
            linear_bounds.region,
            torch.max(interval_bounds.lower, self.input_bounds.lower),
            torch.min(interval_bounds.upper, self.input_bounds.upper)
        )

        self.init_kappa()
        self.bounded = True

    def backward_relaxation(self, region):
        assert self.size is not None

        linear_bounds = self.initial_linear_bounds(region, self.size)
        return linear_bounds, self

    def clear_relaxation(self):
        self.input_bounds = None
        self.kappa_lower, self.kappa_upper = None, None

        self.bounded = False

    def crown_backward(self, linear_bounds, optimize):
        assert self.bounded

        (alpha_x_lower, alpha_x_upper), (alpha_y_lower, alpha_y_upper), (beta_lower, beta_upper) = self.alpha_beta(optimize)

        # NOTE: The order of alpha and beta are deliberately reverse - this is not a mistake!
        if linear_bounds.lower is None:
            lower = None
        else:
            alpha_x = alpha_x_upper, alpha_x_lower
            alpha_y = alpha_y_upper, alpha_y_lower
            beta = beta_upper, beta_lower

            lower = crown_backward_mul_jit(linear_bounds.lower[0], alpha_x, alpha_y, beta)
            lower = (torch.cat(lower[:2], dim=-1), lower[2] + linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            alpha_x = alpha_x_lower, alpha_x_upper
            alpha_y = alpha_y_lower, alpha_y_upper
            beta = beta_lower, beta_upper

            upper = crown_backward_mul_jit(linear_bounds.upper[0], alpha_x, alpha_y, beta)
            upper = (torch.cat(upper[:2], dim=-1), upper[2] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        if save_relaxation:
            self.input_bounds = bounds
            self.init_kappa()
            self.bounded = True

        if save_input_bounds:
            self.input_bounds = bounds

        half_size = bounds.lower.size(-1) // 2

        lower_x, upper_x = bounds.lower[..., :half_size], bounds.upper[..., :half_size]
        lower_y, upper_y = bounds.lower[..., half_size:], bounds.upper[..., half_size:]

        combinations = torch.stack([
            lower_x * lower_y,
            lower_x * upper_y,
            upper_x * lower_y,
            upper_x * upper_y
        ], dim=-1)

        return IntervalBounds(
            bounds.region,
            combinations.min(dim=-1).values,
            combinations.max(dim=-1).values
        )

    def propagate_size(self, in_size):
        assert in_size % 2 == 0

        self.size = in_size
        return in_size // 2

    def bound_parameters(self):
        if self.kappa_lower is None or self.kappa_upper is None:
            logger.warning('VectorMul bound not parameterized but expected to')

        yield self.kappa_lower
        yield self.kappa_upper

    def clip_params(self):
        self.kappa_lower.data.clamp_(min=0.0, max=1.0)
        self.kappa_upper.data.clamp_(min=0.0, max=1.0)
