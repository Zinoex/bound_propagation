from typing import Optional, Tuple, Union

import torch
from torch import nn

from .activation import assert_bound_order
from .general import BoundModule
from .bounds import LinearBounds, IntervalBounds


@torch.jit.script
def crown_backward_linear_jit(weight: torch.Tensor, bias: Optional[torch.Tensor], W_tilde: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if bias is None:
        bias_acc = torch.tensor(0.0, device=W_tilde.device, dtype=W_tilde.dtype)
    else:
        bias = bias.unsqueeze(-2).to(W_tilde.dtype)
        bias_acc = bias.matmul(W_tilde.transpose(-1, -2)).squeeze(-2)

    weight = weight.to(W_tilde.dtype)
    if weight.dim() == 2:
        W_tilde = W_tilde.matmul(weight)
    else:
        if W_tilde.dim() == 3:
            W_tilde = W_tilde.unsqueeze(0)

        W_tilde = W_tilde.matmul(weight.unsqueeze(1))

    return W_tilde, bias_acc


@torch.jit.script
def ibp_forward_linear_jit(weight: torch.Tensor, bias: Optional[torch.Tensor], center: torch.Tensor, diff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    center, diff = center.unsqueeze(-2), diff.unsqueeze(-2)
    device, dtype = center.device, center.dtype

    weight = weight.transpose(-1, -2).to(dtype)

    w_mid = center.matmul(weight) + (bias.to(dtype).unsqueeze(-2) if bias is not None else torch.tensor(0.0, device=device, dtype=dtype))
    w_diff = diff.matmul(weight.abs())

    lower = w_mid - w_diff
    lower = lower.squeeze(-2)

    upper = w_mid + w_diff
    upper = upper.squeeze(-2)

    return lower, upper


class BoundLinear(BoundModule):

    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds, optimize):
        if linear_bounds.lower is None:
            lower = None
        else:
            lower = crown_backward_linear_jit(self.module.weight, self.module.bias, linear_bounds.lower[0])
            lower = (lower[0], lower[1] + linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = crown_backward_linear_jit(self.module.weight, self.module.bias, linear_bounds.upper[0])
            upper = (upper[0], upper[1] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        center, diff = bounds.center, bounds.width / 2
        lower, upper = ibp_forward_linear_jit(self.module.weight, self.module.bias, center, diff)

        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        return self.module.weight.size(-2)


class FixedLinear(nn.Linear):
    def __init__(self, weight, bias=None):
        super().__init__(weight.size(-1), weight.size(-2), bias=bias is not None)

        del self.weight
        self.register_buffer('weight', weight)

        if bias is not None:
            del self.bias
            self.register_buffer('bias', bias)


class ElementWiseLinear(nn.Module):
    def __init__(self, a, b=None):
        super().__init__()

        self.a = a
        self.b = b

    def forward(self, x):
        x = self.a * x
        if self.b is not None:
            x = x + self.b

        return x


def crown_backward_elementwise_linear_jit(a: torch.Tensor, b: Optional[Union[torch.Tensor, float]], W_tilde: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if b is None:
        b = torch.tensor(0.0, device=W_tilde.device, dtype=W_tilde.dtype)
    elif isinstance(b, torch.Tensor):
        b = b.unsqueeze(-2).matmul(W_tilde.transpose(-1, -2)).squeeze(-2)
    else:
        b = b * W_tilde.sum(dim=-1)

    W_tilde = W_tilde * a

    return W_tilde, b


class BoundElementWiseLinear(BoundModule):

    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds, optimize):
        if linear_bounds.lower is None:
            lower = None
        else:
            lower = crown_backward_elementwise_linear_jit(self.module.a, self.module.b, linear_bounds.lower[0])
            lower = (lower[0], lower[1] + linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = crown_backward_elementwise_linear_jit(self.module.a, self.module.b, linear_bounds.upper[0])
            upper = (upper[0], upper[1] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        lower, upper = self(bounds.lower), self(bounds.upper)
        lower, upper = torch.min(lower, upper), torch.max(lower, upper)

        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        return in_size
