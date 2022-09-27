from typing import Optional, Tuple, Union

import torch
from torch import nn

from .activation import assert_bound_order
from .general import BoundModule
from .bounds import LinearBounds, IntervalBounds


@torch.jit.script
def crown_backward_linear_jit(weight: torch.Tensor, bias: Optional[torch.Tensor], bounds: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    W_tilde, bias_acc = bounds
    if bias is not None:
        bias_acc = bias_acc + bias.unsqueeze(-2).matmul(W_tilde.transpose(-1, -2)).squeeze(-2)

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
    weight = weight.transpose(-1, -2)

    w_mid = center.matmul(weight)
    if bias is not None:
        w_mid = w_mid + bias.unsqueeze(-2)
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
            lower = crown_backward_linear_jit(self.module.weight, self.module.bias, linear_bounds.lower)

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = crown_backward_linear_jit(self.module.weight, self.module.bias, linear_bounds.upper)

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
        self.register_buffer('weight', torch.as_tensor(weight))

        if bias is not None:
            del self.bias
            self.register_buffer('bias', torch.as_tensor(bias))


class ElementWiseLinear(nn.Module):
    def __init__(self, a, b=None):
        super().__init__()

        self.register_buffer('a', torch.as_tensor(a))

        if b is not None:
            b = torch.as_tensor(b)
        self.register_buffer('b', b)

    def forward(self, x):
        x = self.a * x
        if self.b is not None:
            x = x + self.b

        return x


def crown_backward_elementwise_linear_jit(a: torch.Tensor, b: Optional[Union[torch.Tensor, float]], bounds: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    W_tilde, bias_acc = bounds

    if b is not None:
        if b.dim() > 0:
            bias_acc = bias_acc + b.unsqueeze(-2).matmul(W_tilde.transpose(-1, -2)).squeeze(-2)
        else:
            bias_acc = bias_acc + b * W_tilde.sum(dim=-1)

    if a.dim() > 0:
        a = a.unsqueeze(-2)

    W_tilde = W_tilde * a

    return W_tilde, bias_acc


class BoundElementWiseLinear(BoundModule):
    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds, optimize):
        if linear_bounds.lower is None:
            lower = None
        else:
            lower = crown_backward_elementwise_linear_jit(self.module.a, self.module.b, linear_bounds.lower)

        if linear_bounds.upper is None:
            upper = None
        else:
            upper = crown_backward_elementwise_linear_jit(self.module.a, self.module.b, linear_bounds.upper)

        return LinearBounds(linear_bounds.region, lower, upper)

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        lower, upper = self(bounds.lower), self(bounds.upper)
        lower, upper = torch.min(lower, upper), torch.max(lower, upper)

        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        return in_size
