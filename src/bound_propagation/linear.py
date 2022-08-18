from typing import Optional, Tuple

import torch

from .activation import assert_bound_order
from .general import BoundModule
from .bounds import LinearBounds, IntervalBounds


@torch.jit.script
def crown_backward_linear_jit(weight: torch.Tensor, bias: Optional[torch.Tensor], W_tilde: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if bias is None:
        bias_acc = torch.tensor(0.0, device=W_tilde.device)
    else:
        bias = bias.unsqueeze(-2)
        bias_acc = bias.matmul(W_tilde.transpose(-1, -2)).squeeze(-2)

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

    w_mid = center.matmul(weight) + (bias.unsqueeze(-2) if bias is not None else torch.tensor(0.0, device=weight.device))
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
    def ibp_forward(self, bounds, save_relaxation=False):
        center, diff = bounds.center, bounds.width / 2
        lower, upper = ibp_forward_linear_jit(self.module.weight, self.module.bias, center, diff)

        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        return self.module.weight.size(-2)
