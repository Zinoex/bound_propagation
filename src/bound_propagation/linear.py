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


class BoundLinear(BoundModule):

    @property
    def need_relaxation(self):
        return False

    def crown_backward(self, linear_bounds):
        lower = crown_backward_linear_jit(self.module.weight, self.module.bias, linear_bounds.lower[0])
        new_lower = (lower[0], lower[1] + linear_bounds.lower[1])

        upper = crown_backward_linear_jit(self.module.weight, self.module.bias, linear_bounds.upper[0])
        new_upper = (upper[0], upper[1] + linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, new_lower, new_upper)

    def crown_ibp(self, region):
        raise NotImplementedError()

    @assert_bound_order
    def ibp_forward(self, bounds, save_relaxation=False):
        center, diff = (bounds.lower + bounds.upper) / 2, (bounds.upper - bounds.lower) / 2
        center, diff = center.unsqueeze(-2), diff.unsqueeze(-2)

        weight = self.module.weight.transpose(-1, -2)

        w_mid = center.matmul(weight) + (self.module.bias.unsqueeze(-2) if self.module.bias is not None else torch.tensor(0.0, device=weight.device))
        w_diff = diff.matmul(weight.abs())

        lower = w_mid - w_diff
        lower = lower.squeeze(-2)

        upper = w_mid + w_diff
        upper = upper.squeeze(-2)

        return IntervalBounds(bounds.region, lower, upper)

    def propagate_size(self, in_size):
        return self.module.weight.size(-2)
