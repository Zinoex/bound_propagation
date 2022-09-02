import abc
import logging

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from .bounds import LinearBounds, IntervalBounds


logger = logging.getLogger(__name__)


class BoundModule(nn.Module, abc.ABC):
    def __init__(self, module, factory, **kwargs):
        super().__init__()
        self.module = module

        self.alpha_optimizer = kwargs.get('alpha_optimizer', Adam)
        self.alpha_iterations = kwargs.get('alpha_iterations', 20)
        self.alpha_lr = kwargs.get('alpha_lr', 1.0)

    @torch.no_grad()
    def crown_relax(self, region):
        # Force bounds based on IBP, which may be tighter. More importantly, this also works if say there
        # is a Clamp in front of a Log which requires x > 0, and a linear relaxation may violate this
        interval_bounds = IntervalBounds(region, region.lower, region.upper)
        self.ibp_forward(interval_bounds, save_input_bounds=True)

        # No grad for relaxations improves accuracy and stabilizes training for CROWN.
        while self.need_relaxation:
            linear_bounds, module, *extra = self.backward_relaxation(region)
            module.set_relaxation(linear_bounds, *extra)

    @torch.no_grad()
    def ibp_relax(self, region):
        # No grad for relaxations improves accuracy. CROWN-IBP is already stable in training.
        bounds = IntervalBounds(region, region.lower, region.upper)
        self.ibp_forward(bounds, save_relaxation=True)

    def crown(self, region, bound_lower=True, bound_upper=True, alpha=False):
        return self.crown_with_relaxation(self.crown_relax, region, bound_lower, bound_upper, alpha)

    def crown_ibp(self, region, bound_lower=True, bound_upper=True, alpha=False):
        return self.crown_with_relaxation(self.ibp_relax, region, bound_lower, bound_upper, alpha)

    def crown_with_relaxation(self, relax, region, bound_lower, bound_upper, alpha):
        out_size = self.propagate_size(region.lower.size(-1))

        relax(region)

        if alpha:
            linear_bounds = self.alpha_crown(region, out_size, bound_lower, bound_upper)
        else:
            linear_bounds = self.initial_linear_bounds(region, out_size, lower=bound_lower, upper=bound_upper)
            linear_bounds = self.crown_backward(linear_bounds, False)

        self.clear_relaxation()
        return linear_bounds

    def initial_linear_bounds(self, region, out_size, lower=True, upper=True):
        device, dtype = region.lower.device, region.lower.dtype
        W_tilde = torch.eye(out_size, device=device, dtype=dtype).unsqueeze(-3).expand(*region.lower.size()[:-1], out_size, out_size)
        bias = torch.zeros((out_size,), device=device, dtype=dtype).unsqueeze(-2).expand(*region.lower.size()[:-1], out_size)

        lower = (W_tilde, bias) if lower else None
        upper = (W_tilde, bias) if upper else None

        linear_bounds = LinearBounds(region, lower, upper)
        return linear_bounds

    def alpha_crown(self, region, out_size, bound_lower, bound_upper):
        params = list(self.bound_parameters())

        if all([param.numel() == 0 for param in params]):
            logger.warning('No parameters available for alpha-CROWN. Check architecture of network.')

            linear_bounds = self.initial_linear_bounds(region, out_size, lower=bound_lower, upper=bound_upper)
            return self.crown_backward(linear_bounds, False)

        self.optimize_bounds(region, out_size, params, bound_lower=bound_lower, bound_upper=bound_upper)

        linear_bounds = self.initial_linear_bounds(region, out_size, lower=bound_lower, upper=bound_upper)
        return self.crown_backward(linear_bounds, True)

    @torch.enable_grad()
    def optimize_bounds(self, region, out_size, params, bound_lower=True, bound_upper=True):
        optimizer = self.alpha_optimizer(params, self.alpha_lr)

        for iteration in range(self.alpha_iterations):
            linear_bounds = self.initial_linear_bounds(region, out_size, lower=bound_lower, upper=bound_upper)
            linear_bounds = self.crown_backward(linear_bounds, True)
            interval_bounds = linear_bounds.concretize()

            loss = 0
            if bound_lower:
                # Use sum for aggregation because bound parameters are per sample, hence we need to scale the loss with
                # the number of samples
                loss = -interval_bounds.lower.sum()

            if bound_upper:
                loss = interval_bounds.upper.sum()

            optimizer.zero_grad(set_to_none=True)  # Bound parameters
            self.zero_grad(set_to_none=True)       # Network/model parameters

            loss.backward()
            optimizer.step()

            self.clip_params()  # Projected Gradient Descent

        optimizer.zero_grad(set_to_none=True)
        self.zero_grad(set_to_none=True)

    @property
    @abc.abstractmethod
    def need_relaxation(self):
        raise NotImplementedError()

    def clear_relaxation(self):
        pass

    def set_relaxation(self, linear_bounds, *args):
        pass

    def backward_relaxation(self, region):
        pass

    @abc.abstractmethod
    def crown_backward(self, linear_bounds, optimize):
        raise NotImplementedError()

    def ibp(self, region):
        bounds = IntervalBounds(region, region.lower, region.upper)
        return self.ibp_forward(bounds)

    @abc.abstractmethod
    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        raise NotImplementedError()

    @abc.abstractmethod
    def propagate_size(self, in_size):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        # Assume that submodules are constructed with the factory, since that produces
        # references down the bound module graph corresponding to the model graph.
        # Then loading the state dict at the top level automatically populates the weights in all sub-bound modules.
        self.module.load_state_dict(state_dict=state_dict, strict=strict)

    def bound_parameters(self):
        return []

    def clip_params(self):
        pass
