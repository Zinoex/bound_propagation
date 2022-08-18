import abc

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from .bounds import LinearBounds, IntervalBounds


class BoundModule(nn.Module, abc.ABC):
    def __init__(self, module, factory, **kwargs):
        super().__init__()
        self.module = module

        self.optimizer = kwargs.get('bounds_optimizer', Adam)
        self.bounds_iterations = kwargs.get('bounds_iterations', 40)
        self.lr = kwargs.get('bounds_lr', 1e-2)
        self.lr_decay = kwargs.get('bounds_lr_decay', 0.97)

    @torch.no_grad()
    def crown_relax(self, region):
        # No grad for relaxations improves accuracy and stabilizes training for CROWN.
        while self.need_relaxation:
            linear_bounds, module = self.backward_relaxation(region)
            module.set_relaxation(linear_bounds)

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
        W_tilde = torch.eye(out_size, device=region.lower.device)\
            .unsqueeze(-3).expand(*region.lower.size()[:-1], out_size, out_size)
        bias = torch.zeros((out_size,), device=region.lower.device)\
            .unsqueeze(-2).expand(*region.lower.size()[:-1], out_size)

        lower = (W_tilde, bias) if lower else None
        upper = (W_tilde, bias) if upper else None

        linear_bounds = LinearBounds(region, lower, upper)
        return linear_bounds

    @torch.enable_grad()
    def alpha_crown(self, region, out_size, bound_lower, bound_upper):
        params = list(self.bound_parameters())

        if bound_lower:
            lower = self.optimize_bounds(region, out_size, True, params).lower
        else:
            lower = None

        if bound_upper:
            upper = self.optimize_bounds(region, out_size, False, params).upper
        else:
            upper = None

        return LinearBounds(region, lower, upper)

    def optimize_bounds(self, region, out_size, lower, params):
        optimizer = self.optimizer(params, self.lr)
        scheduler = ExponentialLR(optimizer, self.lr_decay)

        self.reset_params()

        for iteration in range(self.bounds_iterations):
            linear_bounds = self.initial_linear_bounds(region, out_size, lower=lower, upper=not lower)
            linear_bounds = self.crown_backward(linear_bounds, True)
            interval_bounds = linear_bounds.concretize()

            loss = (-interval_bounds.lower if lower else interval_bounds.upper).sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.clip_params()  # Projected Gradient Descent

        with torch.no_grad():
            linear_bounds = self.initial_linear_bounds(region, out_size, lower=lower, upper=not lower)
            return self.crown_backward(linear_bounds, True)

    @property
    @abc.abstractmethod
    def need_relaxation(self):
        raise NotImplementedError()

    def clear_relaxation(self):
        pass

    def set_relaxation(self, linear_bounds):
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
    def ibp_forward(self, bounds, save_relaxation=False):
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

    def reset_params(self):
        pass

    def clip_params(self):
        pass
