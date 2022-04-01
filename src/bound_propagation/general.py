import abc

import torch
from torch import nn

from .bounds import LinearBounds, IntervalBounds


class BoundModule(nn.Module, abc.ABC):
    def __init__(self, module, factory, **kwargs):
        super().__init__()
        self.module = module

    def crown(self, region, bound_lower=True, bound_upper=True):
        out_size = self.propagate_size(region.lower.size(-1))

        # No grad for relaxations improves accuracy and stabilizes training
        # for CROWN
        with torch.no_grad():
            while self.need_relaxation:
                linear_bounds, module = self.backward_relaxation(region)
                module.set_relaxation(linear_bounds)

        linear_bounds = self.initial_linear_bounds(region, out_size, lower=bound_lower, upper=bound_upper)
        linear_bounds = self.crown_backward(linear_bounds)
        self.clear_relaxation()
        return linear_bounds

    def initial_linear_bounds(self, region, out_size, lower=True, upper=True):
        W_tilde = torch.eye(out_size, device=region.lower.device)\
            .unsqueeze(0).expand(region.lower.size(0), out_size, out_size)
        bias = torch.zeros((out_size,), device=region.lower.device)\
            .unsqueeze(0).expand(region.lower.size(0), out_size)

        lower = (W_tilde, bias) if lower else None
        upper = (W_tilde, bias) if upper else None

        linear_bounds = LinearBounds(region, lower, upper)
        return linear_bounds

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
    def crown_backward(self, linear_bounds):
        raise NotImplementedError()

    def crown_ibp(self, region, bound_lower=True, bound_upper=True):
        out_size = self.propagate_size(region.lower.size(-1))

        # No grad for relaxations improves accuracy
        # CROWN-IBP is already stable in training
        with torch.no_grad():
            bounds = IntervalBounds(region, region.lower, region.upper)
            self.ibp_forward(bounds, save_relaxation=True)

        linear_bounds = self.initial_linear_bounds(region, out_size, lower=bound_lower, upper=bound_upper)
        linear_bounds = self.crown_backward(linear_bounds)
        self.clear_relaxation()
        return linear_bounds

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
