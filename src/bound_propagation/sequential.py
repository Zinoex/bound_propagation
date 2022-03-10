from functools import wraps

from torch import nn

from bound_propagation.activation import BoundReLU, BoundTanh, BoundActivation
from bound_propagation.general import BoundModule
from bound_propagation.linear import BoundLinear


def clear_bounded(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        for module in self.bounded_sequential:
            if isinstance(module, BoundActivation):
                module.clear_alpha_beta()

        return func(self, *args, **kwargs)
    return wrapper


class BoundSequential(BoundModule):
    class_mapping = {
        nn.Linear: BoundLinear,
        nn.ReLU: BoundReLU,
        nn.Tanh: BoundTanh,
    }

    def __init__(self, sequential, **kwargs):
        super().__init__(sequential)
        self.kwargs = kwargs
        self.bound_sequential = nn.Sequential(*list(map(self.parse_module, sequential)))

    def parse_module(self, module):
        for orig_class, bound_class in self.class_mapping.items():
            if isinstance(module, orig_class):
                return bound_class(module, **self.kwargs)

        raise NotImplementedError('Module type not supported - add BoundModule for layer to class_mapping')

    @clear_bounded
    def crown(self, region, **kwargs):
        pass

    def _crown_backward(self, region, **kwargs):
        pass


    @clear_bounded
    def crown_ibp(self, region, **kwargs):
        self._crown_ibp_forward(region, **kwargs)
        return self._crown_backward(region, **kwargs)

    def _crown_ibp_forward(self, region, **kwargs):
        bounds = region
        for module in self.bound_sequential:
            if isinstance(module, BoundActivation):
                module.alpha_beta(preactivation=bounds)

            bounds = module.ibp(bounds, **kwargs)

    def ibp(self, region, **kwargs):
        bounds = region
        for module in self.bound_sequential:
            bounds = module.ibp(bounds, **kwargs)

        bounds.region = region
        return bounds
