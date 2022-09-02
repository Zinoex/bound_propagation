from torch import nn

from bound_propagation.general import BoundModule


class BoundSequential(BoundModule):
    def __init__(self, sequential, factory, **kwargs):
        super().__init__(sequential, factory)
        self.bound_sequential = nn.Sequential(*list(map(factory.build, sequential)))

    @property
    def need_relaxation(self):
        return any((module.need_relaxation for module in self.bound_sequential))

    def clear_relaxation(self):
        for module in self.bound_sequential:
            module.clear_relaxation()

    def backward_relaxation(self, region):
        for i, module in enumerate(self.bound_sequential):
            if module.need_relaxation:
                linear_bounds, relaxation_module, *extra = module.backward_relaxation(region)
                return self.subnetwork_crown_backward(self.bound_sequential[:i], linear_bounds, False), relaxation_module, *extra

        assert False, 'At least one module needs relaxation'

    def subnetwork_crown_backward(self, subnetwork, linear_bounds, optimize):
        for module in list(reversed(subnetwork)):
            linear_bounds = module.crown_backward(linear_bounds, optimize)

        return linear_bounds

    def crown_backward(self, linear_bounds, optimize):
        return self.subnetwork_crown_backward(self.bound_sequential, linear_bounds, optimize)

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        for module in self.bound_sequential:
            bounds = module.ibp_forward(bounds, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds)

        return bounds

    def propagate_size(self, in_size):
        size = in_size
        for module in self.bound_sequential:
            size = module.propagate_size(size)

        return size

    def bound_parameters(self):
        for module in self.bound_sequential:
            yield from module.bound_parameters()

    def clip_params(self):
        for module in self.bound_sequential:
            module.clip_params()
