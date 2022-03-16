from bound_propagation.general import BoundModule, IntervalBounds
from bound_propagation.ibp import ibp_linear_jit


class BoundLinear(BoundModule):
    def crown(self, region, **kwargs):
        pass

    def crown_ibp(self, region, **kwargs):
        pass

    def ibp(self, region, **kwargs):
        lower, upper = ibp_linear_jit(self.module.weight, self.module.bias, region.lower, region.upper)
        return IntervalBounds(region, lower, upper)
