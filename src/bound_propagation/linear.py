from bound_propagation.general import BoundModule, IntervalBounds


class BoundLinear(BoundModule):
    def crown(self, region, **kwargs):
        pass

    def crown_ibp(self, region, **kwargs):
        pass

    def ibp(self, region, **kwargs):
        center, diff = region.center, region.width / 2
        center, diff = center.unsqueeze(-2), diff.unsqueeze(-2)

        weight = self.module.weight.transpose(-1, -2)
        lower = center.matmul(weight) - diff.matmul(weight.abs()) + self.module.bias.unsqueeze(-2)
        lower = lower.squeeze(-2)

        upper = center.matmul(weight) + diff.matmul(weight.abs()) + self.module.bias.unsqueeze(-2)
        upper = upper.squeeze(-2)

        return IntervalBounds(region, lower, upper)
