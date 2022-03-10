from bound_propagation.general import BoundModule, IntervalBounds


class BoundLinear(BoundModule):
    def crown(self, region, **kwargs):
        pass

    def crown_ibp(self, region, **kwargs):
        pass

    def ibp(self, region, **kwargs):
        center, diff = region.center, region.width / 2
        center, diff = center.unsqueeze(-2), diff.unsqueeze(-2)

        weight = self.weight.transpose(-1, -2)

        w_mid = center.matmul(weight)
        w_diff = diff.matmul(weight.abs())

        lower = w_mid - w_diff
        lower = lower.squeeze(-2)

        upper = w_mid + w_diff
        upper = upper.squeeze(-2)

        if self.bias is not None:
            lower = lower + self.bias
            upper = upper + self.bias

        return IntervalBounds(region, lower, upper)
