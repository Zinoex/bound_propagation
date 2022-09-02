import torch
from torch import nn

from .bounds import IntervalBounds, LinearBounds
from .general import BoundModule


class Parallel(nn.Module):
    def __init__(self, *subnetworks, split_size=None):
        super().__init__()

        self.split_size = split_size
        self.subnetworks = nn.ModuleList(subnetworks)

    def forward(self, x):
        if self.split_size is not None:
            x = x.split(self.split_size, dim=-1)
            y = [network(x) for network, x in zip(self.subnetworks, x)]
        else:
            y = [network(x) for network in self.subnetworks]

        return torch.cat(y, dim=-1)


class BoundParallel(BoundModule):
    def __init__(self, module, factory, **kwargs):
        super().__init__(module, factory)

        self.in_sizes = None
        self.out_sizes = None
        self.subnetworks = nn.ModuleList([factory.build(network) for network in module.subnetworks])

    @property
    def need_relaxation(self):
        return any([network.need_relaxation for network in self.subnetworks])

    def clear_relaxation(self):
        for network in self.subnetworks:
            network.clear_relaxation()

    def set_relaxation(self, linear_bounds, modules, sizes, extras):
        indices = self.indices(sizes)
        for module, index, extra in zip(modules, indices, extras):
            module.set_relaxation(linear_bounds[..., index[0]:index[1], :], *extra)

    def backward_relaxation(self, region):
        assert any([network.need_relaxation for network in self.subnetworks])
        bounds = [self.submodule_backward_relaxation(region, network) for network in self.subnetworks]

        if self.module.split_size is None:
            bounds = [bound for bound in bounds if bound[1] is not None]

            modules = [bound[1] for bound in bounds]
            sizes = [bound[0].lower[0].size(-2) for bound in bounds]

            lowerA = torch.cat([bound[0].lower[0] for bound in bounds], dim=-2)
            lower_bias = torch.cat([bound[0].lower[1] for bound in bounds], dim=-1)

            upperA = torch.cat([bound[0].upper[0] for bound in bounds], dim=-2)
            upper_bias = torch.cat([bound[0].upper[1] for bound in bounds], dim=-1)

            extras = [bound[2:] for bound in bounds]
            return LinearBounds(region, (lowerA, lower_bias), (upperA, upper_bias)), self, modules, sizes, extras
        else:
            bounds = [self.submodule_backward_relaxation(region, network) for network in self.subnetworks]

            indices = self.indices(self.in_sizes)
            bounds = [(bound, index) for bound, index in zip(bounds, indices) if bound[1] is not None]

            modules = [bound[1] for bound, index in bounds]
            sizes = [bound[0].lower[0].size(-2) for bound, index in bounds]
            total_size = sum(self.in_sizes)

            lowerA = torch.cat([self.padding(bound[0].lower[0], index, total_size) for bound, index in bounds], dim=-2)
            lower_bias = torch.cat([bound[0].lower[1] for bound, index in bounds], dim=-1)

            upperA = torch.cat([self.padding(bound[0].upper[0], index, total_size) for bound, index in bounds], dim=-2)
            upper_bias = torch.cat([bound[0].upper[1] for bound, index in bounds], dim=-1)

            extras = [bound[2:] for bound, index in bounds]
            return LinearBounds(region, (lowerA, lower_bias), (upperA, upper_bias)), self, modules, sizes, extras

    def submodule_backward_relaxation(self, region, network):
        if network.need_relaxation:
            return network.backward_relaxation(region)
        else:
            return None, None

    def padding(self, A, index, total_size):
        size = list(A.size())
        size[-1] = index[0]
        pre_padding = torch.zeros(size, device=A.device)

        size[-1] = total_size - index[1]
        post_padding = torch.zeros(size, device=A.device)

        return torch.cat([pre_padding, A, post_padding], dim=-1)

    def crown_backward(self, linear_bounds, optimize):
        assert self.out_sizes is not None
        split_bounds = self.split(linear_bounds, self.out_sizes)
        residual_bounds = [network.crown_backward(split_linear_bound, optimize) for network, split_linear_bound in zip(self.subnetworks, split_bounds)]

        if linear_bounds.lower is None:
            lower = None
        else:
            if self.module.split_size is None:
                lowerA = torch.stack([residual_linear_bound.lower[0] for residual_linear_bound in residual_bounds], dim=-1).sum(dim=-1)
            else:
                lowerA = torch.cat([residual_linear_bound.lower[0] for residual_linear_bound in residual_bounds], dim=-1)

            lower = (lowerA, torch.stack([residual_linear_bound.lower[1] for residual_linear_bound in residual_bounds], dim=-1).sum(dim=-1) - (len(self.subnetworks) - 1) * linear_bounds.lower[1])

        if linear_bounds.upper is None:
            upper = None
        else:
            if self.module.split_size is None:
                upperA = torch.stack([residual_linear_bound.upper[0] for residual_linear_bound in residual_bounds], dim=-1).sum(dim=-1)
            else:
                upperA = torch.cat([residual_linear_bound.upper[0] for residual_linear_bound in residual_bounds], dim=-1)

            upper = (upperA, torch.stack([residual_linear_bound.upper[1] for residual_linear_bound in residual_bounds], dim=-1).sum(dim=-1) - (len(self.subnetworks) - 1) * linear_bounds.upper[1])

        return LinearBounds(linear_bounds.region, lower, upper)

    def ibp_forward(self, bounds, save_relaxation=False, save_input_bounds=False):
        if self.module.split_size is None:
            split_bounds = [bounds for _ in range(len(self.subnetworks))]
        else:
            split_bounds = self.split(bounds, self.split_sizes(bounds.lower.size(-1)))

        residual_bounds = [network.ibp_forward(bound, save_relaxation=save_relaxation, save_input_bounds=save_input_bounds) for network, bound in zip(self.subnetworks, split_bounds)]

        lower = torch.cat([bound.lower for bound in residual_bounds], dim=-1)
        upper = torch.cat([bound.upper for bound in residual_bounds], dim=-1)
        return IntervalBounds(bounds.region, lower, upper)

    def split(self, bounds, sizes):
        indices = self.indices(sizes)
        return [bounds[..., from_:to_] for from_, to_ in indices]

    def indices(self, sizes):
        indices = torch.tensor(sizes).cumsum(0).tolist()
        return list(zip([0] + indices[:-1], indices))

    def propagate_size(self, in_size):
        split_sizes = self.module.split_size
        if split_sizes is not None:
            split_sizes = self.split_sizes(in_size)

            out_sizes = [network.propagate_size(split_size) for network, split_size in zip(self.subnetworks, split_sizes)]
        else:
            out_sizes = [network.propagate_size(in_size) for network in self.subnetworks]

        self.out_sizes = out_sizes
        self.in_sizes = split_sizes

        return sum(out_sizes)

    def split_sizes(self, in_size):
        split_sizes = self.module.split_size
        if isinstance(split_sizes, int):
            num_full_size = in_size // split_sizes
            split_sizes = [split_sizes for _ in range(num_full_size)]
            if in_size % self.module.split_size != 0:
                split_sizes.append(in_size - num_full_size * self.module.split_size)

        return split_sizes

    def bound_parameters(self):
        for network in self.subnetworks:
            yield from network.bound_parameters()

    def clip_params(self):
        for network in self.subnetworks:
            network.clip_params()


class Cat(Parallel):
    def __init__(self, subnetwork):
        super().__init__(
            nn.Identity(),
            subnetwork
        )
