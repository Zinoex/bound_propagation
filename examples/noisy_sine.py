import os
from functools import cached_property
from typing import Tuple, Optional
from argparse import ArgumentParser
from math import sqrt

import numpy as np
from tqdm import trange

import torch
from torch import distributions, nn, optim
from torch.utils.data import TensorDataset, DataLoader

from bound_propagation import ibp, crown_ibp, crown

from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from bound_propagation.general import HyperRectangle
from bound_propagation.sequential import BoundSequential


def bound_propagation(model, lower_x, upper_x, interval_method=None):
    if interval_method is None:
        interval_method = model.ibp

    lower_ibp, upper_ibp = interval_method(lower_x, upper_x)
    lower_lbp, upper_lbp = model.crown_linear(lower_x, upper_x)

    input_bounds = lower_x.cpu(), upper_x.cpu()
    ibp_bounds = lower_ibp.cpu(), upper_ibp.cpu()
    lbp_bounds = (lower_lbp[0].cpu(), lower_lbp[1].cpu()), (upper_lbp[0].cpu(), upper_lbp[1].cpu())

    return input_bounds, ibp_bounds, lbp_bounds


def plot_bounds_1d(model, args):
    num_slices = 20

    boundaries = torch.linspace(-2, 2, num_slices + 1, device=args.device).view(-1, 1)
    lower_x, upper_x = boundaries[:-1], boundaries[1:]

    (lower_x, upper_x), (lower_ibp, upper_ibp), (lower_lbp, upper_lbp) = bound_propagation(model, lower_x, upper_x)

    plt.figure(figsize=(6.4 * 2, 4.8 * 2))

    plt.ylim(-4, 4)

    for i in range(num_slices):
        x1, x2 = lower_x[i].item(), upper_x[i].item()
        y1, y2 = lower_ibp[i].item(), upper_ibp[i].item()

        plt.plot([x1, x2], [y1, y1], color='blue', label='IBP' if i == 0 else None)
        plt.plot([x1, x2], [y2, y2], color='blue')

        y1, y2 = lower_lbp[0][i, 0, 0] * x1 + lower_lbp[1][i], lower_lbp[0][i, 0, 0] * x2 + lower_lbp[1][i]
        y3, y4 = upper_lbp[0][i, 0, 0] * x1 + upper_lbp[1][i], upper_lbp[0][i, 0, 0] * x2 + upper_lbp[1][i]

        y1, y2 = y1.item(), y2.item()
        y3, y4 = y3.item(), y4.item()

        plt.plot([x1, x2], [y1, y2], color='green', label='CROWN' if i == 0 else None)
        plt.plot([x1, x2], [y3, y4], color='green')

    X = torch.linspace(-2, 2, 1000, device=args.device).view(-1, 1)
    y = model(X)
    X, y = X.cpu().numpy(), y.cpu().numpy()

    plt.plot(X, y, color='blueviolet', label='Function to bound')

    plt.title(f'Bound propagation')
    plt.legend()

    plt.show()
    # plt.savefig(f'visualization/lbp.png', dpi=300)


LinearBound = Tuple[torch.Tensor, torch.Tensor]


class LinearBounds:
    lower: LinearBound
    upper: LinearBound

    def __init__(self, affine):
        self.lower = affine[0]
        self.upper = affine[1]

    def __getitem__(self, item):
        affine = (
            (self.lower[0][item], self.lower[1][item]),
            (self.upper[0][item], self.upper[1][item])
        )

        return LinearBounds(affine)

    def cat(self, other):
        affine = (
            (torch.cat([self.lower[0], other.lower[0]], dim=0), torch.cat([self.lower[1], other.lower[1]], dim=0)),
            (torch.cat([self.upper[0], other.upper[0]], dim=0), torch.cat([self.upper[1], other.upper[1]], dim=0))
        )

        return LinearBounds(affine)


class HyperRectangles:
    lower: Optional[torch.Tensor]
    upper: Optional[torch.Tensor]
    affine: Optional[LinearBounds]

    def __init__(self, lower, upper, affine):
        self.lower, self.upper = lower, upper

        if affine is None:
            self.affine = None
        elif isinstance(affine, LinearBounds):
            self.affine = affine
        else:
            self.affine = LinearBounds(affine)

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    @cached_property
    def max_dist(self):
        A, b = self.distance

        center, diff = self.center, self.width / 2
        center, diff = center.unsqueeze(-2), diff.unsqueeze(-2)

        A = A.transpose(-1, -2)

        max_dist = center.matmul(A) + diff.matmul(A.abs()) + b.unsqueeze(-2)
        max_dist = max_dist.view(max_dist.size()[:-2])

        return max_dist

    @property
    def distance(self):
        # Distance between bounds for each sample
        return self.affine.upper[0] - self.affine.lower[0], self.affine.upper[1] - self.affine.lower[1]

    @cached_property
    def global_bounds(self):
        diff = self.width / 2

        A, b = self.affine.lower
        lower = A.matmul(self.center) - A.abs().matmul(diff) + b

        A, b = self.affine.upper
        upper = A.matmul(self.center) + A.abs().matmul(diff) + b

        return lower, upper

    def __getitem__(self, item):
        new_lower = self.lower[item]
        if new_lower.size(0) == 0:
            return HyperRectangles.empty()

        return HyperRectangles(new_lower, self.upper[item], self.affine[item])

    def cat(self, other):
        if self.lower is None:
            return other

        if other.lower is None:
            return self

        return HyperRectangles(
            torch.cat([self.lower, other.lower], dim=0),
            torch.cat([self.upper, other.upper], dim=0),
            self.affine.cat(other.affine)
        )

    @staticmethod
    def empty():
        return HyperRectangles(None, None, None)

    def __bool__(self):
        return self.lower is not None

    def __len__(self):
        return 0 if self.lower is None else self.lower.size(0)


def construct_hyperrects(net, lower, upper):
    affine = net.crown_linear(lower, upper)
    return HyperRectangles(lower, upper, affine)


def plot_partition(model, args, rect):
        x1, x2 = rect.lower, rect.upper

        plt.clf()
        ax = plt.axes(projection='3d')

        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 10), torch.linspace(x1[1], x2[1], 10))

        # Plot IBP
        y1, y2 = rect.global_bounds[0].item(), rect.global_bounds[1].item()
        y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

        surf = ax.plot_surface(x1, x2, y1, color='blue', label='CROWN interval', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y2, color='blue', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # Plot LBP
        y_lower = rect.affine.lower[0][0, 0] * x1 + rect.affine.lower[0][0, 1] * x2 + rect.affine.lower[1]
        y_upper = rect.affine.upper[0][0, 0] * x1 + rect.affine.upper[0][0, 1] * x2 + rect.affine.upper[1]

        surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN linear', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y_upper, color='green', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # y_dist = rect.distance[0][0, 0] * x1 + rect.distance[0][0, 1] * x2 + rect.distance[1]
        # surf = ax.plot_surface(x1, x2, y_dist, cmap='viridis', vmin=0.0, vmax=0.1)
        # surf._facecolors2d = surf._facecolor3d
        # surf._edgecolors2d = surf._edgecolor3d

        m = cm.ScalarMappable(cmap='viridis')
        m.set_clim(0.0, 0.1)
        cb = plt.colorbar(m)

        cb.ax.plot([0, 1], [rect.max_dist] * 2, 'white', label='Max dist')
        legend = cb.ax.legend(bbox_to_anchor=(2.0, 1.0), loc='upper left')
        frame = legend.get_frame()
        frame.set_facecolor('gray')
        frame.set_edgecolor('black')

        # Plot function
        x1, x2 = rect.lower, rect.upper
        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 50), torch.linspace(x1[1], x2[1], 50))
        X = torch.cat(tuple(torch.dstack([x1, x2]))).to(args.device)
        y = model(X).view(50, 50)
        y = y.cpu()

        surf = ax.plot_surface(x1, x2, y, color='red', label='Function to bound', shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # General plot config
        plt.xlabel('x')
        plt.ylabel('y')

        plt.title(f'Bound propagation')
        plt.legend()

        plt.show()


def plot_bounds_2d(model, args):
    num_slices = 20

    x_space = torch.linspace(-2.0, 2.0, num_slices + 1, device=args.device)
    cell_width = (x_space[1] - x_space[0]) / 2
    slice_centers = (x_space[:-1] + x_space[1:]) / 2

    cell_centers = torch.cartesian_prod(slice_centers, slice_centers)
    lower_x, upper_x = cell_centers - cell_width, + cell_centers + cell_width

    (lower_x, upper_x), (lower_ibp, upper_ibp), (lower_lbp, upper_lbp) = bound_propagation(model, lower_x, upper_x)

    # Plot function over entire space
    plt.clf()
    ax = plt.axes(projection='3d')

    x1, x2 = torch.meshgrid(torch.linspace(-2.0, 2.0, 500), torch.linspace(-2.0, 2.0, 500))
    X = torch.cat(tuple(torch.dstack([x1, x2]))).to(args.device)
    y = model(X).view(500, 500)
    y = y.cpu()

    surf = ax.plot_surface(x1, x2, y, color='red', alpha=0.8)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # General plot config
    plt.xlabel('x')
    plt.ylabel('y')

    plt.title(f'Sine approximator & partitioning')
    plt.show()

    for i in trange(num_slices ** 2):
        x1, x2 = lower_x[i], upper_x[i]
        affine_bounds = (lower_lbp[0][i], lower_lbp[1][i]), (upper_lbp[0][i], upper_lbp[1][i])
        rect = HyperRectangles(x1, x2, affine_bounds)
        plot_partition(model, args, rect)


@torch.no_grad()
def plot_bounds(model, args):
    if args.dim == 1:
        plot_bounds_1d(model, args)
    elif args.dim == 2:
        plot_bounds_2d(model, args)
    else:
        raise NotImplementedError()


def evaluate(model, args):
    criterion = nn.MSELoss()

    dataset = NoisySineDataset(dim=args.dim)
    X_train, y_train = dataset[:]
    X, y = X_train.to(args.device), y_train.to(args.device)

    y_pred = model(X)
    loss = criterion(y_pred, y)

    print(f'MSE: {loss.item()}')


@torch.no_grad()
def test(model, args):
    os.makedirs('visualization', exist_ok=True)

    evaluate(model, args)
    plot_bounds(model, args)


def noise(train_size, sigma):
    dist = distributions.Normal(0.0, sigma)

    train_size = (train_size,)
    return dist.sample(train_size)


def f_1d(train_size, sigma):
    X = torch.linspace(-1.0, 1.0, train_size).view(-1, 1)
    return X, torch.sin(2 * np.pi * X) + noise(train_size, sigma).view(-1, 1)


def f_2d(train_size, sigma):
    x_space = torch.linspace(-1.0, 1.0, int(sqrt(train_size)))
    X = torch.cartesian_prod(x_space, x_space)
    y = 0.5 * torch.sin(2 * np.pi * X[:, 0]) + 0.5 * torch.sin(2 * np.pi * X[:, 1]) + noise(train_size, sigma)

    return X, y.view(-1, 1)


class NoisySineDataset(TensorDataset):
    def __init__(self, dim=1, sigma=0.05, train_size=2 ** 10):
        if dim == 1:
            X, y = f_1d(train_size, sigma)
        elif dim == 2:
            X, y = f_2d(train_size, sigma)
        else:
            raise NotImplementedError()

        super(NoisySineDataset, self).__init__(X, y)


@crown
@crown_ibp
@ibp
class Model(nn.Sequential):
    def __init__(self, *args, dim=1):
        if args:
            super().__init__(*args)
        else:
            super().__init__(
                nn.Linear(dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )


def train(model, args, eps=0.005):
    dataset = NoisySineDataset(dim=args.dim, train_size=2**12)
    X, y = dataset[:]

    bounded_model = BoundSequential(model)

    optimizer = optim.Adam(model.parameters(), lr=4e-3)
    criterion = nn.MSELoss()

    for epoch in trange(1000):
        optimizer.zero_grad(set_to_none=True)

        y_pred = model(X)
        loss = criterion(y_pred, y)

        lower, upper = X - eps, X + eps
        y_min, y_max = model.ibp(lower, upper)
        loss = loss + torch.max(criterion(y_min, y), criterion(y_max, y))

        # bounds = bounded_model.ibp(HyperRectangle.from_eps(X, eps))
        # loss = loss + torch.max(criterion(bounds.lower, y), criterion(bounds.upper, y))

        loss.backward()
        optimizer.step()

        if epoch % 100 == 99:
            with torch.no_grad():
                y_pred = model(X)
                loss = criterion(y_pred, y).item()

                print(f"loss: {loss:>7f}")


def main(args):
    net = Model(dim=args.dim).to(args.device)

    train(net, args)
    test(net, args)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default='cuda',
                        help='Select device for tensor operations')
    parser.add_argument('--dim', choices=[1, 2], type=int, default=1, help='Dimensionality of the noisy sine')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
