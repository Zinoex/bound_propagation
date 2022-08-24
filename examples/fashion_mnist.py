import timeit
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import trange

from bound_propagation import BoundModelFactory, HyperRectangle


class FashionMNISTNetwork(nn.Sequential):
    def __init__(self, *args):
        if args:
            # To support __get_index__ of nn.Sequential when slice indexing
            super().__init__(*args)
        else:
            img_size = 28 * 28
            classes = 10

            super().__init__(
                nn.Linear(img_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, classes)
            )


def construct_transform():
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Lambda(torch.flatten)
    ])

    # Identity transform - because cross entropy loss supports class indexing
    target_transform = transforms.Compose([])

    return transform, target_transform


def train(net, args):
    print('')
    print('[TRAINING]')

    transform, target_transform = construct_transform()
    train_data = datasets.FashionMNIST('../fashion_data', train=True, download=True,
                                       transform=transform, target_transform=target_transform)
    train_loader = DataLoader(train_data, batch_size=500, shuffle=True, num_workers=8)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

    k = 1.0
    for epoch in trange(20):

        running_loss = 0.0
        running_cross_entropy = 0.0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(args.device), y.to(args.device)
            optimizer.zero_grad(set_to_none=True)

            y_hat = net(X)

            cross_entropy = criterion(y_hat, y)

            bounds = net.ibp(HyperRectangle.from_eps(X, 0.01))
            logit = adversarial_logit(bounds, y)

            loss = k * cross_entropy + (1 - k) * criterion(logit, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_cross_entropy += cross_entropy.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:3d}] loss: {running_loss / 100:.3f}, cross entropy: {running_cross_entropy / 100:.3f}')
                running_loss = 0.0
                running_cross_entropy = 0.0

        k = max(k - 0.1, 0.5)


@torch.no_grad()
def test(net, args):
    print('')
    print('[TEST]')

    transform, target_transform = construct_transform()
    test_data = datasets.FashionMNIST('../fashion_data', train=False, download=True,
                                      transform=transform, target_transform=target_transform)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=8)

    correct = 0
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to(args.device), y.to(args.device)

        y_hat = net(X)

        predicted = torch.argmax(y_hat, 1)
        correct += (predicted == y).sum().item()

    print(f'Accuracy: {correct / len(test_data):.3f}')


# torch.no_grad() has a huge impact on the memory consumption (due to the need to store intermediate tensors)
# so if you don't need gradients (e.g. for verifying but not training) then do use it to allow larger batch sizes
@torch.no_grad()
def timing(net, args):
    print('')
    print('[TIMING]')

    methods = [
        ('ibp', net.ibp),
        ('crown_ibp', lambda x: net.crown_ibp(x).concretize()),
        ('crown', lambda x: net.crown(x).concretize()),
        ('alpha_crown_ibp', lambda x: net.crown_ibp(x, alpha=True).concretize()),
        ('alpha_crown', lambda x: net.crown(x, alpha=True).concretize())
    ]

    for method_name, method in methods:
        print(f'Method: {method_name}')

        for batch_size, iterations in [(10, 1000), (100, 100), (1000, 10)]:
            x = torch.rand(batch_size, 28 * 28, device=args.device)
            x = HyperRectangle.from_eps(x, 0.01)
            exec_time = timeit.timeit(lambda: method(x), number=iterations)
            out_size = method(x).lower.size()

            print(f'Out size: {out_size}, iterations: {iterations}, execution time: {exec_time}')


def adversarial_logit(y_hat, y):
    batch_size = y.size(0)
    classes = torch.arange(10, device=y.device).unsqueeze(0).expand(batch_size, -1)
    mask = (classes == y.unsqueeze(-1)).to(dtype=y_hat.lower.dtype)

    # Take upper bound for logit of all but the correct class where you take the lower bound
    adversarial_logit = (1 - mask) * y_hat.upper + mask * y_hat.lower

    return adversarial_logit


def adversarial_prob_margin(y_hat, y):
    batch_size = y.size(0)
    y_index = (torch.arange(batch_size, device=y.device), y)

    logit = adversarial_logit(y_hat, y)

    probs = F.softmax(logit, dim=1)
    label_probs = probs.gather(1, y.unsqueeze(1))

    others_mask = torch.ones_like(probs, dtype=torch.bool)
    others_mask[y_index] = False
    others_probs = probs[others_mask].view(batch_size, -1)

    return torch.min(label_probs - others_probs, dim=1).values


@torch.no_grad()
def verify(net, args):
    print('')
    print('[VERIFY]')

    methods = [
        ('ibp', net.ibp),
        ('crown_ibp', lambda x: net.crown_ibp(x).concretize()),
        ('crown', lambda x: net.crown(x).concretize()),
        ('alpha_crown_ibp', lambda x: net.crown_ibp(x, alpha=True).concretize()),
        ('alpha_crown', lambda x: net.crown(x, alpha=True).concretize())
    ]

    transform, target_transform = construct_transform()
    test_data = datasets.FashionMNIST('../fashion_data', train=False, download=True,
                                      transform=transform, target_transform=target_transform)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=8)

    for method_name, method in methods:
        print(f'Method: {method_name}')

        margin_sum = 0.0
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(args.device), y.to(args.device)
            X = HyperRectangle.from_eps(X, 0.01)
            y_hat = method(X)
            worst_margin = adversarial_prob_margin(y_hat, y)

            margin_sum += worst_margin.sum().item()

        print(f'[{method_name}] Average adversarial margin: {margin_sum / len(test_data):.3f}')


def main(args):
    net = FashionMNISTNetwork().to(args.device)

    factory = BoundModelFactory()
    net = factory.build(net)

    train(net, args)
    test(net, args)
    timing(net, args)
    verify(net, args)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default='cuda',
                        help='Select device for tensor operations')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
