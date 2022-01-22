import timeit
from argparse import ArgumentParser

import torch
from torch import nn

from bound_propagation import crown, crown_ibp, ibp


@crown
@crown_ibp
@ibp
class FashionMNISTNetwork(nn.Sequential):
    def __init__(self, *args):
        if args:
            # To support __get_index__ of nn.Sequential when slice indexing
            super().__init__(*args)
        else:
            img_size = 28 * 28
            classes = 10

            super().__init__(
                nn.Linear(img_size, 16),
                nn.Tanh(),
                nn.Linear(16, 16),
                nn.Tanh(),
                nn.Linear(16, classes)
            )


# torch.no_grad() has a huge impact on the memory consumption (due to the need to store intermediate tensors)
# so if you don't need gradients (e.g. for verifying but not training) then do use it to allow larger batch sizes
@torch.no_grad()
def main(args):
    net = FashionMNISTNetwork().to(args.device)

    methods = [
        ('ibp', lambda: net.ibp(x - epsilon, x + epsilon)),
        ('crown_ibp', lambda: net.crown_ibp_interval(x - epsilon, x + epsilon)),
        ('crown', lambda: net.crown_interval(x - epsilon, x + epsilon))
    ]

    for method_name, method in methods:
        print(f'Method: {method_name}')

        for batch_size, iterations in [(100, 1000), (1000, 100), (10000, 10)]:
            x = torch.rand(batch_size, 28 * 28, device=args.device)
            epsilon = 0.1
            exec_time = timeit.timeit(method, number=iterations)

            print(f'Batch size: {batch_size}, iterations: {iterations}, execution time: {exec_time}')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=['cuda', 'cpu'], type=torch.device, default='cuda', help='Select device for tensor operations')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
