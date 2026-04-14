import math

import numpy as np
import torch
import torch.fx


def f(x, y):
    return x + y + torch.sin(x) * torch.cos(y) + torch.exp(x * y) + x**2 - y**2


def main():
    tracer = torch.fx.Tracer()
    graph = tracer.trace(f)
    print(graph)


if __name__ == "__main__":
    main()
