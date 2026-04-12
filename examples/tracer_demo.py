import math

import numpy as np
import torch
import torch.fx
from torch import nn


def f(x):
    return x.view(3, -1)


def main():
    tracer = torch.fx.Tracer()
    graph = tracer.trace(f)
    print(graph)  # Should print 'add'


if __name__ == "__main__":
    main()
