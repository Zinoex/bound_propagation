import math

import numpy as np
import torch
import torch.fx


def f(x):
    return x + np.pi * math.sqrt(2)


def main():
    tracer = torch.fx.Tracer()
    graph = tracer.trace(f)
    print(graph)


if __name__ == "__main__":
    main()
