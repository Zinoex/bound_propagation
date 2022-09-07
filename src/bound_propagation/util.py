from typing import Callable

import torch


# For convenience in specifying type hints and for a semantic name (understanding)
TensorFunction = Callable[[torch.Tensor], torch.Tensor]


def proj_grad_to_range_(param, range):
    param.grad = param.data - (param.data - param.grad).clamp(min=range[0], max=range[1])


def clip_param_to_range_(param, range):
    param.data.clamp_(min=range[0], max=range[1])
