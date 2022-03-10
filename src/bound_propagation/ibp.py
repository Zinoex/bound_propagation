from typing import Union, Optional

import torch

from torch import nn

from .util import add_method, IntervalBounds, LayerBounds


def ibp(class_or_obj):
    """
    Propagate a sample interval through a deterministic fully-connected feedforward network.
    This can correspond to a weight sample using MCMC.

    Assumptions:
    - The network is nn.Sequential
    - All layers of the network is nn.Linear or a monotone activation function

    This method does _not_ support weight bounds, only input bounds.

    @misc{wicker2020probabilistic,
      title={Probabilistic Safety for Bayesian Neural Networks},
      author={Matthew Wicker and Luca Laurenti and Andrea Patane and Marta Kwiatkowska},
      year={2020},
      eprint={2004.10281},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }
    """

    if isinstance(class_or_obj, nn.Sequential) or (isinstance(class_or_obj, type) and issubclass(class_or_obj, nn.Sequential)):
        return ibp_sequential(class_or_obj)
    elif isinstance(class_or_obj, nn.Linear) or (isinstance(class_or_obj, type) and issubclass(class_or_obj, nn.Linear)):
        return ibp_linear(class_or_obj)
    else:
        return ibp_activation(class_or_obj)


def ibp_sequential(class_or_obj):
    def _ibp(self: nn.Sequential, lower: torch.Tensor, upper: torch.Tensor, pre: bool = False) -> Union[IntervalBounds, LayerBounds]:
        layer_bounds = []

        for i, module in enumerate(self):
            if not hasattr(module, 'ibp'):
                # Decorator also adds the method inplace.
                ibp(module)

            if pre:
                layer_bounds.append((lower, upper))

            lower, upper = module.ibp(lower, upper)

        if pre:
            return layer_bounds
        else:
            return lower, upper

    add_method(class_or_obj, 'ibp', _ibp)
    return class_or_obj


def ibp_linear(class_or_obj):
    def ibp(self: nn.Linear, lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
        center, diff = (lower + upper) / 2, (upper - lower) / 2
        center, diff = center.unsqueeze(-2), diff.unsqueeze(-2)

        weight = self.weight.transpose(-1, -2)
        lower = center.matmul(weight) - diff.matmul(weight.abs()) + self.bias.unsqueeze(-2)
        lower = lower.squeeze(-2)

        upper = center.matmul(weight) + diff.matmul(weight.abs()) + self.bias.unsqueeze(-2)
        upper = upper.squeeze(-2)

        return lower, upper

    add_method(class_or_obj, 'ibp', ibp)
    return class_or_obj


def ibp_activation(class_or_obj):
    def ibp(self: nn.Module, lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
        return self(lower), self(upper)

    add_method(class_or_obj, 'ibp', ibp)
    return class_or_obj
