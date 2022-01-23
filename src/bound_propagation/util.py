import types
from typing import Tuple, List, Optional, Callable

import torch


def add_method(class_or_obj, name, func):
    if not isinstance(class_or_obj, type):
        func = types.MethodType(func, class_or_obj)

    setattr(class_or_obj, name, func)


# For convenience in specifying type hints and for a semantic name (understanding)
OptionalTensor = Optional[torch.Tensor]
TensorFunction = Callable[[torch.Tensor], torch.Tensor]

LinearBound = Tuple[torch.Tensor, torch.Tensor]
LinearBounds = Tuple[LinearBound, LinearBound]
IntervalBounds = Tuple[torch.Tensor, torch.Tensor]

LayerBound = Tuple[torch.Tensor, torch.Tensor]
LayerBounds = List[LayerBound]

AlphaBeta = Tuple[Tuple[OptionalTensor, OptionalTensor], Tuple[OptionalTensor, OptionalTensor]]
AlphaBetas = List[AlphaBeta]

WeightBias = Tuple[torch.Tensor, torch.Tensor]