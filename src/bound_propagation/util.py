from typing import Callable

import torch


# For convenience in specifying type hints and for a semantic name (understanding)
TensorFunction = Callable[[torch.Tensor], torch.Tensor]
