from typing import TYPE_CHECKING

from plum import dispatch

from bound_propagation.bounds import AbstractBounds
from bound_propagation.regions import AbstractInputRegion

if TYPE_CHECKING:
    import torch


@dispatch.abstract
def concretize(region: AbstractInputRegion, bounds: AbstractBounds) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("Abstract method for concretize")

from . import hyperrectangle  # noqa: E402, F401

__all__ = [
    "concretize",
]
