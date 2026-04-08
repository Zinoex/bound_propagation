"""
Abstract base class for bound representations.

Defines the interface that all bound types must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from ..concretize import concretize

if TYPE_CHECKING:
    from ..regions import AbstractInputRegion


class AbstractBounds(ABC):
    """
    Abstract base class for bound representations.

    All bound types (interval, linear, symbolic) must implement this interface.
    Bounds represent constraints on tensor values - typically lower and upper bounds,
    but can be more complex (e.g., affine relaxations).

    Each bounds object carries a reference to the input region, which is needed
    for concretization (converting symbolic/affine bounds to concrete intervals).

    The key operations are:
    - Propagation through operations (add, mul, matmul, etc.)
    - Combination of bounds from different sources
    - Concretization to intervals for final analysis

    Attributes:
        region: The input region (HyperRectangle, LpNormBall, etc.) that defines
                the domain of input variables
    """

    def __init__(self, region: AbstractInputRegion) -> None:
        """
        Initialize bounds with an input region.

        Args:
            region: Input region defining the domain
        """
        self.region = region

    @property
    @abstractmethod
    def lower(self) -> torch.Tensor:
        """
        Get concrete lower bound.

        Returns:
            Tensor of lower bounds for each element
        """
        pass

    @property
    @abstractmethod
    def upper(self) -> torch.Tensor:
        """
        Get concrete upper bound.

        Returns:
            Tensor of upper bounds for each element
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """
        Get shape of bounded tensor.

        Returns:
            Shape tuple
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        Get device of bounds.

        Returns:
            Device where bound tensors are stored
        """
        pass

    @abstractmethod
    def to(self, device: str | torch.device) -> AbstractBounds:
        """
        Move bounds to a device.

        Args:
            device: Target device

        Returns:
            New bounds on target device
        """
        pass

    def concretize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concretize bounds to interval bounds.

        This method uses the input region to convert symbolic/affine bounds
        into concrete interval bounds. The default implementation assumes that
        the bounds are already concrete intervals and simply returns them.

        Subclasses with more complex bound types (e.g., linear bounds) should
        override this method to perform the necessary concretization logic.

        Returns:
            IntervalBounds representing the concretized bounds
        """
        return concretize(self.region, self)  # ty:ignore[invalid-return-type]

    @abstractmethod
    def clone(self) -> AbstractBounds:
        """
        Create a deep copy of these bounds.

        Returns:
            Cloned bounds
        """
        pass
