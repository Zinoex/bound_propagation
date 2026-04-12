"""
Abstract base class for bound representations.

Defines the interface that all bound types must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..regions import AbstractRegion


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
    - Concretization to intervals for local analysis
    """

    def __init__(self, region: AbstractRegion):
        self.region = region

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

    @abstractmethod
    def __getitem__(self, item) -> AbstractBounds:
        """
        Slice/index the bounds.

        Args:
            item: Slice/index specification (e.g., for batch slicing)
        Returns:
            New bounds corresponding to the slice/index
        """
        raise NotImplementedError("Bounds slicing not implemented for this bound type")

    @abstractmethod
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
        raise NotImplementedError("Concretize method must be implemented by subclasses")

    @abstractmethod
    def clone(self) -> AbstractBounds:
        """
        Create a deep copy of these bounds.

        Returns:
            Cloned bounds
        """
        pass
