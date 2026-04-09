"""
Abstract input region definitions for bound propagation.

Input regions represent the domain of input variables - these can be
hyperrectangles, Lp norm balls, or other convex sets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class AbstractRegion(ABC):
    """
    Abstract base class for input regions.

    An input region defines the domain of input variables. This could be:
    - HyperRectangle: [lower, upper] box constraints
    - LpNormBall: ||x - center|| <= epsilon in some Lp norm
    - Other convex sets

    The region is used to concretize bounds - converting symbolic/linear
    bounds to concrete interval bounds.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """
        Get shape of the input region.

        Returns:
            Shape tuple
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        Get device of the region.

        Returns:
            Device where region tensors are stored
        """
        pass

    @abstractmethod
    def to(self, device: str | torch.device) -> AbstractRegion:
        """
        Move region to a device.

        Args:
            device: Target device

        Returns:
            New region on target device
        """
        pass

    @abstractmethod
    def __getitem__(self, item) -> AbstractRegion:
        """
        Slice/index the input region.

        Args:
            item: Slice/index specification

        Returns:
            Sliced region
        """
        pass
