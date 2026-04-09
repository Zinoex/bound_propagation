"""
Multi-input region for graphs with multiple input nodes.

Provides a way to specify different regions for different input nodes.
"""

from __future__ import annotations

import torch

from .abstract import AbstractRegion
from .hyperrectangle import HyperRectangle


class MultiInputRegion(AbstractRegion):
    """
    Region for graphs with multiple input nodes.

    Maps each input node ID to its corresponding HyperRectangle region.
    This allows specifying different bounds for different inputs.

    Example:
        >>> # Graph with two inputs
        >>> region = MultiInputRegion({
        ...     0: HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0])),
        ...     1: HyperRectangle(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])),
        ... })
        >>> region[0]  # Returns HyperRectangle for input node 0
    """

    def __init__(self, regions: dict[int, HyperRectangle]) -> None:
        """
        Initialize multi-input region.

        Args:
            regions: Dictionary mapping input node IDs to their regions

        Raises:
            ValueError: If regions dict is empty
        """
        if not regions:
            raise ValueError("MultiInputRegion requires at least one input region")

        self.regions = regions

        # Validate all regions are on the same device
        devices = {region.device for region in regions.values()}
        if len(devices) > 1:
            raise ValueError(f"All regions must be on the same device, found: {devices}")

        self._device = next(iter(regions.values())).device

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Get shape of the region.

        For multi-input regions, this returns the shape of the first input.
        To get shapes for specific inputs, use region[input_id].shape.
        """
        return next(iter(self.regions.values())).shape

    @property
    def device(self) -> torch.device:
        """Get device of the region."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of the region (uses first input's dtype)."""
        return next(iter(self.regions.values())).dtype

    def to(self, device: str | torch.device) -> MultiInputRegion:
        """Move all regions to a device."""
        return MultiInputRegion({
            input_id: region.to(device)
            for input_id, region in self.regions.items()
        })

    def __getitem__(self, item) -> MultiInputRegion:
        return MultiInputRegion({
            input_id: region[item]
            for input_id, region in self.regions.items()
        })

    def get(self, input_id: int) -> HyperRectangle | None:
        """
        Get region for an input node with optional default.

        Args:
            input_id: Input node ID
            default: Default value if input_id not found

        Returns:
            Region for the input, or default if not found
        """
        return self.regions.get(input_id)

    def __contains__(self, input_id: int) -> bool:
        """Check if region contains bounds for an input node."""
        return input_id in self.regions

    def __len__(self) -> int:
        """Number of input nodes."""
        return len(self.regions)

    def maximize(self, direction: torch.Tensor) -> torch.Tensor:
        """
        Maximize over the first region (for single-input compatibility).

        Args:
            direction: Direction to maximize

        Returns:
            Maximum value over the first region
        """
        return sum(region.maximize(direction) for region in self.regions.values())

    def minimize(self, direction: torch.Tensor) -> torch.Tensor:
        """
        Minimize over the first region (for single-input compatibility).

        Args:
            direction: Direction to minimize

        Returns:
            Minimum value over the first region
        """
        return sum(region.minimize(direction) for region in self.regions.values())

    @classmethod
    def from_single_region(cls, region: HyperRectangle, input_id: int = 0) -> MultiInputRegion:
        """
        Create a MultiInputRegion from a single HyperRectangle.

        Useful for converting single-input regions to multi-input format.

        Args:
            region: HyperRectangle for the input
            input_id: Input node ID (default: 0)

        Returns:
            MultiInputRegion containing the single region
        """
        return cls({input_id: region})

    def keys(self):
        """Get iterator over input node IDs."""
        return self.regions.keys()

    def values(self):
        """Get iterator over regions."""
        return self.regions.values()

    def items(self):
        """Get iterator over (input_id, region) pairs."""
        return self.regions.items()
