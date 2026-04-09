"""
Configuration for bounding strategy execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategyConfig:
    """
    Configuration for bounding strategy execution.

    This allows customization of strategy behavior on a per-node basis.
    Different operations may need different settings for relaxations,
    optimization, etc.

    Attributes:
        same_slope: Whether to use the same slope for upper and lower bounds
                   in activation relaxations (relevant for LBP methods)
        custom_params: Dictionary of operation-specific custom parameters
    """

    same_slope: bool = True
    custom_params: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a custom parameter value.

        Args:
            key: Parameter name
            default: Default value if key not found

        Returns:
            Parameter value or default
        """
        return self.custom_params.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a custom parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        self.custom_params[key] = value
