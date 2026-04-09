from ...bounds import LinearBounds
from ..strategy import BackwardBoundingStrategy


class BackwardLBPBoundingStrategy(BackwardBoundingStrategy[LinearBounds]):
    """Backward-mode Linear Bound Propagation (LBP) strategy base class."""

    @property
    def method_name(self) -> str:
        """Return the method name for this strategy."""
        return "backward"
