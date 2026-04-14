from ...bounds import LinearBounds
from ..strategy import ForwardBoundingStrategy


class ForwardLBPStrategy(ForwardBoundingStrategy[LinearBounds]):
    """Base class for forward linear bound propagation strategies."""

    @property
    def method_name(self) -> str:
        return "forward_lbp"
