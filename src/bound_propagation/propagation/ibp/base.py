from ...bounds import IntervalBounds
from ..strategy import ForwardBoundingStrategy


class ForwardIBPStrategy(ForwardBoundingStrategy[IntervalBounds]):
    """Base class for IBP (interval bound propagation) strategies."""

    @property
    def method_name(self) -> str:
        return "ibp"
