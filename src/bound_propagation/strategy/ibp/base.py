from ...bounds import IntervalBounds
from ..strategy import ForwardBoundingStrategy


class IntervalBoundingStrategy(ForwardBoundingStrategy[IntervalBounds]):

    @property
    def method_name(self) -> str:
        return "ibp"
