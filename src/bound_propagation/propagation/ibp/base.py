from ...bounds import IntervalBounds
from ..strategy import ForwardBoundingStrategy


class ForwardIBPStrategy(ForwardBoundingStrategy[IntervalBounds]):
    @property
    def method_name(self) -> str:
        return "ibp"
