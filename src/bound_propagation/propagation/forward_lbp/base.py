from ...bounds import LinearBounds
from ..strategy import ForwardBoundingStrategy


class ForwardLBPStrategy(ForwardBoundingStrategy[LinearBounds]):

    @property
    def method_name(self) -> str:
        return "forward"
