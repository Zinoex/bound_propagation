from ...bounds import LinearBounds
from ..strategy import ForwardBoundingStrategy


class ForwardLinearBoundingStrategy(ForwardBoundingStrategy[LinearBounds]):

    @property
    def method_name(self) -> str:
        return "forward"
