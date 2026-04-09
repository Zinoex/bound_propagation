"""
Bilinear operation relaxation strategies.

This package contains relaxation strategies for bilinear operations
like MUL and DIV that require non-trivial relaxations.

Import this module to auto-register all bilinear relaxation strategies.
"""

# Import to trigger auto-registration
from bound_propagation.relaxations.bilinear.div import DivRelaxationStrategy
from bound_propagation.relaxations.bilinear.mul import MulRelaxationStrategy

__all__ = [
    "MulRelaxationStrategy",
    "DivRelaxationStrategy",
]
