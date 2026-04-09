"""
Backward-mode Linear Bound Propagation (Backward CROWN) strategies.

Backward LBP propagates linear bounds backwards through the network in reverse
topological order, computing how the output depends on each intermediate node.

This module now uses the new backward propagation algorithm (Algorithm 2 from auto_LiRPA).
Use BackwardLBPPropagator instead of the regular BoundPropagator for backward mode.
"""

from __future__ import annotations

from ...ir import OperationType

# Auto-register strategies on module import
_register_backward_strategies()
