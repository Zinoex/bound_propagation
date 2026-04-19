"""Compiler passes for torch.fx graphs."""

from .graph_simplification import SimplificationPass, default_rewriters
from .metadata import MetadataPass

__all__ = ["MetadataPass", "SimplificationPass", "default_rewriters"]
