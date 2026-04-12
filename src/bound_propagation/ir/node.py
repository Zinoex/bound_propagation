"""
Node representation for computation graphs.

A Node represents a single operation in a computation graph, with inputs,
outputs, and associated metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch

from .metadata import TensorMetadata
from .operations import OperationType


class NodeType(StrEnum):
    """Classification of nodes by their role in the computation graph."""

    INPUT = "input"  # Graph input placeholder
    OPERATION = "operation"  # Standard operation node
    OUTPUT = "output"  # Graph output (may just be alias to operation node)
    CONSTANT = "constant"  # Constant value
    PARAMETER = "parameter"  # Learnable parameter


class AbstractValueType(StrEnum):
    """Abstract value types for dispatching bounding strategies."""

    CONSTANT = "constant"  # Known constant value (e.g., weights, fixed scalars)
    ABSTRACT = "abstract"  # Abstract bounds (e.g., IntervalBounds, LinearBounds)


@dataclass
class Node:
    """
    Represents a single operation node in a computation graph.

    A node encapsulates:
    - The operation type being performed
    - Input nodes (dependencies)
    - Output metadata (shape, dtype, device)
    - Operation-specific attributes
    - Unique identifier for reference

    Attributes:
        id: Unique identifier for this node (assigned by Graph)
        op_type: The type of operation this node performs
        inputs: List of input nodes this operation depends on
        output_metadata: Metadata describing the output tensor(s)
        attributes: Operation-specific configuration (e.g., axis for reduction)
        node_type: Classification of this node's role
        name: Optional human-readable name for debugging
    """

    id: int
    op_type: OperationType
    inputs: list[Node]
    output_metadata: TensorMetadata | tuple[TensorMetadata, ...]
    attributes: dict[str, Any] = field(default_factory=dict)
    input_signature: tuple[AbstractValueType, ...] | None = None
    output_signature: AbstractValueType | None = None
    node_type: NodeType = NodeType.OPERATION
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate node after initialization."""
        # Ensure inputs is a list
        if not isinstance(self.inputs, list):
            object.__setattr__(self, "inputs", list(self.inputs))

        # Validate node types match operation types
        if self.node_type == NodeType.INPUT and self.op_type != OperationType.INPUT:
            raise ValueError(f"INPUT node must have INPUT op_type, got {self.op_type}")

        if self.node_type == NodeType.CONSTANT and self.op_type != OperationType.CONSTANT:
            raise ValueError(f"CONSTANT node must have CONSTANT op_type, got {self.op_type}")

        if self.node_type == NodeType.PARAMETER and self.op_type != OperationType.PARAMETER:
            raise ValueError(f"PARAMETER node must have PARAMETER op_type, got {self.op_type}")

    @property
    def num_inputs(self) -> int:
        """Number of input nodes."""
        return len(self.inputs)

    @property
    def is_input(self) -> bool:
        """Check if this is an input node."""
        return self.node_type == NodeType.INPUT

    @property
    def is_output(self) -> bool:
        """Check if this is an output node."""
        return self.node_type == NodeType.OUTPUT

    @property
    def is_value(self) -> bool:
        """Check if this is a constant or parameter node."""
        return self.node_type in [NodeType.CONSTANT, NodeType.PARAMETER]

    @property
    def is_operation(self) -> bool:
        """Check if this is a standard operation node."""
        return self.node_type in [NodeType.OPERATION, NodeType.OUTPUT]

    def get_output_metadata(self, output_idx: int = 0) -> TensorMetadata:
        """
        Get output metadata for a specific output index.

        Args:
            output_idx: Index of the output (0 for single-output ops)

        Returns:
            TensorMetadata for the specified output

        Raises:
            IndexError: If output_idx is out of range
        """
        if isinstance(self.output_metadata, tuple):
            if output_idx < 0 or output_idx >= len(self.output_metadata):
                raise IndexError(
                    f"Node {self.id} has {len(self.output_metadata)} outputs, but index {output_idx} requested"
                )
            metadata = self.output_metadata[output_idx]
            assert isinstance(metadata, TensorMetadata)  # type: ignore[misc]
            return metadata
        if output_idx != 0:
            raise IndexError(f"Node {self.id} has single output, but index {output_idx} requested")
        return self.output_metadata

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get an attribute value, with optional default."""
        return self.attributes.get(key, default)

    def has_attribute(self, key: str) -> bool:
        """Check if an attribute exists."""
        return key in self.attributes

    @property
    def value(self) -> torch.Tensor:
        """Return the tensor value for constant or parameter nodes."""
        if self.node_type not in {NodeType.CONSTANT, NodeType.PARAMETER}:
            raise ValueError(f"Node {self.id} ({self.node_type}) does not hold a constant value")

        if "value" not in self.attributes:
            raise ValueError(f"Node {self.id} ({self.node_type}) is missing required 'value' attribute")

        value = self.attributes["value"]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Node {self.id} ({self.node_type}) has non-tensor value type {type(value)}")

        return value

    @property
    def resolved_input_signature(self) -> tuple[AbstractValueType, ...]:
        """Return node abstract input signature annotation required for dispatch."""
        if self.input_signature is None:
            raise ValueError(f"Node {self.id} is missing input_signature annotation")
        return self.input_signature

    @property
    def resolved_output_signature(self) -> AbstractValueType:
        """Return node abstract output signature annotation required for dispatch."""
        if self.output_signature is None:
            raise ValueError(f"Node {self.id} is missing output_signature annotation")
        return self.output_signature

    def validate_inputs(
        self,
        expected_count: int | None = None,
        min_count: int | None = None,
        max_count: int | None = None,
    ) -> None:
        """
        Validate the number of inputs to this node.

        Args:
            expected_count: Exact number of inputs expected (if specified)
            min_count: Minimum number of inputs (if specified)
            max_count: Maximum number of inputs (if specified)

        Raises:
            ValueError: If validation fails
        """
        num_inputs = self.num_inputs

        if expected_count is not None and num_inputs != expected_count:
            raise ValueError(f"Node {self.id} ({self.op_type}) expects {expected_count} inputs, got {num_inputs}")

        if min_count is not None and num_inputs < min_count:
            raise ValueError(f"Node {self.id} ({self.op_type}) expects at least {min_count} inputs, got {num_inputs}")

        if max_count is not None and num_inputs > max_count:
            raise ValueError(f"Node {self.id} ({self.op_type}) expects at most {max_count} inputs, got {num_inputs}")

    def __str__(self) -> str:
        """Human-readable string representation."""
        name_str = f" '{self.name}'" if self.name else ""
        inputs_str = f", {self.num_inputs} inputs" if not self.is_input else ""

        if isinstance(self.output_metadata, tuple):
            output_str = f", {len(self.output_metadata)} outputs"
        else:
            output_str = f" -> {self.output_metadata.shape}"

        return f"Node({self.id}{name_str}: {self.op_type}{inputs_str}{output_str})"

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (
            f"Node(id={self.id}, op_type={self.op_type}, "
            f"inputs=[{', '.join(str(n.id) for n in self.inputs)}], "
            f"output_metadata={self.output_metadata}, "
            f"node_type={self.node_type}, name={self.name!r})"
        )

    def __hash__(self) -> int:
        """Hash based on node ID (must be unique within graph)."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on node ID."""
        if not isinstance(other, Node):
            return NotImplemented
        return self.id == other.id
