"""
Convert torch.fx.GraphModule to internal IR Graph representation.

This module handles the conversion from PyTorch's torch.fx intermediate representation
to our custom IR that is optimized for bound propagation analysis.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.fx as fx

from ..ir import Graph, Node, NodeType, OperationType, TensorMetadata
from .op_mapping import get_operation_type


class ConversionError(Exception):
    """Exception raised when graph conversion fails."""

    pass


class GraphConverter:
    """
    Converts torch.fx.GraphModule to internal IR Graph.

    This converter:
    - Maps torch.fx nodes to IR nodes
    - Extracts tensor metadata (shape, dtype, device)
    - Handles multi-input/multi-output operations
    - Validates graph structure
    - Preserves operation attributes
    """

    def __init__(self, fx_graph_module: fx.GraphModule) -> None:
        """
        Initialize converter with fx.GraphModule.

        Args:
            fx_graph_module: The traced torch.fx.GraphModule to convert
        """
        self.fx_module = fx_graph_module
        self.fx_to_ir: dict[fx.Node, Node] = {}  # Map fx.Node -> IR Node
        self.node_id_counter = 0

    def convert(self, example_inputs: tuple[torch.Tensor, ...] | None = None) -> Graph:
        """
        Convert fx.GraphModule to IR Graph.

        Args:
            example_inputs: Optional example inputs for shape inference.
                If not provided, shapes will be symbolic/unknown.

        Returns:
            Converted IR Graph

        Raises:
            ConversionError: If conversion fails
        """
        # Run shape propagation if example inputs provided
        if example_inputs is not None:
            self._run_shape_propagation(example_inputs)

        ir_nodes: list[Node] = []
        output_nodes: list[Node] = []

        # Convert each fx.Node to IR Node
        for fx_node in self.fx_module.graph.nodes:
            if fx_node.op == "placeholder":
                ir_node = self._convert_input_node(fx_node)
            elif fx_node.op == "get_attr":
                ir_node = self._convert_parameter_node(fx_node)
            elif fx_node.op in ["call_function", "call_method", "call_module"]:
                ir_node = self._convert_operation_node(fx_node)
            elif fx_node.op == "output":
                # Mark output nodes from output specification
                output_nodes = self._extract_output_nodes(fx_node)
                continue  # Don't create IR node for output marker
            else:
                raise ConversionError(f"Unsupported fx node type: {fx_node.op}")

            self.fx_to_ir[fx_node] = ir_node
            ir_nodes.append(ir_node)

        # Create IR graph
        graph = Graph(ir_nodes)

        # Mark outputs
        if output_nodes:
            graph.mark_outputs(output_nodes)
        else:
            graph.infer_outputs()

        # Annotate graph nodes with constant/abstract input/output signatures.
        graph.annotate_input_kinds()

        # Validate graph
        graph.validate()

        return graph

    def _run_shape_propagation(self, example_inputs: tuple[torch.Tensor, ...]) -> None:
        """
        Run shape propagation through fx graph using example inputs.

        This executes the graph with example inputs to capture actual tensor
        metadata (shape, dtype, device) at each node.

        Args:
            example_inputs: Example tensor inputs
        """
        # Use fx's shape propagation
        from torch.fx.passes.shape_prop import ShapeProp

        try:
            ShapeProp(self.fx_module).propagate(*example_inputs)
        except Exception as e:
            raise ConversionError(f"Shape propagation failed: {e}") from e

    def _convert_input_node(self, fx_node: fx.Node) -> Node:
        """Convert fx placeholder (input) node to IR Node."""
        metadata = self._extract_metadata(fx_node)
        if metadata is None:
            # Create unknown metadata for symbolic shapes
            metadata = TensorMetadata(shape=(-1,), dtype="float32")

        ir_node = Node(
            id=self._next_node_id(),
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            node_type=NodeType.INPUT,
            name=str(fx_node.name),
        )
        return ir_node

    def _convert_parameter_node(self, fx_node: fx.Node) -> Node:
        """Convert fx get_attr (parameter/buffer) node to IR Node."""
        # Get the actual parameter tensor
        target_str = str(fx_node.target)
        param = self._get_attribute(target_str)

        # Convert device string to torch.device
        device = torch.device(param.device)

        metadata = TensorMetadata(shape=tuple(param.shape), dtype=str(param.dtype).replace("torch.", ""), device=device)

        ir_node = Node(
            id=self._next_node_id(),
            op_type=OperationType.PARAMETER,
            inputs=[],
            output_metadata=metadata,
            node_type=NodeType.PARAMETER,
            name=str(fx_node.target),
            attributes={"value": param},
        )
        return ir_node

    def _convert_operation_node(self, fx_node: fx.Node) -> Node:
        """Convert fx operation node to IR Node."""
        # Get operation type
        op_type = get_operation_type(fx_node.target)

        # For call_module nodes, check the actual module type
        if op_type is None and fx_node.op == "call_module":
            target_str = str(fx_node.target)
            module = self.fx_module.get_submodule(target_str)
            op_type = get_operation_type(type(module))

        if op_type is None:
            raise ConversionError(f"Unsupported operation: {fx_node.target} at node {fx_node.name}")

        # Get input nodes
        input_nodes = self._get_input_nodes(fx_node)

        # Extract metadata
        metadata = self._extract_metadata(fx_node)
        if metadata is None:
            # Infer metadata from inputs if possible
            metadata = self._infer_output_metadata(op_type, input_nodes, fx_node)

        # Extract operation attributes
        attributes = self._extract_attributes(fx_node, op_type)

        ir_node = Node(
            id=self._next_node_id(),
            op_type=op_type,
            inputs=input_nodes,
            output_metadata=metadata,
            attributes=attributes,
            node_type=NodeType.OPERATION,
            name=str(fx_node.name),
        )
        return ir_node

    def _get_input_nodes(self, fx_node: fx.Node) -> list[Node]:
        """Get IR nodes corresponding to fx node's inputs."""
        input_nodes: list[Node] = []

        for arg in fx_node.args:
            if isinstance(arg, fx.Node):
                if arg not in self.fx_to_ir:
                    raise ConversionError(f"Input node {arg.name} not yet converted (topological order issue)")
                input_nodes.append(self.fx_to_ir[arg])
            # Skip non-Node arguments (constants, etc.)

        return input_nodes

    def _extract_output_nodes(self, fx_output_node: fx.Node) -> list[Node]:
        """Extract IR nodes from fx output node."""
        output_nodes: list[Node] = []

        # The output node's args contain the returned values
        output_val = fx_output_node.args[0]

        if isinstance(output_val, fx.Node):
            output_nodes.append(self.fx_to_ir[output_val])
        elif isinstance(output_val, (tuple, list)):
            for val in output_val:
                if isinstance(val, fx.Node):
                    output_nodes.append(self.fx_to_ir[val])
        else:
            raise ConversionError(f"Unsupported output type: {type(output_val)}")

        return output_nodes

    def _extract_metadata(self, fx_node: fx.Node) -> TensorMetadata | None:
        """
        Extract tensor metadata from fx node.

        Returns None if metadata not available (no shape propagation).
        """
        if not hasattr(fx_node, "meta") or "tensor_meta" not in fx_node.meta:
            return None

        tensor_meta = fx_node.meta["tensor_meta"]

        # Extract shape, dtype, device
        shape = tuple(tensor_meta.shape)
        dtype = str(tensor_meta.dtype)

        # Handle device - may be missing or have different name
        if hasattr(tensor_meta, "device"):
            device = torch.device(tensor_meta.device)
        else:
            device = torch.device('cpu')  # Default to CPU if not specified

        return TensorMetadata(shape=shape, dtype=dtype, device=device)

    def _infer_output_metadata(self, op_type: OperationType, input_nodes: list[Node], fx_node: fx.Node) -> TensorMetadata:
        """
        Infer output metadata when shape propagation not available.

        This is a fallback for when we don't have concrete shapes.
        """
        if not input_nodes:
            # No inputs - use symbolic shape
            return TensorMetadata(shape=(-1,), dtype="float32")

        # For most elementwise ops, output shape matches input shape
        if op_type.is_elementwise and len(input_nodes) > 0:
            output = input_nodes[0].output_metadata
            # Handle potentially multi-output nodes
            if isinstance(output, tuple):
                # Cast first element to TensorMetadata
                return cast(TensorMetadata, output[0])
            return output

        # For other ops, use symbolic shape
        return TensorMetadata(shape=(-1,), dtype="float32")

    def _extract_attributes(self, fx_node: fx.Node, op_type: OperationType) -> dict[str, Any]:
        """
        Extract operation-specific attributes from fx node.

        Examples:
        - dim/axis for reductions (sum, mean)
        - keepdim for reductions
        - min/max for clamp
        """
        attributes: dict[str, Any] = {}

        # Extract from kwargs
        if fx_node.kwargs:
            attributes.update(fx_node.kwargs)

        # Operation-specific attribute extraction
        if op_type in [OperationType.SUM, OperationType.MEAN, OperationType.MAX, OperationType.MIN]:
            # Reduction operations: extract dim and keepdim
            if "dim" in attributes or (len(fx_node.args) > 1 and isinstance(fx_node.args[1], int)):
                dim = attributes.get("dim", fx_node.args[1] if len(fx_node.args) > 1 else None)
                if dim is not None:
                    attributes["dim"] = dim
            if "keepdim" in attributes:
                attributes["keepdim"] = attributes["keepdim"]

        elif op_type == OperationType.CLAMP:
            # Clamp: extract min and max
            if "min" in attributes:
                attributes["min"] = attributes["min"]
            if "max" in attributes:
                attributes["max"] = attributes["max"]

        elif op_type in [OperationType.TRANSPOSE, OperationType.PERMUTE]:
            # Transpose/permute: extract dimensions
            if len(fx_node.args) > 1:
                attributes["dims"] = fx_node.args[1:]

        return attributes

    def _get_attribute(self, target: str) -> Any:
        """Get attribute from fx module by name."""
        attrs = target.split(".")
        obj = self.fx_module
        for attr in attrs:
            obj = getattr(obj, attr)
        return obj

    def _next_node_id(self) -> int:
        """Get next unique node ID."""
        node_id = self.node_id_counter
        self.node_id_counter += 1
        return node_id
