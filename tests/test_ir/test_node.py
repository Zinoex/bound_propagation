"""
Tests for computation graph nodes.
"""

import pytest

from bound_propagation.ir import Node, NodeType, OperationType, TensorMetadata


class TestNodeType:
    """Tests for NodeType enum."""

    def test_node_types(self):
        """Test all node type values."""
        assert NodeType.INPUT == "input"
        assert NodeType.OPERATION == "operation"
        assert NodeType.OUTPUT == "output"
        assert NodeType.CONSTANT == "constant"
        assert NodeType.PARAMETER == "parameter"

    def test_node_type_is_string(self):
        """Test that NodeType values are strings."""
        for node_type in NodeType:
            assert isinstance(node_type, str)


class TestNode:
    """Tests for Node dataclass."""

    @pytest.fixture
    def sample_metadata(self):
        """Sample tensor metadata for testing."""
        return TensorMetadata(shape=(2, 3), dtype="float32")

    @pytest.fixture
    def input_node(self, sample_metadata):
        """Sample input node."""
        return Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT, name="input_0")

    @pytest.fixture
    def operation_node(self, sample_metadata, input_node):
        """Sample operation node."""
        return Node(id=1, op_type=OperationType.RELU, inputs=[input_node], output_metadata=sample_metadata, node_type=NodeType.OPERATION, name="relu_1")

    def test_basic_creation(self, sample_metadata):
        """Test basic node creation."""
        node = Node(id=0, op_type=OperationType.RELU, inputs=[], output_metadata=sample_metadata)
        assert node.id == 0
        assert node.inputs == []
        assert node.output_metadata == sample_metadata
        assert node.attributes == {}
        assert node.node_type == NodeType.OPERATION  # default
        assert node.name is None
        assert node.op_type == OperationType.RELU

    def test_creation_with_all_parameters(self, sample_metadata, input_node):
        """Test node creation with all parameters."""
        attributes = {"alpha": 0.5, "beta": 1.0}
        node = Node(
            id=1,
            op_type=OperationType.ADD,
            inputs=[input_node],
            output_metadata=sample_metadata,
            attributes=attributes,
            node_type=NodeType.OPERATION,
            name="add_layer",
        )
        assert node.id == 1
        assert node.op_type == OperationType.ADD
        assert node.inputs == [input_node]
        assert node.output_metadata == sample_metadata
        assert node.attributes == attributes
        assert node.node_type == NodeType.OPERATION
        assert node.name == "add_layer"

    def test_is_input_property(self, input_node, operation_node):
        """Test is_input property."""
        assert input_node.is_input is True
        assert operation_node.is_input is False

        constant_node = Node(id=2, op_type=OperationType.CONSTANT, inputs=[], output_metadata=input_node.output_metadata, node_type=NodeType.CONSTANT)
        assert constant_node.is_input is False

    def test_is_value_property_for_constant(self):
        """Test is_value property for constant nodes."""
        metadata = TensorMetadata(shape=(2, 3))
        constant_node = Node(id=0, op_type=OperationType.CONSTANT, inputs=[], output_metadata=metadata, node_type=NodeType.CONSTANT)
        assert constant_node.is_value is True

        input_node = Node(id=1, op_type=OperationType.INPUT, inputs=[], output_metadata=metadata, node_type=NodeType.INPUT)
        assert input_node.is_value is False

    def test_is_value_property_for_parameter(self):
        """Test is_value property for parameter nodes."""
        metadata = TensorMetadata(shape=(2, 3))
        param_node = Node(id=0, op_type=OperationType.PARAMETER, inputs=[], output_metadata=metadata, node_type=NodeType.PARAMETER)
        assert param_node.is_value is True

        input_node = Node(id=1, op_type=OperationType.INPUT, inputs=[], output_metadata=metadata, node_type=NodeType.INPUT)
        assert input_node.is_value is False

    def test_is_operation_property(self, input_node, operation_node):
        """Test is_operation property."""
        assert input_node.is_operation is False
        assert operation_node.is_operation is True

    def test_get_output_metadata_single_output(self, operation_node):
        """Test get_output_metadata for single-output operations."""
        metadata = operation_node.get_output_metadata(0)
        assert metadata == operation_node.output_metadata

        # Default index should work
        metadata_default = operation_node.get_output_metadata()
        assert metadata_default == operation_node.output_metadata

    def test_get_output_metadata_multi_output(self, sample_metadata):
        """Test get_output_metadata for multi-output operations."""
        meta2 = TensorMetadata(shape=(3, 4))
        meta3 = TensorMetadata(shape=(4, 5))
        multi_output_node = Node(id=0, op_type=OperationType.SPLIT, inputs=[], output_metadata=(sample_metadata, meta2, meta3))

        assert multi_output_node.get_output_metadata(0) == sample_metadata
        assert multi_output_node.get_output_metadata(1) == meta2
        assert multi_output_node.get_output_metadata(2) == meta3

    def test_get_output_metadata_invalid_index_single(self, operation_node):
        """Test get_output_metadata with invalid index for single output."""
        with pytest.raises(IndexError, match="has single output"):
            operation_node.get_output_metadata(1)

    def test_get_output_metadata_invalid_index_multi(self, sample_metadata):
        """Test get_output_metadata with invalid index for multi output."""
        meta2 = TensorMetadata(shape=(3, 4))
        multi_output_node = Node(id=0, op_type=OperationType.SPLIT, inputs=[], output_metadata=(sample_metadata, meta2))

        with pytest.raises(IndexError, match="has 2 outputs"):
            multi_output_node.get_output_metadata(2)

        with pytest.raises(IndexError, match="has 2 outputs"):
            multi_output_node.get_output_metadata(-1)

    def test_get_attribute(self, sample_metadata):
        """Test get_attribute method."""
        attributes = {"alpha": 0.5, "beta": 1.0}
        node = Node(id=0, op_type=OperationType.RELU, inputs=[], output_metadata=sample_metadata, attributes=attributes)

        assert node.get_attribute("alpha") == 0.5
        assert node.get_attribute("beta") == 1.0
        assert node.get_attribute("gamma") is None
        assert node.get_attribute("gamma", default=2.0) == 2.0

    def test_has_attribute(self, sample_metadata):
        """Test has_attribute method."""
        attributes = {"alpha": 0.5, "beta": 1.0}
        node = Node(id=0, op_type=OperationType.RELU, inputs=[], output_metadata=sample_metadata, attributes=attributes)

        assert node.has_attribute("alpha") is True
        assert node.has_attribute("beta") is True
        assert node.has_attribute("gamma") is False

    def test_validate_inputs_count(self, sample_metadata, input_node):
        """Test validate_inputs with expected count."""
        # Node with 1 input
        node = Node(id=1, op_type=OperationType.RELU, inputs=[input_node], output_metadata=sample_metadata)
        node.validate_inputs(expected_count=1)  # Should not raise

        with pytest.raises(ValueError, match="expects 2 inputs"):
            node.validate_inputs(expected_count=2)

    def test_validate_inputs_types(self, sample_metadata, input_node):
        """Test validate_inputs with min/max counts."""
        operation_node = Node(id=1, op_type=OperationType.ADD, inputs=[input_node], output_metadata=sample_metadata)

        # Validate with min count
        operation_node.validate_inputs(min_count=1)  # Should not raise

        with pytest.raises(ValueError, match="at least 2 inputs"):
            operation_node.validate_inputs(min_count=2)

    def test_validate_inputs_combined(self, sample_metadata, input_node):
        """Test validate_inputs with both min and max."""
        node1 = Node(id=1, op_type=OperationType.ADD, inputs=[input_node], output_metadata=sample_metadata)
        node2 = Node(id=2, op_type=OperationType.MUL, inputs=[input_node, node1], output_metadata=sample_metadata)

        # Should not raise (2 inputs, min=1, max=3)
        node2.validate_inputs(min_count=1, max_count=3)

        with pytest.raises(ValueError, match="at most 1 inputs"):
            node2.validate_inputs(max_count=1)

    def test_validate_compatible_shapes(self, input_node):
        """Test validate_inputs with shape compatibility."""
        # Create operation node with compatible shapes
        node1 = Node(id=1, op_type=OperationType.ADD, inputs=[input_node], output_metadata=input_node.output_metadata)

        # This should not raise
        node1.validate_inputs()

    def test_node_with_multiple_inputs(self, sample_metadata, input_node):
        """Test node with multiple inputs."""
        input_node2 = Node(id=10, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)
        add_node = Node(id=2, op_type=OperationType.ADD, inputs=[input_node, input_node2], output_metadata=sample_metadata)

        assert len(add_node.inputs) == 2
        assert input_node in add_node.inputs
        assert input_node2 in add_node.inputs

    def test_node_with_empty_inputs(self, sample_metadata):
        """Test node with no inputs (input/constant nodes)."""
        input_node = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)
        assert len(input_node.inputs) == 0

        constant_node = Node(id=1, op_type=OperationType.CONSTANT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.CONSTANT)
        assert len(constant_node.inputs) == 0

    def test_node_id_uniqueness(self, sample_metadata):
        """Test that nodes can have different IDs."""
        node1 = Node(id=0, op_type=OperationType.RELU, inputs=[], output_metadata=sample_metadata)
        node2 = Node(id=1, op_type=OperationType.RELU, inputs=[], output_metadata=sample_metadata)
        node3 = Node(id=100, op_type=OperationType.RELU, inputs=[], output_metadata=sample_metadata)

        assert node1.id != node2.id
        assert node2.id != node3.id
        assert node1.id != node3.id

    def test_node_with_operation_type(self, sample_metadata, input_node):
        """Test node with various operation types."""
        for op_type in [OperationType.RELU, OperationType.ADD, OperationType.MATMUL, OperationType.SIN]:
            node = Node(id=1, op_type=op_type, inputs=[input_node], output_metadata=sample_metadata)
            assert node.op_type == op_type

    def test_node_attributes_default_empty_dict(self, sample_metadata):
        """Test that attributes default to empty dict."""
        node = Node(id=0, op_type=OperationType.RELU, inputs=[], output_metadata=sample_metadata)
        assert node.attributes == {}
        assert isinstance(node.attributes, dict)

    def test_node_attributes_are_mutable(self, sample_metadata):
        """Test that node attributes can be modified."""
        node = Node(id=0, op_type=OperationType.RELU, inputs=[], output_metadata=sample_metadata)
        node.attributes["key"] = "value"
        assert node.attributes["key"] == "value"

    def test_node_name_optional(self, sample_metadata):
        """Test that node name is optional."""
        node_with_name = Node(id=0, op_type=OperationType.RELU, inputs=[], output_metadata=sample_metadata, name="my_layer")
        assert node_with_name.name == "my_layer"

        node_without_name = Node(id=1, op_type=OperationType.RELU, inputs=[], output_metadata=sample_metadata)
        assert node_without_name.name is None

    def test_repr(self, input_node):
        """Test string representation."""
        repr_str = repr(input_node)
        assert "Node" in repr_str
        assert "id=0" in repr_str
        assert "input" in repr_str.lower()
