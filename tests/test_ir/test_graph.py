"""
Tests for computation graph representation.
"""

import pytest

from bound_propagation.ir import DeviceType, Graph, Node, NodeType, OperationType, TensorMetadata


class TestGraph:
    """Tests for Graph class."""

    @pytest.fixture
    def sample_metadata(self):
        """Sample tensor metadata."""
        return TensorMetadata(shape=(2, 3), dtype="float32", device=DeviceType.CPU)

    @pytest.fixture
    def simple_graph(self, sample_metadata):
        """Create a simple linear graph: input -> relu -> output."""
        input_node = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT, name="x")
        relu_node = Node(id=1, op_type=OperationType.RELU, inputs=[input_node], output_metadata=sample_metadata, name="relu")

        graph = Graph([input_node, relu_node])
        graph.mark_outputs([relu_node])
        return graph, input_node, relu_node

    @pytest.fixture
    def multi_input_graph(self, sample_metadata):
        """Create graph with multiple inputs: x1, x2 -> add -> relu -> output."""
        x1 = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT, name="x1")
        x2 = Node(id=1, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT, name="x2")
        add_node = Node(id=2, op_type=OperationType.ADD, inputs=[x1, x2], output_metadata=sample_metadata, name="add")
        relu_node = Node(id=3, op_type=OperationType.RELU, inputs=[add_node], output_metadata=sample_metadata, name="relu")

        graph = Graph([x1, x2, add_node, relu_node])
        graph.mark_outputs([relu_node])
        return graph, [x1, x2], [add_node, relu_node]

    def test_empty_graph_creation(self):
        """Test creating an empty graph."""
        graph = Graph([])
        assert graph.num_nodes == 0
        assert graph.num_inputs == 0
        assert graph.num_outputs == 0
        assert len(graph.nodes) == 0

    def test_graph_creation_with_nodes(self, simple_graph):
        """Test creating graph with initial nodes."""
        graph, input_node, relu_node = simple_graph
        assert graph.num_nodes == 2
        assert graph.num_inputs == 1
        assert graph.num_outputs == 1

    def test_graph_creation_with_single_node(self, sample_metadata):
        """Test creating a graph with a single node."""
        node = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)

        graph = Graph([node])
        assert graph.num_nodes == 1
        assert graph.has_node(0)
        assert graph.nodes == [node]

    def test_graph_preserves_nodes_with_duplicate_ids(self, sample_metadata):
        """Test that graph construction preserves the provided node sequence."""
        node1 = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)
        node2 = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)  # Same ID

        graph = Graph([node1, node2])

        assert graph.num_nodes == 2
        assert graph.nodes == [node1, node2]

    def test_nodes_can_be_filtered_by_id(self, simple_graph):
        """Test accessing nodes by ID through the nodes property."""
        graph, input_node, relu_node = simple_graph
        nodes_by_id = {node.id: node for node in graph.nodes}
        assert nodes_by_id[0] == input_node
        assert nodes_by_id[1] == relu_node

    def test_missing_node_id_is_reported_by_has_node(self, simple_graph):
        """Test that missing node IDs are reported by has_node."""
        graph, _, _ = simple_graph
        assert graph.has_node(999) is False

    def test_has_node(self, simple_graph):
        """Test checking if node exists."""
        graph, _, _ = simple_graph
        assert graph.has_node(0) is True
        assert graph.has_node(1) is True
        assert graph.has_node(2) is False
        assert graph.has_node(999) is False

    def test_nodes_property(self, simple_graph):
        """Test nodes property returns sorted list."""
        graph, input_node, relu_node = simple_graph
        nodes = graph.nodes
        assert len(nodes) == 2
        assert nodes[0] == input_node
        assert nodes[1] == relu_node
        # Should be sorted by ID
        assert nodes[0].id < nodes[1].id

    def test_input_nodes_property(self, multi_input_graph):
        """Test input_nodes property."""
        graph, inputs, _ = multi_input_graph
        input_nodes = graph.input_nodes
        assert len(input_nodes) == 2
        assert inputs[0] in input_nodes
        assert inputs[1] in input_nodes

    def test_output_nodes_property(self, simple_graph):
        """Test output_nodes property."""
        graph, _, relu_node = simple_graph
        output_nodes = graph.output_nodes
        assert len(output_nodes) == 1
        assert output_nodes[0] == relu_node

    def test_mark_outputs(self, sample_metadata):
        """Test marking specific nodes as outputs."""
        x = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)
        y = Node(id=1, op_type=OperationType.RELU, inputs=[x], output_metadata=sample_metadata)
        z = Node(id=2, op_type=OperationType.TANH, inputs=[y], output_metadata=sample_metadata)

        graph = Graph([x, y, z])
        graph.mark_outputs([z])

        assert len(graph.output_nodes) == 1
        assert z in graph.output_nodes

        # Can mark multiple outputs
        graph.mark_outputs([y, z])
        assert len(graph.output_nodes) == 2

    def test_mark_outputs_with_invalid_node_raises_error(self, sample_metadata):
        """Test marking non-graph node as output raises error."""
        graph = Graph([])
        node = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)

        with pytest.raises(ValueError, match="not in graph"):
            graph.mark_outputs([node])

    def test_infer_outputs(self, sample_metadata):
        """Test automatic output inference."""
        x = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)
        y = Node(id=1, op_type=OperationType.RELU, inputs=[x], output_metadata=sample_metadata)
        z = Node(id=2, op_type=OperationType.TANH, inputs=[y], output_metadata=sample_metadata)

        graph = Graph([x, y, z])
        graph.infer_outputs()

        # z should be inferred as output (no downstream consumers)
        assert len(graph.output_nodes) == 1
        assert z in graph.output_nodes

    def test_infer_outputs_with_multiple_branches(self, sample_metadata):
        """Test output inference with multiple output branches."""
        x = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)
        y1 = Node(id=1, op_type=OperationType.RELU, inputs=[x], output_metadata=sample_metadata)
        y2 = Node(id=2, op_type=OperationType.TANH, inputs=[x], output_metadata=sample_metadata)

        graph = Graph([x, y1, y2])
        graph.infer_outputs()

        # Both y1 and y2 should be outputs
        assert len(graph.output_nodes) == 2
        assert y1 in graph.output_nodes
        assert y2 in graph.output_nodes

    def test_topological_order(self, simple_graph):
        """Test topological ordering of nodes."""
        graph, input_node, relu_node = simple_graph
        order = graph.topological_order()

        assert len(order) == 2
        assert order[0] == input_node
        assert order[1] == relu_node
        # Input should come before operation that uses it
        assert order.index(input_node) < order.index(relu_node)

    def test_topological_order_complex(self, multi_input_graph):
        """Test topological order on more complex graph."""
        graph, inputs, ops = multi_input_graph
        order = graph.topological_order()

        assert len(order) == 4
        # Inputs should come first
        for input_node in inputs:
            assert input_node in order[:2]

        # Operations should come after their inputs
        add_node, relu_node = ops
        assert order.index(add_node) > max(order.index(inputs[0]), order.index(inputs[1]))
        assert order.index(relu_node) > order.index(add_node)

    def test_topological_order_is_cached(self, simple_graph):
        """Test that topological order is cached."""
        graph, _, _ = simple_graph
        order1 = graph.topological_order()
        order2 = graph.topological_order()

        # Should return same object (cached)
        assert order1 is order2

    def test_reverse_topological_order(self, simple_graph):
        """Test reverse topological ordering."""
        graph, input_node, relu_node = simple_graph
        rev_order = graph.reverse_topological_order()

        assert len(rev_order) == 2
        assert rev_order[0] == relu_node
        assert rev_order[1] == input_node
        # Should be exact reverse of forward order
        forward_order = graph.topological_order()
        assert rev_order == list(reversed(forward_order))

    def test_validate_valid_graph(self, simple_graph):
        """Test validation of valid graph."""
        graph, _, _ = simple_graph
        graph.validate()  # Should not raise

    def test_validate_graph_with_no_inputs_raises_error(self, sample_metadata):
        """Test validation fails when graph has no inputs."""
        node = Node(id=0, op_type=OperationType.RELU, inputs=[], output_metadata=sample_metadata)
        graph = Graph([node])

        with pytest.raises(ValueError, match="no input nodes"):
            graph.validate()

    def test_validate_graph_with_no_outputs_infers_them(self, sample_metadata):
        """Test validation infers outputs if not specified."""
        x = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)
        y = Node(id=1, op_type=OperationType.RELU, inputs=[x], output_metadata=sample_metadata)
        graph = Graph([x, y])

        # Don't mark outputs explicitly
        graph.validate()  # Should infer y as output

        assert len(graph.output_nodes) == 1
        assert y in graph.output_nodes

    def test_validate_input_node_with_inputs_raises_error(self, sample_metadata):
        """Test validation fails when input node has inputs."""
        dummy = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)
        invalid_input = Node(id=1, op_type=OperationType.INPUT, inputs=[dummy], output_metadata=sample_metadata, node_type=NodeType.INPUT)

        graph = Graph([dummy, invalid_input])
        graph.mark_outputs([invalid_input])

        with pytest.raises(ValueError, match="Input node.*has inputs"):
            graph.validate()

    def test_validate_node_with_missing_input_raises_error(self, sample_metadata):
        """Test validation fails when node references missing input."""
        input_node = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)
        external_node = Node(id=99, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)
        node_with_external_input = Node(id=1, op_type=OperationType.RELU, inputs=[external_node], output_metadata=sample_metadata)

        graph = Graph([input_node, node_with_external_input])  # external_node not in graph
        graph.mark_outputs([node_with_external_input])

        with pytest.raises(ValueError, match="not in graph"):
            graph.validate()

    def test_get_dependencies_direct(self, multi_input_graph):
        """Test getting direct dependencies."""
        graph, inputs, ops = multi_input_graph
        add_node, relu_node = ops

        # Add node depends on both inputs
        add_deps = graph.get_dependencies(add_node, recursive=False)
        assert len(add_deps) == 2
        assert inputs[0] in add_deps
        assert inputs[1] in add_deps

        # Relu depends on add
        relu_deps = graph.get_dependencies(relu_node, recursive=False)
        assert len(relu_deps) == 1
        assert add_node in relu_deps

    def test_get_dependencies_recursive(self, multi_input_graph):
        """Test getting transitive dependencies."""
        graph, inputs, ops = multi_input_graph
        _, relu_node = ops

        # Relu's transitive dependencies include add and both inputs
        relu_deps = graph.get_dependencies(relu_node, recursive=True)
        assert len(relu_deps) == 3
        assert inputs[0] in relu_deps
        assert inputs[1] in relu_deps

    def test_get_dependents_direct(self, multi_input_graph):
        """Test getting direct dependents."""
        graph, inputs, ops = multi_input_graph
        add_node, relu_node = ops

        # Both inputs are used by add
        x1_dependents = graph.get_dependents(inputs[0], recursive=False)
        assert add_node in x1_dependents

        # Add is used by relu
        add_dependents = graph.get_dependents(add_node, recursive=False)
        assert relu_node in add_dependents

    def test_get_dependents_recursive(self, multi_input_graph):
        """Test getting transitive dependents."""
        graph, inputs, ops = multi_input_graph
        add_node, relu_node = ops

        # Input's transitive dependents include both add and relu
        x1_dependents = graph.get_dependents(inputs[0], recursive=True)
        assert len(x1_dependents) == 2
        assert add_node in x1_dependents
        assert relu_node in x1_dependents

    def test_len(self, simple_graph):
        """Test __len__ returns number of nodes."""
        graph, _, _ = simple_graph
        assert len(graph) == 2

    def test_contains_with_node(self, simple_graph):
        """Test __contains__ with Node object."""
        graph, input_node, relu_node = simple_graph
        assert input_node in graph
        assert relu_node in graph

        other_node = Node(id=999, op_type=OperationType.INPUT, inputs=[], output_metadata=input_node.output_metadata, node_type=NodeType.INPUT)
        assert other_node not in graph

    def test_contains_with_id(self, simple_graph):
        """Test __contains__ with node ID."""
        graph, _, _ = simple_graph
        assert 0 in graph
        assert 1 in graph
        assert 999 not in graph

    def test_iter(self, simple_graph):
        """Test __iter__ iterates in topological order."""
        graph, input_node, relu_node = simple_graph
        nodes_iterated = list(graph)

        assert len(nodes_iterated) == 2
        assert nodes_iterated[0] == input_node
        assert nodes_iterated[1] == relu_node

    def test_str(self, simple_graph):
        """Test __str__ representation."""
        graph, _, _ = simple_graph
        str_repr = str(graph)

        assert "Graph" in str_repr
        assert "2 nodes" in str_repr
        assert "1 inputs" in str_repr
        assert "1 outputs" in str_repr

    def test_repr(self, simple_graph):
        """Test __repr__ representation."""
        graph, _, _ = simple_graph
        repr_str = repr(graph)

        assert "Graph" in repr_str
        assert "nodes=" in repr_str
        assert "inputs=" in repr_str
        assert "outputs=" in repr_str

    def test_topological_order_updates_when_graph_is_rebuilt(self, sample_metadata):
        """Test that rebuilding a graph with new nodes produces a new order."""
        x = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)
        graph = Graph([x])
        graph.mark_outputs([x])

        order1 = graph.topological_order()

        y = Node(id=1, op_type=OperationType.RELU, inputs=[x], output_metadata=sample_metadata)
        expanded_graph = Graph([x, y])
        expanded_graph.mark_outputs([y])

        order2 = expanded_graph.topological_order()
        assert order1 is not order2
        assert len(order2) == 2

    def test_cyclic_graph_detection(self, sample_metadata):
        """Test that cyclic graphs are detected."""
        # Create a cycle: a -> b -> c -> a
        node_a = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)
        node_b = Node(id=1, op_type=OperationType.RELU, inputs=[node_a], output_metadata=sample_metadata)
        node_c = Node(id=2, op_type=OperationType.TANH, inputs=[node_b], output_metadata=sample_metadata)

        # Create cycle by making node_a depend on node_c
        # (In practice, this would be caught at construction, but testing graph validation)
        node_a.inputs.append(node_c)  # type: ignore[arg-type]

        graph = Graph([node_a, node_b, node_c])
        graph.mark_outputs([node_c])

        with pytest.raises(ValueError, match="cycle"):
            graph.topological_order()

    def test_complex_graph_structure(self, sample_metadata):
        """Test a more complex graph structure."""
        #     x1   x2
        #      \  /
        #      add
        #     /   \
        #   relu  tanh
        #     \   /
        #      mul
        x1 = Node(id=0, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)
        x2 = Node(id=1, op_type=OperationType.INPUT, inputs=[], output_metadata=sample_metadata, node_type=NodeType.INPUT)
        add = Node(id=2, op_type=OperationType.ADD, inputs=[x1, x2], output_metadata=sample_metadata)
        relu = Node(id=3, op_type=OperationType.RELU, inputs=[add], output_metadata=sample_metadata)
        tanh = Node(id=4, op_type=OperationType.TANH, inputs=[add], output_metadata=sample_metadata)
        mul = Node(id=5, op_type=OperationType.MUL, inputs=[relu, tanh], output_metadata=sample_metadata)

        graph = Graph([x1, x2, add, relu, tanh, mul])
        graph.mark_outputs([mul])

        # Validate structure
        graph.validate()
        assert graph.num_nodes == 6
        assert graph.num_inputs == 2
        assert graph.num_outputs == 1

        # Check topological order
        order = graph.topological_order()
        assert len(order) == 6

        # Inputs should come first
        assert x1 in order[:2]
        assert x2 in order[:2]

        # Add should come before relu and tanh
        assert order.index(add) < order.index(relu)
        assert order.index(add) < order.index(tanh)

        # Mul should come last
        assert order[-1] == mul

    def test_empty_graph_iteration(self):
        """Test iterating over empty graph."""
        graph = Graph([])
        nodes = list(graph)
        assert len(nodes) == 0
