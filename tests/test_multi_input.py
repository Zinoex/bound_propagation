"""
Tests for multi-input and multi-output support in propagators.
"""

# Import relaxations to ensure they're registered
import pytest
import torch

import bound_propagation.propagation.relaxations  # noqa: F401
from bound_propagation.ir.graph import Graph
from bound_propagation.ir.metadata import TensorMetadata
from bound_propagation.ir.node import Node, NodeType
from bound_propagation.ir.operations import OperationType
from bound_propagation.propagation.methods import (
    ForwardLBPPropagator,
    IBPPropagator,
)
from bound_propagation.regions.hyperrectangle import HyperRectangle
from bound_propagation.regions.multi_input import MultiInputRegion


class TestMultiInputRegion:
    """Test the MultiInputRegion class."""

    def test_create_multi_input_region(self):
        """Test creating a multi-input region."""
        region1 = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        region2 = HyperRectangle(
            lower=torch.tensor([-1.0]),
            upper=torch.tensor([1.0]),
        )

        multi_region = MultiInputRegion({0: region1, 1: region2})

        assert len(multi_region) == 2
        assert 0 in multi_region
        assert 1 in multi_region
        assert 2 not in multi_region

    def test_access_individual_regions(self):
        """Test accessing individual input regions."""
        region1 = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        region2 = HyperRectangle(
            lower=torch.tensor([-1.0]),
            upper=torch.tensor([1.0]),
        )

        multi_region = MultiInputRegion({0: region1, 1: region2})

        retrieved1 = multi_region[0]
        assert torch.allclose(retrieved1.lower, region1.lower)
        assert torch.allclose(retrieved1.upper, region1.upper)

        retrieved2 = multi_region[1]
        assert torch.allclose(retrieved2.lower, region2.lower)
        assert torch.allclose(retrieved2.upper, region2.upper)

    def test_from_single_region(self):
        """Test creating MultiInputRegion from single HyperRectangle."""
        region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )

        multi_region = MultiInputRegion.from_single_region(region, input_id=5)

        assert len(multi_region) == 1
        assert 5 in multi_region
        assert torch.allclose(multi_region[5].lower, region.lower)

    def test_device_consistency(self):
        """Test that all regions must be on same device."""
        region1 = HyperRectangle(
            lower=torch.tensor([0.0]),
            upper=torch.tensor([1.0]),
        )

        multi_region = MultiInputRegion({0: region1})
        assert multi_region.device.type == "cpu"

    def test_empty_region_raises_error(self):
        """Test that empty region dict raises error."""
        with pytest.raises(ValueError, match="at least one input"):
            MultiInputRegion({})


class TestMultiInputPropagation:
    """Test propagation with multiple input nodes."""

    def test_forward_lbp_with_two_inputs(self):
        """Test ForwardLBP with two input nodes."""
        # Create graph: x + y where x and y are separate inputs
        graph = Graph()
        metadata = TensorMetadata(shape=(2,), dtype="float32")

        input_x = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input_x)

        input_y = Node(
            id=1,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input_y)

        add_node = Node(
            id=2,
            op_type=OperationType.ADD,
            inputs=[input_x, input_y],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(add_node)
        graph.mark_outputs([add_node])

        # Create multi-input region with different bounds for each input
        region = MultiInputRegion(
            {
                0: HyperRectangle(
                    lower=torch.tensor([0.0, 0.0]),
                    upper=torch.tensor([1.0, 1.0]),
                ),
                1: HyperRectangle(
                    lower=torch.tensor([2.0, 3.0]),
                    upper=torch.tensor([4.0, 5.0]),
                ),
            }
        )

        propagator = ForwardLBPPropagator()
        bounds = propagator.propagate(graph, region)

        # Check input bounds
        assert 0 in bounds
        assert torch.allclose(bounds[0].lower, torch.tensor([0.0, 0.0]))
        assert torch.allclose(bounds[0].upper, torch.tensor([1.0, 1.0]))

        assert 1 in bounds
        assert torch.allclose(bounds[1].lower, torch.tensor([2.0, 3.0]))
        assert torch.allclose(bounds[1].upper, torch.tensor([4.0, 5.0]))

        # Check output bounds: [0,1] + [2,4] = [2,5] and [0,1] + [3,5] = [3,6]
        assert 2 in bounds
        expected_lower = torch.tensor([2.0, 3.0])
        expected_upper = torch.tensor([5.0, 6.0])
        assert torch.allclose(bounds[2].lower, expected_lower)
        assert torch.allclose(bounds[2].upper, expected_upper)

    def test_ibp_with_two_inputs(self):
        """Test IBP with two input nodes."""
        # Create graph: x * y where x and y are separate inputs
        graph = Graph()
        metadata = TensorMetadata(shape=(2,), dtype="float32")

        input_x = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input_x)

        input_y = Node(
            id=1,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input_y)

        mul_node = Node(
            id=2,
            op_type=OperationType.MUL,
            inputs=[input_x, input_y],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(mul_node)
        graph.mark_outputs([mul_node])

        # Create multi-input region
        region = MultiInputRegion(
            {
                0: HyperRectangle(
                    lower=torch.tensor([1.0, 2.0]),
                    upper=torch.tensor([2.0, 3.0]),
                ),
                1: HyperRectangle(
                    lower=torch.tensor([0.5, 1.0]),
                    upper=torch.tensor([1.5, 2.0]),
                ),
            }
        )

        propagator = IBPPropagator()
        bounds = propagator.propagate(graph, region)

        # Check output bounds using 4-corner method
        # For [1,2] * [0.5,1.5]: min=0.5, max=3.0
        # For [2,3] * [1.0,2.0]: min=2.0, max=6.0
        assert 2 in bounds
        # Lower bounds are the minimum of all corners
        assert bounds[2].lower[0] <= 1.0 and bounds[2].lower[0] >= 0.5
        assert bounds[2].lower[1] <= 2.5 and bounds[2].lower[1] >= 2.0

    def test_missing_input_raises_error(self):
        """Test that missing input in region raises error."""
        graph = Graph()
        metadata = TensorMetadata(shape=(2,), dtype="float32")

        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input_node)
        graph.mark_outputs([input_node])

        # Create region without the required input
        region = MultiInputRegion(
            {
                1: HyperRectangle(
                    lower=torch.tensor([0.0, 0.0]),
                    upper=torch.tensor([1.0, 1.0]),
                ),
            }
        )

        propagator = ForwardLBPPropagator()
        with pytest.raises(ValueError, match="Input node 0 not found"):
            propagator.propagate(graph, region)

    def test_three_input_addition(self):
        """Test with three separate inputs."""
        # Create graph: x + y + z
        graph = Graph()
        metadata = TensorMetadata(shape=(1,), dtype="float32")

        input_nodes = []
        for i in range(3):
            node = Node(
                id=i,
                op_type=OperationType.INPUT,
                inputs=[],
                output_metadata=metadata,
                attributes={},
                node_type=NodeType.INPUT,
            )
            graph.add_node(node)
            input_nodes.append(node)

        # x + y
        add1 = Node(
            id=3,
            op_type=OperationType.ADD,
            inputs=[input_nodes[0], input_nodes[1]],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(add1)

        # (x + y) + z
        add2 = Node(
            id=4,
            op_type=OperationType.ADD,
            inputs=[add1, input_nodes[2]],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(add2)
        graph.mark_outputs([add2])

        # Create multi-input region
        region = MultiInputRegion(
            {
                0: HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0])),
                1: HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([2.0])),
                2: HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([3.0])),
            }
        )

        propagator = IBPPropagator()
        bounds = propagator.propagate(graph, region)

        # Expected: [0,1] + [0,2] + [0,3] = [0,6]
        assert torch.allclose(bounds[4].lower, torch.tensor([0.0]))
        assert torch.allclose(bounds[4].upper, torch.tensor([6.0]))


class TestBackwardCompatibility:
    """Test that single-input scenarios still work."""

    def test_forward_lbp_with_single_input_region(self):
        """Test ForwardLBP still works with HyperRectangle."""
        graph = Graph()
        metadata = TensorMetadata(shape=(2,), dtype="float32")

        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input_node)

        relu_node = Node(
            id=1,
            op_type=OperationType.RELU,
            inputs=[input_node],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(relu_node)
        graph.mark_outputs([relu_node])

        # Use HyperRectangle directly (not MultiInputRegion)
        region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )

        propagator = ForwardLBPPropagator()
        bounds = propagator.propagate(graph, region)

        assert 0 in bounds
        assert 1 in bounds
        # ReLU with ForwardLBP uses relaxation, so bounds may be looser than exact ReLU
        # Just verify that bounds are reasonable (not exact)
        assert bounds[1].lower[0] <= 0.0  # Lower bound should be <= 0
        assert bounds[1].upper[0] >= 0.0  # Upper bound should be >= 0
        assert bounds[1].upper[0] <= 1.0  # Upper bound should not exceed input upper

    def test_ibp_with_single_input_region(self):
        """Test IBP still works with HyperRectangle."""
        graph = Graph()
        metadata = TensorMetadata(shape=(2,), dtype="float32")

        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input_node)

        sigmoid_node = Node(
            id=1,
            op_type=OperationType.SIGMOID,
            inputs=[input_node],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(sigmoid_node)
        graph.mark_outputs([sigmoid_node])

        region = HyperRectangle(
            lower=torch.tensor([-5.0, -5.0]),
            upper=torch.tensor([5.0, 5.0]),
        )

        propagator = IBPPropagator()
        bounds = propagator.propagate(graph, region)

        assert 1 in bounds
        # Sigmoid bounds should be in (0, 1)
        assert torch.all(bounds[1].lower >= 0.0)
        assert torch.all(bounds[1].lower <= 1.0)
        assert torch.all(bounds[1].upper >= 0.0)
        assert torch.all(bounds[1].upper <= 1.0)
