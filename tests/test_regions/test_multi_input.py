"""
Tests for multi-input and multi-output support in propagators.
"""

import pytest
import torch

from bound_propagation.ir import Graph, Node, NodeType, OperationType, TensorMetadata
from bound_propagation.propagation import (
    IBPPropagator,
)
from bound_propagation.regions import HyperRectangle, MultiInputRegion


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
