from __future__ import annotations

import pytest
import torch

from bound_propagation.ir import Graph, Node, NodeType, OperationType, TensorMetadata
from bound_propagation.propagation.ibp import (
    ForwardIBPStrategyRegistry,
    IBPAdd,
    IBPAddWithConstant,
)
from bound_propagation.propagation.methods import IBPPropagator
from bound_propagation.regions import HyperRectangle


def _meta(shape: tuple[int, ...]) -> TensorMetadata:
    return TensorMetadata(shape=shape, dtype="float32")


class TestNodeValueAccess:
    def test_constant_value_returns_tensor(self) -> None:
        value = torch.tensor([1.0, 2.0])
        node = Node(
            id=0,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=_meta((2,)),
            attributes={"value": value},
            node_type=NodeType.CONSTANT,
        )

        assert torch.allclose(node.value, value)

    def test_non_constant_value_raises(self) -> None:
        node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((2,)),
            node_type=NodeType.INPUT,
        )

        with pytest.raises(ValueError, match="does not hold a constant value"):
            _ = node.value


class TestInputKindTraversal:
    def test_graph_annotation_marks_signatures(self) -> None:
        x = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((2,)),
            node_type=NodeType.INPUT,
        )
        c1 = Node(
            id=1,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=_meta((2,)),
            attributes={"value": torch.tensor([2.0, 3.0])},
            node_type=NodeType.CONSTANT,
        )
        c2 = Node(
            id=2,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=_meta((2,)),
            attributes={"value": torch.tensor([4.0, 5.0])},
            node_type=NodeType.CONSTANT,
        )
        const_add = Node(
            id=3,
            op_type=OperationType.ADD,
            inputs=[c1, c2],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )
        mixed_add = Node(
            id=4,
            op_type=OperationType.ADD,
            inputs=[x, c1],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )
        out = Node(
            id=5,
            op_type=OperationType.MUL,
            inputs=[mixed_add, const_add],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )

        graph = Graph([x, c1, c2, const_add, mixed_add, out])
        graph.mark_outputs([out])

        signatures = graph.annotate_input_kinds()

        assert signatures[const_add.id] == ("constant", "constant")
        assert signatures[mixed_add.id] == ("abstract", "constant")
        assert signatures[out.id] == ("abstract", "constant")
        assert const_add.attributes["output_kind"] == "constant"
        assert mixed_add.attributes["output_kind"] == "abstract"


class TestIBPSignatureDispatch:
    def test_dispatch_uses_input_kind_signature(self) -> None:
        x = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((2,)),
            node_type=NodeType.INPUT,
        )
        c = Node(
            id=1,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=_meta((2,)),
            attributes={"value": torch.tensor([2.0, 3.0])},
            node_type=NodeType.CONSTANT,
        )
        mixed_add = Node(
            id=2,
            op_type=OperationType.ADD,
            inputs=[x, c],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )
        abstract_add = Node(
            id=3,
            op_type=OperationType.ADD,
            inputs=[mixed_add, x],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )

        graph = Graph([x, c, mixed_add, abstract_add])
        graph.mark_outputs([abstract_add])

        propagator = IBPPropagator(graph)

        mixed_strategy = propagator._bound_strategies[mixed_add.id]
        abstract_strategy = propagator._bound_strategies[abstract_add.id]

        assert isinstance(mixed_strategy, IBPAddWithConstant)
        assert isinstance(abstract_strategy, IBPAdd)

    def test_abstract_constant_add_and_mul_are_computed_correctly(self) -> None:
        x = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((2,)),
            node_type=NodeType.INPUT,
        )
        c_add = Node(
            id=1,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=_meta((2,)),
            attributes={"value": torch.tensor([2.0, 3.0])},
            node_type=NodeType.CONSTANT,
        )
        c_mul = Node(
            id=2,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=_meta((2,)),
            attributes={"value": torch.tensor([2.0, -1.0])},
            node_type=NodeType.CONSTANT,
        )
        add_node = Node(
            id=3,
            op_type=OperationType.ADD,
            inputs=[x, c_add],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )
        mul_node = Node(
            id=4,
            op_type=OperationType.MUL,
            inputs=[add_node, c_mul],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )

        graph = Graph([x, c_add, c_mul, add_node, mul_node])
        graph.mark_outputs([mul_node])

        propagator = IBPPropagator(graph)
        outputs = propagator.propagate(
            [
                HyperRectangle(
                    lower=torch.tensor([0.0, 1.0]),
                    upper=torch.tensor([1.0, 2.0]),
                )
            ]
        )

        # x in [0,1]x[1,2], add [2,3] -> [2,3]x[4,5]
        # multiply by [2,-1] -> [4,6]x[-5,-4]
        assert len(outputs) == 1
        out = outputs[0]
        assert torch.allclose(out.lower, torch.tensor([4.0, -5.0]))
        assert torch.allclose(out.upper, torch.tensor([6.0, -4.0]))

    def test_number_constants_are_supported(self) -> None:
        x = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((2,)),
            node_type=NodeType.INPUT,
        )
        c_add = Node(
            id=1,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=_meta((2,)),
            attributes={"value": 2.0},
            node_type=NodeType.CONSTANT,
        )
        c_mul = Node(
            id=2,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=_meta((2,)),
            attributes={"value": -1.0},
            node_type=NodeType.CONSTANT,
        )
        add_node = Node(
            id=3,
            op_type=OperationType.ADD,
            inputs=[x, c_add],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )
        mul_node = Node(
            id=4,
            op_type=OperationType.MUL,
            inputs=[add_node, c_mul],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )

        graph = Graph([x, c_add, c_mul, add_node, mul_node])
        graph.mark_outputs([mul_node])

        propagator = IBPPropagator(graph)
        outputs = propagator.propagate(
            [
                HyperRectangle(
                    lower=torch.tensor([0.0, 1.0]),
                    upper=torch.tensor([1.0, 2.0]),
                )
            ]
        )

        # (x + 2) * (-1) with x in [0,1]x[1,2] -> [-3,-2]x[-4,-3]
        out = outputs[0]
        assert torch.allclose(out.lower, torch.tensor([-3.0, -4.0]))
        assert torch.allclose(out.upper, torch.tensor([-2.0, -3.0]))


class TestIBPRegistryStrictness:
    def test_lookup_requires_exact_input_signature(self) -> None:
        registry = ForwardIBPStrategyRegistry()
        strategy = IBPAdd()
        registry.register(OperationType.ADD, strategy, signature=("abstract", "abstract"))

        assert registry.get_strategy(OperationType.ADD, ("abstract", "abstract")) is strategy

        with pytest.raises(ValueError, match="with input_kinds"):
            registry.get_strategy(OperationType.ADD, ("abstract", "constant"))

    def test_default_registry_does_not_register_folded_constant_signatures(self) -> None:
        registry = ForwardIBPStrategyRegistry.default_registry()

        with pytest.raises(ValueError, match="with input_kinds"):
            registry.get_strategy(OperationType.RELU, ("constant",))

        with pytest.raises(ValueError, match="with input_kinds"):
            registry.get_strategy(OperationType.ADD, ("constant", "constant"))
