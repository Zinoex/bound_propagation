from __future__ import annotations

import pytest
import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.ir import (
    AbstractValueType,
    Graph,
    Node,
    NodeType,
    OperationType,
    TensorMetadata,
)
from bound_propagation.propagation import IBPPropagator
from bound_propagation.propagation.ibp import (
    ForwardIBPStrategyRegistry,
    IBPAdd,
    IBPAddWithConstant,
)
from bound_propagation.regions import HyperRectangle


def _meta(shape: tuple[int, ...]) -> TensorMetadata:
    return TensorMetadata(shape=shape, dtype="torch.float32")

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
        graph.annotate_input_kinds()

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
        graph.annotate_input_kinds()

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
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([4.0, -5.0]))
        assert torch.allclose(out.upper, torch.tensor([6.0, -4.0]))

    def test_number_constants_are_rejected(self) -> None:
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
        graph.annotate_input_kinds()

        propagator = IBPPropagator(graph)
        with pytest.raises(TypeError, match="non-tensor value type"):
            propagator.propagate(
                [
                    HyperRectangle(
                        lower=torch.tensor([0.0, 1.0]),
                        upper=torch.tensor([1.0, 2.0]),
                    )
                ]
            )


class TestIBPRegistryStrictness:
    def test_lookup_requires_exact_input_signature(self) -> None:
        registry = ForwardIBPStrategyRegistry()
        strategy = IBPAdd()
        registry.register(
            OperationType.ADD,
            strategy,
            abstract_signature=(
                AbstractValueType.ABSTRACT,
                AbstractValueType.ABSTRACT,
            ),
        )

        assert (
            registry.get_strategy(
                OperationType.ADD,
                (AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT),
            )
            is strategy
        )

        with pytest.raises(ValueError, match="signature"):
            registry.get_strategy(
                OperationType.ADD,
                (AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
            )

    def test_default_registry_does_not_register_folded_constant_signatures(self) -> None:
        registry = ForwardIBPStrategyRegistry.default_registry()

        with pytest.raises(ValueError, match="signature"):
            registry.get_strategy(OperationType.RELU, (AbstractValueType.CONSTANT,))

        with pytest.raises(ValueError, match="signature"):
            registry.get_strategy(
                OperationType.ADD,
                (AbstractValueType.CONSTANT, AbstractValueType.CONSTANT),
            )
