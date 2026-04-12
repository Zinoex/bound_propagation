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
    IBPConstantMatmul,
    IBPMatmul,
    IBPMatmulConstant,
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

    def test_dispatch_uses_matmul_input_kind_signature(self) -> None:
        x = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((2,)),
            node_type=NodeType.INPUT,
        )
        y = Node(
            id=1,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((2, 2)),
            node_type=NodeType.INPUT,
        )
        c = Node(
            id=2,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=_meta((2, 2)),
            attributes={"value": torch.tensor([[1.0, -1.0], [0.5, 2.0]])},
            node_type=NodeType.CONSTANT,
        )
        abstract_matmul = Node(
            id=3,
            op_type=OperationType.MATMUL,
            inputs=[x, y],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )
        mixed_matmul = Node(
            id=4,
            op_type=OperationType.MATMUL,
            inputs=[x, c],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )
        constant_left_matmul = Node(
            id=5,
            op_type=OperationType.MATMUL,
            inputs=[c, x],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )

        graph = Graph([x, y, c, abstract_matmul, mixed_matmul, constant_left_matmul])
        graph.mark_outputs([abstract_matmul, mixed_matmul, constant_left_matmul])
        graph.annotate_input_kinds()

        propagator = IBPPropagator(graph)

        assert isinstance(propagator._bound_strategies[abstract_matmul.id], IBPMatmul)
        assert isinstance(propagator._bound_strategies[mixed_matmul.id], IBPMatmulConstant)
        assert isinstance(propagator._bound_strategies[constant_left_matmul.id], IBPConstantMatmul)

    def test_interval_constant_matmul_handles_mixed_sign_weights(self) -> None:
        x = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((2,)),
            node_type=NodeType.INPUT,
        )
        weight = Node(
            id=1,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=_meta((2, 3)),
            attributes={"value": torch.tensor([[1.0, -0.5, 2.0], [0.5, 1.0, -1.0]])},
            node_type=NodeType.CONSTANT,
        )
        matmul = Node(
            id=2,
            op_type=OperationType.MATMUL,
            inputs=[x, weight],
            output_metadata=_meta((3,)),
            node_type=NodeType.OPERATION,
        )

        graph = Graph([x, weight, matmul])
        graph.mark_outputs([matmul])
        graph.annotate_input_kinds()

        propagator = IBPPropagator(graph)
        outputs = propagator.propagate(
            [
                HyperRectangle(
                    lower=torch.tensor([0.0, 0.0]),
                    upper=torch.tensor([1.0, 1.0]),
                )
            ]
        )

        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([0.0, -0.5, -1.0]))
        assert torch.allclose(out.upper, torch.tensor([1.5, 1.0, 2.0]))

    def test_constant_interval_matmul_handles_mixed_sign_weights(self) -> None:
        weight = Node(
            id=0,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=_meta((2, 2)),
            attributes={"value": torch.tensor([[1.0, -2.0], [-1.0, 0.5]])},
            node_type=NodeType.CONSTANT,
        )
        x = Node(
            id=1,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((2,)),
            node_type=NodeType.INPUT,
        )
        matmul = Node(
            id=2,
            op_type=OperationType.MATMUL,
            inputs=[weight, x],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )

        graph = Graph([weight, x, matmul])
        graph.mark_outputs([matmul])
        graph.annotate_input_kinds()

        propagator = IBPPropagator(graph)
        outputs = propagator.propagate(
            [
                HyperRectangle(
                    lower=torch.tensor([0.0, 1.0]),
                    upper=torch.tensor([2.0, 3.0]),
                )
            ]
        )

        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([-6.0, -1.5]))
        assert torch.allclose(out.upper, torch.tensor([0.0, 1.5]))

    def test_interval_interval_matmul_supports_broadcasted_batches(self) -> None:
        x = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((1, 2, 1)),
            node_type=NodeType.INPUT,
        )
        y = Node(
            id=1,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((3, 1, 2)),
            node_type=NodeType.INPUT,
        )
        matmul = Node(
            id=2,
            op_type=OperationType.MATMUL,
            inputs=[x, y],
            output_metadata=_meta((3, 2, 2)),
            node_type=NodeType.OPERATION,
        )

        graph = Graph([x, y, matmul])
        graph.mark_outputs([matmul])
        graph.annotate_input_kinds()

        propagator = IBPPropagator(graph)
        outputs = propagator.propagate(
            [
                HyperRectangle(
                    lower=torch.tensor([[[-1.0], [2.0]]]),
                    upper=torch.tensor([[[1.0], [3.0]]]),
                ),
                HyperRectangle(
                    lower=torch.tensor([
                        [[2.0, -2.0]],
                        [[-3.0, 1.0]],
                        [[0.5, -1.0]],
                    ]),
                    upper=torch.tensor([
                        [[4.0, -1.0]],
                        [[-1.0, 2.0]],
                        [[1.5, 2.0]],
                    ]),
                ),
            ]
        )

        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        expected_lower = torch.tensor(
            [
                [[-4.0, -2.0], [4.0, -6.0]],
                [[-3.0, -2.0], [-9.0, 2.0]],
                [[-1.5, -2.0], [1.0, -3.0]],
            ]
        )
        expected_upper = torch.tensor(
            [
                [[4.0, 2.0], [12.0, -2.0]],
                [[3.0, 2.0], [-2.0, 6.0]],
                [[1.5, 2.0], [4.5, 6.0]],
            ]
        )

        assert out.lower.shape == (3, 2, 2)
        assert out.upper.shape == (3, 2, 2)
        assert torch.allclose(out.lower, expected_lower)
        assert torch.allclose(out.upper, expected_upper)


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
