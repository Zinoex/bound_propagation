from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.ir import (
    AbstractValueType,
    Graph,
    Node,
    NodeType,
    OperationType,
    TensorMetadata,
)
from bound_propagation.propagation.forward_lbp import (
    ForwardLBPAddStrategy,
    ForwardLBPAddWithConstant,
    ForwardLBPMatmulConstant,
    ForwardLBPMatmulStrategy,
    ForwardLBPMulStrategy,
    ForwardLBPMulWithConstant,
)
from bound_propagation.propagation.methods import ForwardLBPPropagator
from bound_propagation.regions import HyperRectangle


def _meta(shape: tuple[int, ...]) -> TensorMetadata:
    """Create tensor metadata."""
    return TensorMetadata(shape=shape, dtype="torch.float32")


class TestForwardLBPSignatureDispatch:
    """Test that Forward LBP propagator correctly dispatches based on input signatures."""

    def test_dispatch_uses_input_kind_signature(self) -> None:
        """Test that strategy selection depends on abstract vs constant inputs."""
        # Graph: x (abstract) + c (constant) -> mixed_add
        #        mixed_add + x -> abstract_add
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

        propagator = ForwardLBPPropagator(graph)

        # Check strategy types
        mixed_strategy = propagator._bound_strategies[mixed_add.id]
        abstract_strategy = propagator._bound_strategies[abstract_add.id]

        # mixed_add has signature (ABSTRACT, CONSTANT) -> ForwardLBPAddWithConstant
        assert isinstance(mixed_strategy, ForwardLBPAddWithConstant)
        # abstract_add has signature (ABSTRACT, ABSTRACT) -> ForwardLBPAddStrategy
        assert isinstance(abstract_strategy, ForwardLBPAddStrategy)

    def test_constant_abstract_vs_abstract_constant_add(self) -> None:
        """Test that both orderings of (abstract, constant) use the same strategy."""
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
            attributes={"value": torch.tensor([5.0, 10.0])},
            node_type=NodeType.CONSTANT,
        )
        add1 = Node(
            id=2,
            op_type=OperationType.ADD,
            inputs=[x, c],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )
        add2 = Node(
            id=3,
            op_type=OperationType.ADD,
            inputs=[c, x],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )

        graph = Graph([x, c, add1, add2])
        graph.mark_outputs([add1, add2])
        graph.annotate_input_kinds()

        propagator = ForwardLBPPropagator(graph)

        # Both should use ForwardLBPAddWithConstant (addition is commutative)
        assert isinstance(propagator._bound_strategies[add1.id], ForwardLBPAddWithConstant)
        assert isinstance(propagator._bound_strategies[add2.id], ForwardLBPAddWithConstant)

    def test_mul_abstract_constant_dispatch(self) -> None:
        """Test multiplication strategy dispatch for abstract and constant inputs."""
        x = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((3,)),
            node_type=NodeType.INPUT,
        )
        c = Node(
            id=1,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=_meta((3,)),
            attributes={"value": torch.tensor([2.0, -1.0, 0.5])},
            node_type=NodeType.CONSTANT,
        )
        y = Node(
            id=2,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((3,)),
            node_type=NodeType.INPUT,
        )

        # x * c (abstract * constant)
        mul1 = Node(
            id=3,
            op_type=OperationType.MUL,
            inputs=[x, c],
            output_metadata=_meta((3,)),
            node_type=NodeType.OPERATION,
        )
        # x * y (abstract * abstract)
        mul2 = Node(
            id=4,
            op_type=OperationType.MUL,
            inputs=[x, y],
            output_metadata=_meta((3,)),
            node_type=NodeType.OPERATION,
        )

        graph = Graph([x, c, y, mul1, mul2])
        graph.mark_outputs([mul1, mul2])
        graph.annotate_input_kinds()

        propagator = ForwardLBPPropagator(graph)

        # mul1: (ABSTRACT, CONSTANT) -> ForwardLBPMulWithConstant
        assert isinstance(propagator._bound_strategies[mul1.id], ForwardLBPMulWithConstant)
        # mul2: (ABSTRACT, ABSTRACT) -> ForwardLBPMulStrategy
        assert isinstance(propagator._bound_strategies[mul2.id], ForwardLBPMulStrategy)

    def test_matmul_signature_dispatch(self) -> None:
        """Test matmul strategy dispatch for different signatures."""
        x = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((2, 3)),
            node_type=NodeType.INPUT,
        )
        w = Node(
            id=1,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=_meta((3, 4)),
            attributes={"value": torch.randn(3, 4)},
            node_type=NodeType.CONSTANT,
        )
        y = Node(
            id=2,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=_meta((2, 3)),
            node_type=NodeType.INPUT,
        )

        # x @ w (abstract @ constant)
        matmul1 = Node(
            id=3,
            op_type=OperationType.MATMUL,
            inputs=[x, w],
            output_metadata=_meta((2, 4)),
            node_type=NodeType.OPERATION,
        )
        # x @ y (abstract @ abstract)
        matmul2 = Node(
            id=4,
            op_type=OperationType.MATMUL,
            inputs=[x, y],
            output_metadata=_meta((2, 3)),  # Broadcasting might differ
            node_type=NodeType.OPERATION,
        )

        graph = Graph([x, w, y, matmul1, matmul2])
        graph.mark_outputs([matmul1, matmul2])
        graph.annotate_input_kinds()

        propagator = ForwardLBPPropagator(graph)

        # matmul1: (ABSTRACT, CONSTANT) -> ForwardLBPMatmulConstant
        assert isinstance(propagator._bound_strategies[matmul1.id], ForwardLBPMatmulConstant)
        # matmul2: (ABSTRACT, ABSTRACT) -> ForwardLBPMatmulStrategy
        assert isinstance(propagator._bound_strategies[matmul2.id], ForwardLBPMatmulStrategy)

    def test_computation_with_mixed_signatures(self) -> None:
        """Test end-to-end computation with mixed abstract/constant signatures."""
        # Graph: x + 2 -> add1, add1 * 3 -> mul1
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
            attributes={"value": torch.tensor([2.0, 2.0])},
            node_type=NodeType.CONSTANT,
        )
        c2 = Node(
            id=2,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=_meta((2,)),
            attributes={"value": torch.tensor([3.0, 3.0])},
            node_type=NodeType.CONSTANT,
        )

        add1 = Node(
            id=3,
            op_type=OperationType.ADD,
            inputs=[x, c1],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )
        mul1 = Node(
            id=4,
            op_type=OperationType.MUL,
            inputs=[add1, c2],
            output_metadata=_meta((2,)),
            node_type=NodeType.OPERATION,
        )

        graph = Graph([x, c1, c2, add1, mul1])
        graph.mark_outputs([mul1])
        graph.annotate_input_kinds()

        # Input region: x ∈ [1, 2]
        input_region = HyperRectangle(lower=torch.tensor([1.0, 1.0]), upper=torch.tensor([2.0, 2.0]))

        propagator = ForwardLBPPropagator(graph)
        outputs = propagator.propagate([input_region])

        # x ∈ [1, 2], (x + 2) ∈ [3, 4], 3(x + 2) ∈ [9, 12]
        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, LinearBounds)

        lower, upper = result.concretize()
        assert torch.allclose(lower, torch.tensor([9.0, 9.0]))
        assert torch.allclose(upper, torch.tensor([12.0, 12.0]))
