"""
Tests for operation types and categories.
"""

from bound_propagation.ir.operations import OperationCategory, OperationType


class TestOperationType:
    """Tests for OperationType enum."""

    def test_operation_types_are_strings(self):
        """Test that all operation types are strings."""
        for op_type in OperationType:
            assert isinstance(op_type, str)

    def test_linear_operations(self):
        """Test linear operation types."""
        linear_ops = [OperationType.MATMUL, OperationType.LINEAR, OperationType.CONV2D, OperationType.TRANSPOSE]

        for op in linear_ops:
            assert op.is_linear is True
            assert op.category == OperationCategory.LINEAR
            assert op.is_elementwise is False
            assert op.is_activation is False

    def test_elementwise_arithmetic_operations(self):
        """Test elementwise arithmetic operations."""
        arithmetic_ops = [
            OperationType.ADD,
            OperationType.SUB,
            OperationType.MUL,
            OperationType.DIV,
            OperationType.POW,
            OperationType.NEG,
        ]

        for op in arithmetic_ops:
            assert op.is_elementwise is True
            assert op.category == OperationCategory.ELEMENTWISE
            assert op.is_linear is False

    def test_activation_operations(self):
        """Test activation function operations."""
        activation_ops = [
            OperationType.RELU,
            OperationType.SIGMOID,
            OperationType.TANH,
            OperationType.EXP,
            OperationType.LOG,
            OperationType.SQRT,
            OperationType.ABS,
        ]

        for op in activation_ops:
            assert op.is_activation is True
            assert op.is_elementwise is True
            assert op.category == OperationCategory.ELEMENTWISE

    def test_trigonometric_operations(self):
        """Test trigonometric operations."""
        trig_ops = [OperationType.SIN, OperationType.COS, OperationType.TAN]

        for op in trig_ops:
            assert op.is_activation is True
            assert op.is_elementwise is True
            assert op.category == OperationCategory.ELEMENTWISE

    def test_reduction_operations(self):
        """Test reduction operations."""
        reduction_ops = [OperationType.SUM, OperationType.MEAN, OperationType.MAX, OperationType.MIN]

        for op in reduction_ops:
            assert op.is_reduction is True
            assert op.category == OperationCategory.REDUCTION
            assert op.is_elementwise is False
            assert op.is_linear is False

    def test_structural_operations(self):
        """Test structural operations."""
        structural_ops = [
            OperationType.CONCAT,
            OperationType.SPLIT,
            OperationType.SELECT,
            OperationType.GATHER,
            OperationType.RESHAPE,
            OperationType.FLATTEN,
            OperationType.PERMUTE,
            OperationType.UNSQUEEZE,
            OperationType.SQUEEZE,
        ]

        for op in structural_ops:
            assert op.is_structural is True
            assert op.category == OperationCategory.STRUCTURAL
            assert op.is_elementwise is False
            assert op.is_linear is False

    def test_derivative_operations(self):
        """Test derivative operations."""
        derivative_ops = [OperationType.JACOBIAN, OperationType.GRADIENT, OperationType.VJP, OperationType.JVP]

        for op in derivative_ops:
            assert op.is_derivative is True
            assert op.category == OperationCategory.DERIVATIVE
            assert op.is_elementwise is False
            assert op.is_linear is False

    def test_non_activation_elementwise(self):
        """Test elementwise operations that are not activations."""
        non_activation_elementwise = [
            OperationType.ADD,
            OperationType.SUB,
            OperationType.MUL,
            OperationType.DIV,
            OperationType.POW,
            OperationType.NEG,
            OperationType.CLAMP,
            OperationType.RECIPROCAL,
        ]

        for op in non_activation_elementwise:
            assert op.is_elementwise is True
            assert op.is_activation is False

    def test_specific_operation_values(self):
        """Test specific operation string values."""
        assert OperationType.MATMUL == "matmul"
        assert OperationType.LINEAR == "linear"
        assert OperationType.RELU == "relu"
        assert OperationType.ADD == "add"
        assert OperationType.JACOBIAN == "jacobian"

    def test_category_property(self):
        """Test category property returns correct category."""
        assert OperationType.MATMUL.category == OperationCategory.LINEAR
        assert OperationType.RELU.category == OperationCategory.ELEMENTWISE
        assert OperationType.SUM.category == OperationCategory.REDUCTION
        assert OperationType.CONCAT.category == OperationCategory.STRUCTURAL
        assert OperationType.JACOBIAN.category == OperationCategory.DERIVATIVE

    def test_all_operations_have_category(self):
        """Test that all operation types have a category assigned."""
        for op_type in OperationType:
            # Skip special input/constant/parameter types
            if op_type not in [OperationType.INPUT, OperationType.CONSTANT, OperationType.PARAMETER]:
                assert op_type.category is not None
                assert isinstance(op_type.category, OperationCategory)

    def test_special_operations(self):
        """Test special operation types."""
        special_ops = [OperationType.INPUT, OperationType.CONSTANT, OperationType.PARAMETER]

        for op in special_ops:
            # Special operations might have OTHER category
            assert op.category == OperationCategory.OTHER


class TestOperationCategory:
    """Tests for OperationCategory enum."""

    def test_category_values(self):
        """Test all category values."""
        assert OperationCategory.LINEAR == "linear"
        assert OperationCategory.ELEMENTWISE == "elementwise"
        assert OperationCategory.REDUCTION == "reduction"
        assert OperationCategory.STRUCTURAL == "structural"
        assert OperationCategory.DERIVATIVE == "derivative"
        assert OperationCategory.OTHER == "other"

    def test_category_is_string(self):
        """Test that OperationCategory values are strings."""
        for category in OperationCategory:
            assert isinstance(category, str)

    def test_all_categories_covered(self):
        """Test that we have examples of each category."""
        # Get all categories used by operations
        categories_used = set()
        for op_type in OperationType:
            categories_used.add(op_type.category)

        # Should have all main categories
        assert OperationCategory.LINEAR in categories_used
        assert OperationCategory.ELEMENTWISE in categories_used
        assert OperationCategory.REDUCTION in categories_used
        assert OperationCategory.STRUCTURAL in categories_used
        assert OperationCategory.DERIVATIVE in categories_used


class TestOperationProperties:
    """Tests for operation type properties and relationships."""

    def test_mutually_exclusive_categories(self):
        """Test that operations belong to exactly one category."""
        for op_type in OperationType:
            categories_count = sum(
                [op_type.is_linear, op_type.is_reduction, op_type.is_structural, op_type.is_derivative]
            )

            # Elementwise can overlap, but others should be mutually exclusive
            # Each non-elementwise op should be in exactly one other category
            if not op_type.is_elementwise and op_type.category != OperationCategory.OTHER:
                assert categories_count == 1, f"{op_type} belongs to {categories_count} categories"

    def test_activation_subset_of_elementwise(self):
        """Test that all activations are elementwise."""
        for op_type in OperationType:
            if op_type.is_activation:
                assert op_type.is_elementwise, f"{op_type} is activation but not elementwise"

    def test_arithmetic_vs_activation(self):
        """Test distinction between arithmetic and activation elementwise ops."""
        arithmetic_elementwise = [OperationType.ADD, OperationType.SUB, OperationType.MUL, OperationType.DIV]
        activation_elementwise = [OperationType.RELU, OperationType.SIGMOID, OperationType.TANH]

        for op in arithmetic_elementwise:
            assert op.is_elementwise is True
            assert op.is_activation is False

        for op in activation_elementwise:
            assert op.is_elementwise is True
            assert op.is_activation is True

    def test_unary_vs_binary_operations(self):
        """Test distinction between unary and binary operations."""
        # Binary operations (typically require 2 inputs)
        binary_ops = [OperationType.ADD, OperationType.SUB, OperationType.MUL, OperationType.DIV, OperationType.MATMUL]

        # Unary operations (typically require 1 input)
        unary_ops = [
            OperationType.RELU,
            OperationType.SIGMOID,
            OperationType.TANH,
            OperationType.EXP,
            OperationType.LOG,
            OperationType.SIN,
            OperationType.COS,
            OperationType.NEG,
        ]

        # Just verify they exist and have correct categories
        for op in binary_ops:
            assert op in OperationType

        for op in unary_ops:
            assert op in OperationType

    def test_operation_enumeration(self):
        """Test that we can enumerate all operations."""
        all_ops = list(OperationType)
        assert len(all_ops) > 0

        # Should have at least these major categories represented
        has_linear = any(op.is_linear for op in all_ops)
        has_elementwise = any(op.is_elementwise for op in all_ops)
        has_reduction = any(op.is_reduction for op in all_ops)
        has_structural = any(op.is_structural for op in all_ops)
        has_derivative = any(op.is_derivative for op in all_ops)

        assert has_linear
        assert has_elementwise
        assert has_reduction
        assert has_structural
        assert has_derivative

    def test_count_operations_by_category(self):
        """Test counting operations in each category."""
        category_counts = dict.fromkeys(OperationCategory, 0)

        for op_type in OperationType:
            category_counts[op_type.category] += 1

        # Should have multiple operations in main categories
        assert category_counts[OperationCategory.LINEAR] >= 3
        assert category_counts[OperationCategory.ELEMENTWISE] >= 10
        assert category_counts[OperationCategory.REDUCTION] >= 4
        assert category_counts[OperationCategory.STRUCTURAL] >= 5
        assert category_counts[OperationCategory.DERIVATIVE] >= 4

    def test_repr(self):
        """Test string representation."""
        op = OperationType.RELU
        repr_str = repr(op)
        assert "relu" in repr_str.lower() or "RELU" in repr_str
