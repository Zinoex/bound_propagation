"""
Operation type taxonomy for computation graph nodes.

This module defines the types of operations that can appear in computation graphs,
organized by category. Each operation type corresponds to a specific mathematical
or structural transformation.
"""

from __future__ import annotations

from enum import StrEnum


class OperationType(StrEnum):
    """
    Enumeration of all supported operation types in computation graphs.

    Operations are organized into categories:
    - LINEAR: Linear transformations (matmul, linear layers, convolutions)
    - ELEMENTWISE: Element-wise operations (add, mul, activations)
    - REDUCTION: Operations that reduce dimensions (sum, mean, max)
    - STRUCTURAL: Operations that reorganize tensors (concat, select)
    - DERIVATIVE: Differentiation operations (jacobian, gradient)
    """

    # ============================================================================
    # LINEAR OPERATIONS
    # ============================================================================
    MATMUL = "matmul"  # Matrix multiplication: y = x @ W
    LINEAR = "linear"  # Affine transformation: y = x @ W + b
    CONV2D = "conv2d"  # 2D convolution (for future support)

    # ============================================================================
    # BASIC ARITHMETIC OPERATIONS
    # ============================================================================
    ADD = "add"  # Element-wise addition: y = x1 + x2
    SUB = "sub"  # Element-wise subtraction: y = x1 - x2
    MUL = "mul"  # Element-wise multiplication: y = x1 * x2
    DIV = "div"  # Element-wise division: y = x1 / x2
    POW = "pow"  # Element-wise power: y = x^p
    NEG = "neg"  # Element-wise negation: y = -x
    RECIPROCAL = "reciprocal"  # Element-wise reciprocal: y = 1/x
    MAXIMUM = "maximum"  # Element-wise maximum: y = max(x1, x2)
    MINIMUM = "minimum"  # Element-wise minimum: y = min(x1, x2)

    # ============================================================================
    # ELEMENTWISE OPERATIONS - Activations
    # ============================================================================
    RELU = "relu"  # Rectified Linear Unit: y = max(0, x)
    SIGMOID = "sigmoid"  # Sigmoid: y = 1 / (1 + exp(-x))
    TANH = "tanh"  # Hyperbolic tangent: y = tanh(x)
    EXP = "exp"  # Exponential: y = exp(x)
    LOG = "log"  # Natural logarithm: y = log(x)
    SQRT = "sqrt"  # Square root: y = sqrt(x)
    ABS = "abs"  # Absolute value: y = |x|

    # ============================================================================
    # ELEMENTWISE OPERATIONS - Trigonometric
    # ============================================================================
    SIN = "sin"  # Sine: y = sin(x)
    COS = "cos"  # Cosine: y = cos(x)
    TAN = "tan"  # Tangent: y = tan(x)

    # ============================================================================
    # ELEMENTWISE OPERATIONS - Other
    # ============================================================================
    CLAMP = "clamp"  # Clamp values: y = min(max(x, min_val), max_val)

    # ============================================================================
    # REDUCTION OPERATIONS
    # ============================================================================
    SUM = "sum"  # Sum reduction
    MEAN = "mean"  # Mean reduction
    MAX = "max"  # Maximum reduction
    MIN = "min"  # Minimum reduction

    # ============================================================================
    # STRUCTURAL OPERATIONS
    # ============================================================================
    CONCAT = "concat"  # Concatenation along a dimension
    STACK = "stack"  # Stack tensors along a new dimension
    SELECT = "select"  # Select specific elements/slices
    GATHER = "gather"  # Gather elements according to indices
    RESHAPE = "reshape"  # Reshape tensor (preserves data)
    FLATTEN = "flatten"  # Flatten tensor to 1D or 2D
    PERMUTE = "permute"  # Permute dimensions
    UNSQUEEZE = "unsqueeze"  # Add dimension of size 1
    SQUEEZE = "squeeze"  # Remove dimensions of size 1
    VIEW = "view"  # Reshape without copying (like PyTorch's view)
    TRANSPOSE = "transpose"  # Transpose operation
    GETITEM = "getitem"  # Indexing/slicing operation

    # ============================================================================
    # DERIVATIVE OPERATIONS
    # ============================================================================
    JACOBIAN = "jacobian"  # Jacobian matrix computation (via torch.func)
    GRADIENT = "gradient"  # Gradient computation (via torch.func.grad)
    VJP = "vjp"  # Vector-Jacobian product (via torch.func.vjp)
    JVP = "jvp"  # Jacobian-Vector product (via torch.func.jvp)

    # ============================================================================
    # SPECIAL OPERATIONS
    # ============================================================================
    INPUT = "input"  # Graph input node (placeholder)
    CONSTANT = "constant"  # Constant value
    PARAMETER = "parameter"  # Learnable parameter (e.g., weights)

    @property
    def category(self) -> OperationCategory:
        """Get the category this operation belongs to."""
        return _OPERATION_CATEGORIES.get(self, OperationCategory.OTHER)

    @property
    def is_elementwise(self) -> bool:
        """Check if this is an element-wise operation."""
        return self.category == OperationCategory.ELEMENTWISE

    @property
    def is_reduction(self) -> bool:
        """Check if this is a reduction operation."""
        return self.category == OperationCategory.REDUCTION

    @property
    def is_structural(self) -> bool:
        """Check if this is a structural operation."""
        return self.category == OperationCategory.STRUCTURAL

    @property
    def is_derivative(self) -> bool:
        """Check if this is a derivative operation."""
        return self.category == OperationCategory.DERIVATIVE


class OperationCategory(StrEnum):
    """High-level categories for operation types."""

    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    STRUCTURAL = "structural"
    DERIVATIVE = "derivative"
    OTHER = "other"


# Mapping from operation types to categories
_OPERATION_CATEGORIES: dict[OperationType, OperationCategory] = {
    # Linear
    OperationType.MATMUL: OperationCategory.OTHER,
    OperationType.LINEAR: OperationCategory.OTHER,
    # Basic arithmetic
    OperationType.ADD: OperationCategory.OTHER,
    OperationType.SUB: OperationCategory.OTHER,
    OperationType.MUL: OperationCategory.OTHER,
    OperationType.DIV: OperationCategory.OTHER,
    OperationType.POW: OperationCategory.ELEMENTWISE,
    OperationType.NEG: OperationCategory.ELEMENTWISE,
    OperationType.RECIPROCAL: OperationCategory.ELEMENTWISE,
    OperationType.MAXIMUM: OperationCategory.OTHER,
    OperationType.MINIMUM: OperationCategory.OTHER,
    # Elementwise - Activations
    OperationType.RELU: OperationCategory.ELEMENTWISE,
    OperationType.SIGMOID: OperationCategory.ELEMENTWISE,
    OperationType.TANH: OperationCategory.ELEMENTWISE,
    OperationType.EXP: OperationCategory.ELEMENTWISE,
    OperationType.LOG: OperationCategory.ELEMENTWISE,
    OperationType.SQRT: OperationCategory.ELEMENTWISE,
    OperationType.ABS: OperationCategory.ELEMENTWISE,
    # Elementwise - Trigonometric
    OperationType.SIN: OperationCategory.ELEMENTWISE,
    OperationType.COS: OperationCategory.ELEMENTWISE,
    OperationType.TAN: OperationCategory.ELEMENTWISE,
    # Elementwise - Other
    OperationType.CLAMP: OperationCategory.ELEMENTWISE,
    # Reduction
    OperationType.SUM: OperationCategory.REDUCTION,
    OperationType.MEAN: OperationCategory.REDUCTION,
    OperationType.MAX: OperationCategory.REDUCTION,
    OperationType.MIN: OperationCategory.REDUCTION,
    # Structural
    OperationType.CONCAT: OperationCategory.STRUCTURAL,
    OperationType.STACK: OperationCategory.STRUCTURAL,
    OperationType.SELECT: OperationCategory.STRUCTURAL,
    OperationType.RESHAPE: OperationCategory.STRUCTURAL,
    OperationType.FLATTEN: OperationCategory.STRUCTURAL,
    OperationType.PERMUTE: OperationCategory.STRUCTURAL,
    OperationType.UNSQUEEZE: OperationCategory.STRUCTURAL,
    OperationType.SQUEEZE: OperationCategory.STRUCTURAL,
    OperationType.VIEW: OperationCategory.STRUCTURAL,
    OperationType.TRANSPOSE: OperationCategory.STRUCTURAL,
    # Derivative
    OperationType.JACOBIAN: OperationCategory.DERIVATIVE,
    OperationType.GRADIENT: OperationCategory.DERIVATIVE,
    OperationType.VJP: OperationCategory.DERIVATIVE,
    OperationType.JVP: OperationCategory.DERIVATIVE,
}
