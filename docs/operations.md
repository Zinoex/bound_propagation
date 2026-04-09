# Tracer Operation Constraints

## Overview

The tracer now accepts any operation that can be represented in the IR mapping and is compatible with PyTorch autograd/FX tracing behavior.

## Policy

An operation is accepted when:
1. It can be traced by `torch.fx` in the model context (no unsupported dynamic Python control flow).
2. It is mapped to an internal `OperationType` by `tracer/op_mapping.py`.

This intentionally includes non-smooth and discrete-style ops when supported by PyTorch autograd conventions.

Examples currently accepted:
- Smooth functions: `sigmoid`, `tanh`, `exp`, `log`, `sin`, `cos`
- Piecewise functions: `relu`, `abs`, `clamp`, `heaviside`
- Reductions: `sum`, `mean`, `min`, `max`
- Structural/discrete indexing: slicing (`getitem`), `gather`, `split`, `reshape`, `flatten`

Unsupported operations are those not in the mapping (or not traceable by `torch.fx` in the traced form).

## Why This Policy?

This keeps tracing flexible and aligned with practical PyTorch usage while preserving a clear conversion boundary.

Operation-specific bound tightness remains the responsibility of downstream propagator strategies.

## Notes On Requested Cases

- `heaviside` is accepted at tracing time and mapped to the IR as a clamp-like piecewise op.
- `min`/`max` are accepted.
- Discrete indexing (`getitem`, `gather`) is accepted.

## Error Handling

If an operation is not mapped, conversion fails with an `Unsupported operation` error from the converter.

## Checking Operation Mapping

```python
from bound_propagation.tracer.op_mapping import get_operation_type

print(get_operation_type(torch.max))        # OperationType.MAX
print(get_operation_type(torch.min))        # OperationType.MIN
print(get_operation_type(torch.gather))     # OperationType.GATHER
print(get_operation_type(torch.heaviside))  # OperationType.CLAMP
```

## Implementation Details

Conversion validates only that an operation can be mapped:

```python
def _convert_operation_node(self, fx_node: fx.Node) -> Node:
    op_type = get_operation_type(fx_node.target)

    # Continue with conversion...
```

If an op is mapped but lacks a bound strategy implementation, that error appears later during propagation.
