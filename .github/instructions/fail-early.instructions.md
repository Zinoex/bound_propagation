---
description: "Use when writing or reviewing Python code in this project. Enforces fail-early, fail-loud error handling: validate at boundaries, raise immediately, never swallow exceptions silently."
applyTo: "**/*.py"
---
# Fail Early, Fail Loud

## Core Rules

- **Validate at public API entry points.** Check inputs before doing any computation. Every public method/constructor with preconditions must verify them at the top.
- **Raise `Exception` immediately.** Don't defer, log-and-continue, or return sentinel values (`None`, `-1`) to signal failure. Raise an exception right where the bad state is detected. 
- **Never use `assert` for runtime validation.** `assert` is silently disabled with `python -O`. Use explicit `if / raise` instead.
- **Never swallow exceptions silently.** Avoid bare `except Exception: return None` or `except Exception: pass`. If you must catch a broad exception, re-raise with context (`raise ConversionError(...) from e`) or let it propagate.

## Exception Types

Use built-ins for standard violations:

| Condition | Exception |
|-----------|-----------|
| Wrong value (e.g. negative log input, mismatched dims) | `ValueError` |
| Wrong type | `TypeError` |
| Unimplemented operation | `NotImplementedError` |

Use the project's custom hierarchy for domain errors:

```python
# tracer/fx_tracer.py
class TraceError(Exception): ...
class UnsupportedOperationError(TraceError): ...
class ConversionError(Exception): ...
```

Add custom exception types for domain-specific errors if needed.

Raise the most specific exception available. When wrapping a lower-level error, always chain it:

```python
except SomeError as e:
    raise ConversionError(f"Cannot convert node '{node.name}': ...") from e
```

## Error Messages

Messages must answer: **what** failed, **why**, and (when possible) **what value was seen**.

```python
# Bad
raise ValueError("Invalid bounds")

# Good
raise ValueError(
    f"LinearBounds requires lower <= upper, "
    f"but got lower={lower!r}, upper={upper!r}"
)
```

## Patterns to Follow

**Constructor validation** — see `bounds/linear_bounds.py`:
```python
def __init__(self, lower, upper):
    if lower.shape != upper.shape:
        raise ValueError(
            f"Shape mismatch: lower {lower.shape} vs upper {upper.shape}"
        )
    if torch.any(lower > upper):
        raise ValueError("lower must be <= upper element-wise")
```

**Domain constraints before computation** — see `propagation/ibp/log.py`:
```python
if torch.any(x_bounds.lower <= 0):
    raise ValueError(
        "log requires strictly positive input bounds, "
        f"but lower bound contains values <= 0"
    )
```

**Graph integrity** — see `ir/graph.py`: run a structured validation pass after assembly, not inline.

## Anti-Patterns to Avoid

```python
# BAD: silent swallow
try:
    result = compute(x)
except Exception:
    return None

# BAD: assert (disabled with -O)
assert isinstance(node, IRNode), "expected IRNode"

# BAD: vague message
raise ValueError("bad input")

# BAD: deferred error (computed wrong value, fail later)
if lower > upper:
    lower, upper = upper, lower  # silently "fix" it
```

## Validation Checklist for New Code

- [ ] Public constructors validate all preconditions before storing state
- [ ] Public methods validate inputs before doing work
- [ ] No `assert` statements outside test files
- [ ] No bare `except` that discards the exception
- [ ] Error messages include concrete values, not just type names
- [ ] Exceptions are chained (`from e`) when wrapping lower-level errors
