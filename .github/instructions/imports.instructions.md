---
description: "Use when writing or reviewing Python imports in src/bound_propagation. Enforces relative intra-package imports and mandatory use of highest-level re-exports."
applyTo: "src/bound_propagation/**/*.py"
---
# Import Rules

## Scope

These rules apply to Python modules inside `src/bound_propagation`.

## Rules

- Use relative imports for intra-project imports within `src/bound_propagation`.
- Do not use absolute package imports like `from bound_propagation...` from inside the package.
- If a symbol is re-exported by a higher-level local package `__init__.py`, import from that highest local re-export point.
- Exception: when you are inside the same submodule that defines/re-exports the symbol, import directly from local sibling modules (for example, `from .foo import Bar`), not from the submodule itself.

## Examples

```python
# BAD (absolute import from inside package)
from bound_propagation.bounds.interval_bounds import IntervalBounds

# GOOD (relative intra-package import)
from ...bounds import IntervalBounds
```

```python
# BAD (imports from deep module when public re-export exists)
from ...ir.node import Node
from ...ir.operations import OperationType

# GOOD (use highest local re-export)
from ...ir import Node, OperationType
```

```python
# Inside src/bound_propagation/ir/node.py
# BAD (self-package re-export import from within same submodule)
from . import OperationType

# GOOD (local sibling import)
from .operations import OperationType
```
