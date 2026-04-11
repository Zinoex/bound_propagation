---
description: "Use when writing or reviewing Python imports. Enforces Ruff (isort) import sorting and grouping conventions."
applyTo: "**/*.py"
---
# Ruff Import Sorting

## Rule

All Python imports must be sorted and grouped according to Ruff isort rules.
Do not hand-format import ordering when Ruff can do it.

## Required Workflow

- After editing imports, run: `ruff check --select I --fix .`
- If additional formatting changes are needed, run: `ruff format .`
- Ensure no import-sorting violations remain before finishing.

## Grouping Expectations

- Standard library imports first.
- Third-party imports next.
- First-party/local imports last.
- Keep one blank line between groups.
- Within each group, sort imports alphabetically.
- Prefer one import per line unless Ruff combines them.

## Interaction With Project Import Rules

- This file controls ordering and grouping only.
- For intra-package import style (relative imports, re-export usage), follow [imports.instructions.md](imports.instructions.md).

## Examples

```python
# BAD
from .local_b import B
import torch
import os
from .local_a import A

# GOOD
import os

import torch

from .local_a import A
from .local_b import B
```
