---
description: "Use when running or suggesting test commands in this project. Enforces pytest execution through uv."
---
# Test Command Rules

## Rule

Use `uv run pytest` for all pytest invocations in this repository.

Do not use bare `pytest` commands.

## Examples

```bash
# GOOD
uv run pytest
uv run pytest tests/test_bounds.py
uv run pytest -k linear

# BAD
pytest
pytest tests/test_bounds.py
<path_to>/python3 -m pytest
```

## Notes

- Keep all normal pytest arguments and selectors; only the command prefix changes.
- This aligns with the project's tooling in `DEVELOPMENT.md`.
