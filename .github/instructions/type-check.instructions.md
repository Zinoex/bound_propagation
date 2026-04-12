---
description: "Use when running or suggesting type-check commands in this project. Enforces ty execution through uvx."
---
# Type Check Command Rules

## Rule

Use `uvx ty check` for all type-check invocations in this repository.

Do not use bare `ty` commands.

## Canonical Commands

```bash
# Type check the codebase
uvx ty check src/

# Type check a specific subpath
uvx ty check src/bound_propagation/ir/
```

## Examples

```bash
# GOOD
uvx ty check src/
uvx ty check src/bound_propagation/ir/

# BAD
ty check src/
ty check src/bound_propagation/ir/
```

## Notes

- Keep normal ty flags and target paths; only the command prefix is enforced.
- This aligns with the project's tooling in `DEVELOPMENT.md`.
