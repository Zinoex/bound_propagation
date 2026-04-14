#!/usr/bin/env python3
"""Mechanical transformation script for updating test files to new API.

Transforms:
1. propagate_forwards(node=None, input_bounds=[...]) -> propagate(strategy, ...)
2. propagate_forwards(None, [...]) -> propagate(strategy, ...)
3. propagate_forwards(node, [...]) -> propagate(strategy, ...)
4. Removes IBP*WithConstant / IBPConstantMatmul / IBPMatmulConstant imports
5. Removes ForwardLBP*WithConstant etc.
6. Adds `from tests.helpers import propagate` import
7. Removes old ir imports that were only used for Node construction
"""

import re
from pathlib import Path

# ── Class rename map ────────────────────────────────────────────────

OLD_TO_NEW = {
    # IBP
    "IBPAddWithConstant": "IBPAdd",
    "IBPMulWithConstant": "IBPMul",
    "IBPSubWithConstant": "IBPSub",
    "IBPConstantSub": "IBPSub",
    "IBPDivWithConstant": "IBPDiv",
    "IBPConstantDiv": "IBPDiv",
    "IBPMatmulConstant": "IBPMatmul",
    "IBPConstantMatmul": "IBPMatmul",
    # Forward LBP
    "ForwardLBPAddWithConstant": "ForwardLBPAdd",
    "ForwardLBPMulWithConstant": "ForwardLBPMul",
    "ForwardLBPSubWithConstant": "ForwardLBPSub",
    "ForwardLBPConstantSub": "ForwardLBPSub",
    "ForwardLBPDivWithConstant": "ForwardLBPDiv",
    "ForwardLBPConstantDiv": "ForwardLBPDiv",
    "ForwardLBPMatmulConstant": "ForwardLBPMatmul",
    "ForwardLBPConstantMatmul": "ForwardLBPMatmul",
}


def fix_imports(content: str) -> str:
    """Remove deleted class names from import lines."""
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        if "import" in line and "bound_propagation" in line:
            # Remove any old class names from import statements
            for old, new in OLD_TO_NEW.items():
                # "import OldName, " or ", OldName" or ", OldName,"
                line = re.sub(rf",\s*{old}(?=\s*[,\n)]|$)", "", line)
                line = re.sub(rf"{old},\s*", "", line)
                # Solo import of old name
                line = re.sub(rf"import {old}$", f"import {new}", line)
            # Clean up any double commas or trailing commas
            line = re.sub(r",\s*,", ",", line)
            line = re.sub(r",\s*$", "", line)
            line = re.sub(r"import\s*,", "import ", line)
        new_lines.append(line)
    return "\n".join(new_lines)


def remove_old_ir_imports(content: str) -> str:
    """Remove import lines pulling in Node, OperationType, TensorMetadata from bound_propagation.ir."""
    lines = content.split("\n")
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this is an IR import line
        if re.match(r"\s*from bound_propagation\.ir import ", line):
            # Collect the full import (may span parens)
            full = line
            while "(" in full and ")" not in full:
                i += 1
                full += "\n" + lines[i]
            # Extract what's being imported
            m = re.search(r"import\s+(?:\()?(.+?)(?:\))?\s*$", full, re.DOTALL)
            if m:
                names = [n.strip().rstrip(",") for n in re.split(r"[,\n]", m.group(1)) if n.strip()]
                # Remove old IR names
                old_ir = {"Node", "NodeType", "OperationType", "TensorMetadata", "AbstractValueType", "Graph"}
                remaining = [n for n in names if n not in old_ir and n]
                if not remaining:
                    # Drop the entire import line
                    i += 1
                    continue
                else:
                    # Keep just the remaining names
                    line = re.sub(r"import .+", f"import {', '.join(remaining)}", lines[i if "(" not in full else i])
        new_lines.append(line)
        i += 1
    return "\n".join(new_lines)


def replace_class_instantiation(content: str) -> str:
    """Replace OldClass() with NewClass()."""
    for old, new in OLD_TO_NEW.items():
        content = content.replace(f"{old}()", f"{new}()")
    return content


def replace_propagate_calls(content: str) -> tuple[str, bool]:
    """Replace .propagate_forwards(...) calls with propagate(strategy, ...).

    Returns (new_content, needs_import).
    """
    changed = False

    # Pattern: VAR.propagate_forwards(NODE_OR_NONE, [ARGS])
    # or: VAR.propagate_forwards(NODE_OR_NONE, input_bounds=[ARGS])
    # Where NODE_OR_NONE is None, node=None, or a variable name
    def replacer(match):
        nonlocal changed
        changed = True
        before = match.group("before")  # everything before .propagate_forwards
        strategy = match.group("strategy")
        args_str = match.group("args")
        after = match.group("after") if match.group("after") else ""
        return f"{before}propagate({strategy}, {args_str}){after}"

    # General pattern capturing strategy.propagate_forwards(ANY, [ARGS]) or (ANY, input_bounds=[ARGS])
    pattern = (
        r"(?P<before>.*?)"
        r"(?P<strategy>\w+)\.propagate_forwards\("
        r"(?:node=)?(?:None|\w+),\s*"
        r"(?:input_bounds=)?"
        r"\[(?P<args>[^\]]+)\]"
        r"\)"
        r"(?P<after>\s*#[^\n]*)?"
    )

    new_lines = []
    for line in content.split("\n"):
        m = re.match(pattern, line)
        if m:
            before = m.group("before")
            strategy = m.group("strategy")
            args_str = m.group("args")
            new_lines.append(f"{before}propagate({strategy}, {args_str})")
            changed = True
        else:
            new_lines.append(line)

    return "\n".join(new_lines), changed


def add_propagate_import(content: str) -> str:
    """Add 'from tests.helpers import propagate' after the last bound_propagation import."""
    if "from tests.helpers import propagate" in content:
        return content

    lines = content.split("\n")
    insert_after = -1
    for i, line in enumerate(lines):
        if line.startswith("from bound_propagation."):
            insert_after = i

    if insert_after >= 0:
        lines.insert(insert_after + 1, "")
        lines.insert(insert_after + 2, "from tests.helpers import propagate")

    return "\n".join(lines)


def remove_blank_line_runs(content: str) -> str:
    """Collapse 3+ consecutive blank lines to 2."""
    return re.sub(r"\n{4,}", "\n\n\n", content)


def transform_file(filepath: Path) -> bool:
    """Transform a single test file. Returns True if changed."""
    content = filepath.read_text()
    original = content

    content = fix_imports(content)
    content = remove_old_ir_imports(content)
    content = replace_class_instantiation(content)
    content, needs_import = replace_propagate_calls(content)

    if needs_import:
        content = add_propagate_import(content)

    content = remove_blank_line_runs(content)

    if content != original:
        filepath.write_text(content)
        return True
    return False


def main() -> None:
    workspace = Path("/home/fbmathiesen/Documents/bound_propagation")
    tests_dir = workspace / "tests"

    skip = {"test_signature_dispatch.py", "test_full_workflow.py", "__init__.py"}

    for subdir in ["test_ibp", "test_forward_lbp"]:
        test_dir = tests_dir / subdir
        if not test_dir.exists():
            continue
        for test_file in sorted(test_dir.glob("test_*.py")):
            if test_file.name in skip:
                print(f"  SKIP {subdir}/{test_file.name}")
                continue
            changed = transform_file(test_file)
            status = "CHANGED" if changed else "unchanged"
            print(f"  {status}: {subdir}/{test_file.name}")


if __name__ == "__main__":
    main()
