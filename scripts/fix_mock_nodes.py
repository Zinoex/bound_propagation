#!/usr/bin/env python3
"""Second-pass transformation: remove MockNode definitions and pass attributes as kwargs to propagate().

Pattern before:
    class MockNode:
        def __init__(self):
            self.attributes = {"dim": 1, "keepdim": False}

    node = MockNode()
    result = propagate(strategy, bounds)

Pattern after:
    result = propagate(strategy, bounds, dim=1, keepdim=False)
"""

import re
from pathlib import Path


def extract_attributes(mock_block: str) -> dict[str, str]:
    """Extract attribute key-value pairs from a MockNode class definition."""
    m = re.search(r"self\.attributes\s*=\s*\{([^}]*)\}", mock_block, re.DOTALL)
    if not m:
        return {}
    pairs_str = m.group(1).strip()
    if not pairs_str:
        return {}

    result = {}
    # Parse "key": value pairs
    for pair in re.finditer(r'"(\w+)"\s*:\s*([^,}]+)', pairs_str):
        key = pair.group(1)
        value = pair.group(2).strip()
        result[key] = value
    return result


def transform_file(filepath: Path) -> bool:
    """Transform MockNode patterns in a file. Returns True if changed."""
    content = filepath.read_text()
    original = content

    # Find all MockNode class definitions and their associated node = MockNode() + propagate() calls
    # We process from bottom to top to avoid offset issues

    # Pattern: class MockNode:\n        def __init__(self):\n            self.attributes = {...}\n\n    node = MockNode()\n    ... = propagate(...)
    pattern = re.compile(
        r"(?P<indent>[ \t]*)"
        r"class MockNode:\n"
        r"(?P<indent2>[ \t]*)def __init__\(self\):\n"
        r"(?P<indent3>[ \t]*)self\.attributes\s*=\s*\{(?P<attrs>[^}]*)\}\n"
        r"\n"
        r"(?P=indent)node = MockNode\(\)\n",
        re.MULTILINE,
    )

    matches = list(pattern.finditer(content))

    if not matches:
        return False

    # Process from bottom to top
    for m in reversed(matches):
        attrs = extract_attributes(m.group(0))

        # Remove the MockNode block
        start = m.start()
        end = m.end()

        # Find the next propagate() call after the MockNode block
        after = content[end:]
        prop_match = re.search(r"(propagate\([^)]+)\)", after)

        if prop_match and attrs:
            # Add kwargs to the propagate call
            kwargs_str = ", ".join(f"{k}={v}" for k, v in attrs.items())
            old_call = prop_match.group(1) + ")"
            new_call = prop_match.group(1) + ", " + kwargs_str + ")"

            # Replace the propagate call
            prop_start = end + prop_match.start()
            prop_end = end + prop_match.end()
            content = content[:prop_start] + new_call + content[prop_end:]

        # Remove the MockNode definition
        content = content[:start] + content[end:]

    # Clean up multiple blank lines
    content = re.sub(r"\n{3,}", "\n\n", content)

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
                continue
            changed = transform_file(test_file)
            if changed:
                print(f"  CHANGED: {subdir}/{test_file.name}")


if __name__ == "__main__":
    main()
