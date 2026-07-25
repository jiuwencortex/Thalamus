# recommendation_matrix/tools/scanner.py
# Discover tools by scanning Python source files in tool directories.
from __future__ import annotations

import ast
import re
from pathlib import Path

from ..shared.fingerprint import ComponentRecord


class ToolCodeScanner:
    """Discover tool definitions by walking Python source directories.

    Scans the given directories recursively, parses each .py file with ast,
    and extracts classes whose name contains 'Tool' and which have a docstring.
    """

    def __init__(self, tool_dirs: list[Path]):
        self._tool_dirs = tool_dirs

    def scan(self) -> list[ComponentRecord]:
        """Walk directories, parse Python files, return one ComponentRecord per discovered tool."""
        records: list[ComponentRecord] = []
        seen_names: set[str] = set()

        for root_dir in self._tool_dirs:
            if not root_dir.exists():
                continue
            for py_path in _iter_py_files(root_dir):
                for klass in _extract_tool_classes(py_path):
                    if klass.name in seen_names:
                        continue
                    seen_names.add(klass.name)
                    records.append(ComponentRecord(
                        name=klass.name,
                        description=klass.docstring[:200] if klass.docstring else klass.name,
                        body=_build_body(klass),
                        mtime=py_path.stat().st_mtime,
                        directory=py_path.parent,
                        source_file=py_path.name,
                    ))
        return records

    def scan_and_filter(self, only: list[str] | None) -> list[ComponentRecord]:
        tools = self.scan()
        if only:
            tools = [t for t in tools if t.name in only]
        return tools


def _iter_py_files(root: Path):
    """Yield all .py files under root, excluding tests and __init__.py."""
    for path in root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        if "/test" in str(path).replace("\\", "/") or path.name.startswith("test_"):
            continue
        yield path


class _ToolClassInfo:
    def __init__(self, name: str, docstring: str, source_lines: list[str]):
        self.name = name
        self.docstring = docstring
        self.source_lines = source_lines


# Entry-point method names that distinguish runnable tools from schema/output/backend classes.
_ENTRY_POINT_METHODS = {"invoke", "stream", "__call__", "run", "execute"}


def _extract_tool_classes(py_path: Path) -> list[_ToolClassInfo]:
    """Parse a Python file and return info for classes that look like tools.

    A class is accepted when ALL of the following hold:
      1. Its name contains "Tool" and does not start with "_".
      2. It defines at least one entry-point method (invoke / stream / __call__ /
         run / execute) — distinguishing actual runnable tools from schema classes,
         output types, and backend helpers (e.g. LspToolOutput, CronToolBackend).

    Description resolution (in order):
      a. Class-level docstring.
      b. Docstring of the first `invoke` or `__call__` method (many agent-core
         tools lack a class docstring but document their `invoke` method).
      c. Class name converted to a readable phrase ("ReadFileTool" → "Read file tool").
    """
    try:
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    lines = source.splitlines(keepends=True)
    results: list[_ToolClassInfo] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if "Tool" not in node.name or node.name.startswith("_"):
            continue

        # Collect direct child method nodes (not nested classes).
        method_nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_nodes[child.name] = child

        if not set(method_nodes).intersection(_ENTRY_POINT_METHODS):
            continue

        # Resolve description.
        docstring = ast.get_docstring(node) or ""
        if not docstring.strip():
            # Try invoke / __call__ method docstring.
            for ep in ("invoke", "__call__"):
                if ep in method_nodes:
                    docstring = ast.get_docstring(method_nodes[ep]) or ""
                    if docstring.strip():
                        break
        if not docstring.strip():
            # Last resort: readable class name.
            docstring = _class_name_to_description(node.name)

        # Extract class source lines (approximate — from class def to next top-level node).
        start_line = node.lineno - 1
        end_line = _find_class_end_line(node, lines)
        class_source = "".join(lines[start_line:end_line])
        results.append(_ToolClassInfo(node.name, docstring, [class_source]))

    return results


def _class_name_to_description(name: str) -> str:
    """Convert a CamelCase class name to a readable description.

    Examples:
        ReadFileTool  → "Read file"
        WebFetchWebpageTool → "Web fetch webpage"
    """
    # Strip trailing "Tool"
    if name.endswith("Tool"):
        name = name[:-4]
    # Insert spaces before uppercase letters that follow lowercase.
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return spaced.strip()


def _find_class_end_line(node: ast.ClassDef, lines: list[str]) -> int:
    """Return the line number after the class definition ends."""
    # Use the last line of the last statement in the class body
    if not node.body:
        return node.end_lineno or node.lineno
    last_stmt = node.body[-1]
    end = getattr(last_stmt, "end_lineno", None) or node.lineno
    return min(end + 1, len(lines))


def _build_body(klass: _ToolClassInfo) -> str:
    """Build a textual body from the class info for use in query generation."""
    return klass.docstring
