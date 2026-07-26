# recommendation_matrix/shared/fingerprint.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ComponentRecord:
    name: str
    description: str
    body: str         # component content — SKILL.md body, memory section text, or tool description
    mtime: float      # modification time of source file
    directory: Path
    source_file: str = ""   # optional: source filename (used by memory sections)


# Backward-compatible alias used by skills/scanner.py and skill_matrix compat shim
SkillRecord = ComponentRecord


def component_fingerprint(component: ComponentRecord) -> str:
    """Hash a single component's identity (name + description + body content).
    Content-based: stable regardless of where the source file is installed or
    when it was last touched.  Changes only when actual content changes.
    """
    payload = json.dumps(
        {"name": component.name, "description": component.description, "body": component.body},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# Backward-compatible alias
skill_fingerprint = component_fingerprint


def global_fingerprint(components: list[ComponentRecord]) -> str:
    """Hash the entire component collection.
    Changes when any component is added, removed, or edited.
    Content-based: stable across venv reinstalls and path changes.
    """
    entries = sorted(
        [{"name": c.name, "description": c.description, "body": c.body} for c in components],
        key=lambda x: x["name"],
    )
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
