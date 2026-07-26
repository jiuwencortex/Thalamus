#!/usr/bin/env python3
"""
Phase 1-2: Score all components (skills, memory sections, tools).

Python equivalent of run_01_score.sh — import-and-call so the full call stack
is visible in the debugger (no subprocess boundary).

Config via environment variables (same names as the shell runner):
  SKILLS_DIR      Directory with skill subdirectories (each has SKILL.md)
  PROJECT_DIR     Directory containing project.md / user.md
  ORACLE_DIR      Output directory for scoring_matrix_*.json files
  MODEL           LLM model name            (default: gpt-4o-mini)
  OPENAI_API_KEY  API key
  API_BASE        API base URL              (default: https://api.openai.com/v1)
  N_EXAMPLES      Pairs per component       (default: 20)
  PARALLEL        Concurrent LLM calls      (default: 5)

Tool directory resolution (three strategies, in priority order):
  1. Explicit env vars:  AGENT_CORE_DIR, JIUWENSWARM_DIR, TOOLS_DIR
  2. importlib probe:    finds installed openjiuwen / jiuwenswarm packages
  3. pip show fallback:  parses pip show openjiuwen
  Set TOOLS_DIR=none to skip tool scoring entirely.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


# ── Tool directory discovery ─────────────────────────────────────────────────

def _probe_importlib(module_name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    locs = getattr(spec, "submodule_search_locations", None)
    if locs:
        return list(locs)[0]
    if spec.origin:
        return str(Path(spec.origin).parent)
    return None


def resolve_tool_dirs() -> list[str]:
    tools_dir_env   = os.environ.get("TOOLS_DIR", "")
    agent_core_dir  = os.environ.get("AGENT_CORE_DIR", "")
    jiuwenswarm_dir = os.environ.get("JIUWENSWARM_DIR", "")

    if tools_dir_env == "none":
        print("  [tools] Skipping tool scoring (TOOLS_DIR=none)")
        return []

    dirs: list[str] = []

    # Strategy 1 — explicit env vars
    if agent_core_dir:
        candidate = os.path.join(agent_core_dir, "openjiuwen", "harness", "tools")
        if os.path.isdir(candidate):
            dirs.append(candidate)
            print(f"  [tools] AGENT_CORE_DIR: {candidate}")

    if jiuwenswarm_dir:
        candidate = os.path.join(jiuwenswarm_dir, "jiuwenswarm", "agents", "harness", "code", "tools")
        if os.path.isdir(candidate):
            dirs.append(candidate)
            print(f"  [tools] JIUWENSWARM_DIR: {candidate}")

    if tools_dir_env and os.path.isdir(tools_dir_env):
        dirs.append(tools_dir_env)
        print(f"  [tools] TOOLS_DIR: {tools_dir_env}")
    elif tools_dir_env:
        print(f"  [tools] WARNING: TOOLS_DIR set but not found: {tools_dir_env}", file=sys.stderr)

    # Strategy 2 — importlib probe
    if not dirs:
        for mod, label in [
            ("openjiuwen.harness.tools",              "openjiuwen.harness.tools"),
            ("jiuwenswarm.agents.harness.code.tools", "jiuwenswarm tools"),
        ]:
            path = _probe_importlib(mod)
            if path and os.path.isdir(path):
                dirs.append(path)
                print(f"  [tools] importlib → {label}: {path}")

    # Strategy 3 — pip show fallback
    if not dirs:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "openjiuwen"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Location:"):
                loc = line.split(":", 1)[1].strip()
                candidate = os.path.join(loc, "openjiuwen", "harness", "tools")
                if os.path.isdir(candidate):
                    dirs.append(candidate)
                    print(f"  [tools] pip show → {candidate}")
                break

    if not dirs:
        print(
            "  [tools] No tool directories found — tool scoring will be skipped.\n"
            "          To enable, set one of:\n"
            "            AGENT_CORE_DIR=/path/to/agent-core\n"
            "            JIUWENSWARM_DIR=/path/to/jiuwenswarm\n"
            "            TOOLS_DIR=/path/to/custom/tools"
        )

    return dirs


# ── Repair helpers ───────────────────────────────────────────────────────────

# Map matrix filename pattern → state file
_MATRIX_STATE_MAP = [
    ("scoring_matrix_skill_", "matrix_state_skills.json"),
    ("scoring_matrix_mem_",   "matrix_state_memory.json"),
    ("scoring_matrix_tool_",  "matrix_state_tools.json"),
]


def repair_empty_matrices(oracle_dir: Path) -> int:
    """Remove empty (zero-row) scoring matrices and their state entries.

    Scans oracle_dir for scoring_matrix_*.json files whose baseline_cross_eval
    list is empty.  For each such file:
      - Removes the component's entry from the corresponding state file.
      - Deletes the empty matrix file.

    This allows the next scoring run to re-score only the failed components
    without forcing a full re-score of all others.

    Returns the count of matrices repaired.
    """
    import json

    repaired = 0
    for matrix_path in sorted(oracle_dir.glob("scoring_matrix_*.json")):
        try:
            data = json.loads(matrix_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        rows = data.get("baseline_cross_eval", [])
        if rows:
            continue  # healthy — leave alone

        component_name = data.get("component_name") or data.get("skill_name") or matrix_path.stem
        stem = matrix_path.name

        # Determine which state file covers this matrix
        state_file = None
        for prefix, sf in _MATRIX_STATE_MAP:
            if stem.startswith(prefix):
                state_file = sf
                break
        if state_file is None:
            continue

        state_path = oracle_dir / state_file
        state_updated = False
        if state_path.exists():
            try:
                state_data = json.loads(state_path.read_text(encoding="utf-8"))
                components = state_data.get("components", state_data.get("skills", {}))
                if component_name in components:
                    del components[component_name]
                    # Write back (normalize to "components" key)
                    state_data["components"] = components
                    state_data.pop("skills", None)
                    state_path.write_text(
                        json.dumps(state_data, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    state_updated = True
            except (json.JSONDecodeError, OSError):
                pass

        matrix_path.unlink()
        repaired += 1
        print(
            f"  REPAIR: deleted empty matrix {matrix_path.name}"
            + (f" + removed from {state_file}" if state_updated else ""),
            flush=True,
        )

    return repaired


# ── Runner ───────────────────────────────────────────────────────────────────

def main() -> None:
    skills_dir  = os.path.expanduser(os.environ.get("SKILLS_DIR",  "~/.jiuwenswarm/agent/workspace/skills"))
    project_dir = os.path.expanduser(os.environ.get("PROJECT_DIR", "~/.jiuwenswarm/agent/workspace"))
    oracle_dir  = os.path.expanduser(os.environ.get("ORACLE_DIR",  "~/.jiuwenswarm/agent/workspace/oracle"))
    model       = os.environ.get("MODEL",          "deepseek-v4-flash")
    api_key     = os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", "sk-30b1b0d13d7a467bb30516be6a0dda8f"))
    api_base    = os.environ.get("API_BASE",       "https://api.deepseek.com")
    n_examples  = os.environ.get("N_EXAMPLES",    "20")
    parallel    = os.environ.get("PARALLEL",       "5")

    if not api_key:
        print("ERROR: Set OPENAI_API_KEY before running.", file=sys.stderr)
        sys.exit(1)

    # Auto-repair: delete empty matrix files and clear their state entries
    # so they are re-scored this run without touching healthy components.
    n_repaired = repair_empty_matrices(Path(oracle_dir))
    if n_repaired:
        print(f"  Repaired {n_repaired} empty matrix file(s); they will be re-scored now.")
    else:
        print("  No empty matrix files found (nothing to repair).")

    tool_dirs = resolve_tool_dirs()

    print()
    print("=== Phase 1-2: Component Scoring ===")
    print(f"  Skills dir : {skills_dir}")
    print(f"  Project dir: {project_dir}")
    print(f"  Oracle dir : {oracle_dir}")
    print(f"  Model      : {model}")
    print(f"  Tool dirs  : {len(tool_dirs)} dir(s) queued")
    print()

    from thalamus.scoring.cli import main as score_main  # noqa: PLC0415

    common = [
        "--matrix-dir", oracle_dir,
        "--model",      model,
        "--api-key",    api_key,
        "--api-base",   api_base,
        "--n-examples", n_examples,
        "--parallel",   parallel,
    ]

    # Score skills
    sys.argv = ["thalamus-score", "build", "--type", "skills",
                "--skills-dir", skills_dir, "--project-dir", project_dir,
                *common]
    score_main()

    # Score memory sections
    sys.argv = ["thalamus-score", "build", "--type", "memory",
                "--project-dir", project_dir, *common]
    score_main()

    # Score tools
    if tool_dirs:
        tool_args: list[str] = []
        for d in tool_dirs:
            tool_args += ["--tools-dir", d]
        sys.argv = ["thalamus-score", "build", "--type", "tools",
                    *tool_args, *common]
        score_main()
    else:
        print("Tool scoring skipped — no tool directories available.")

    print()
    print(f"Done. Scoring matrices written to: {oracle_dir}")


if __name__ == "__main__":
    main()
