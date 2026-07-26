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
    spec = importlib.util.find_spec(module_name)
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


# ── Runner ───────────────────────────────────────────────────────────────────

def main() -> None:
    skills_dir  = os.environ.get("SKILLS_DIR",    "~/.jiuwenswarm/agent/workspace/skills")
    project_dir = os.environ.get("PROJECT_DIR",   "~/.jiuwenswarm/agent/workspace")
    oracle_dir  = os.environ.get("ORACLE_DIR",    "~/.jiuwenswarm/agent/workspace/oracle")
    model       = os.environ.get("MODEL",          "gpt-4o-mini")
    api_key     = os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", ""))
    api_base    = os.environ.get("API_BASE",       "https://api.openai.com/v1")
    n_examples  = os.environ.get("N_EXAMPLES",    "20")
    parallel    = os.environ.get("PARALLEL",       "5")

    if not api_key:
        print("ERROR: Set OPENAI_API_KEY before running.", file=sys.stderr)
        sys.exit(1)

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
