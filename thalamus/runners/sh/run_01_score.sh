#!/usr/bin/env bash
# Phase 1-2: Score all components (skills, memory sections, tools).
#
# What this does:
#   - Scans skill SKILL.md files, memory markdown sections, and tool Python files
#   - Calls the LLM to generate (query, answer) pairs per component
#   - Evaluates each component against those pairs with lexical metrics
#   - Writes scoring_matrix_*.json files to ORACLE_DIR
#
# Run this first, before run_02_oracle.sh.
#
# ── Tool directory resolution (three strategies, in order) ───────────────────
#
#   Strategy 1 — Explicit env vars (fastest, always wins if set):
#     AGENT_CORE_DIR=/path/to/agent-core   → scans <dir>/openjiuwen/harness/tools
#     JIUWENSWARM_DIR=/path/to/jiuwenswarm → scans <dir>/jiuwenswarm/agents/harness/code/tools
#     TOOLS_DIR=/path/to/custom/tools      → scans any extra directory you supply
#
#   Strategy 2 — importlib probe (works when openjiuwen is pip-installed):
#     import openjiuwen.harness.tools → extracts installation path automatically
#
#   Strategy 3 — pip show fallback:
#     pip show openjiuwen → parses Location: field → appends /openjiuwen/harness/tools
#
#   Set TOOLS_DIR=none to skip tool scoring entirely.

set -euo pipefail

# ── Configure these paths ────────────────────────────────────────────────────
SKILLS_DIR="${SKILLS_DIR:-$HOME/.jiuwenswarm/agent/workspace/skills}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/.jiuwenswarm/agent/workspace}"
ORACLE_DIR="${ORACLE_DIR:-$HOME/.jiuwenswarm/agent/workspace/oracle}"

# Explicit tool source overrides (leave unset for auto-discovery)
AGENT_CORE_DIR="${AGENT_CORE_DIR:-}"
JIUWENSWARM_DIR="${JIUWENSWARM_DIR:-}"
TOOLS_DIR="${TOOLS_DIR:-}"         # extra custom tool directory (or "none" to skip tools)

# ── LLM settings ─────────────────────────────────────────────────────────────
MODEL="${MODEL:-gpt-4o-mini}"
API_KEY="${OPENAI_API_KEY:-}"
API_BASE="${API_BASE:-https://api.openai.com/v1}"

# ── Tuning ───────────────────────────────────────────────────────────────────
N_EXAMPLES="${N_EXAMPLES:-20}"   # (query, answer) pairs generated per component
PARALLEL="${PARALLEL:-5}"        # concurrent LLM calls

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ -z "$API_KEY" ]]; then
  echo "ERROR: Set OPENAI_API_KEY (or export API_KEY=...) before running." >&2
  exit 1
fi

# ── Tool directory discovery ─────────────────────────────────────────────────
# Returns a list of directories, one per line.

declare -a TOOL_DIR_ARGS=()

_add_tool_dir() {
  local dir="$1"
  local label="$2"
  if [[ -d "$dir" ]]; then
    TOOL_DIR_ARGS+=("--tools-dir" "$dir")
    echo "  [tools] $label: $dir"
  fi
}

if [[ "$TOOLS_DIR" == "none" ]]; then
  echo "  [tools] Skipping tool scoring (TOOLS_DIR=none)"
else

  # ── Strategy 1: Explicit env var overrides ──────────────────────────────
  if [[ -n "$AGENT_CORE_DIR" ]]; then
    _add_tool_dir "$AGENT_CORE_DIR/openjiuwen/harness/tools" "AGENT_CORE_DIR (built-ins)"
  fi
  if [[ -n "$JIUWENSWARM_DIR" ]]; then
    _add_tool_dir "$JIUWENSWARM_DIR/jiuwenswarm/agents/harness/code/tools" "JIUWENSWARM_DIR (swarm tools)"
  fi
  if [[ -n "$TOOLS_DIR" && -d "$TOOLS_DIR" ]]; then
    _add_tool_dir "$TOOLS_DIR" "TOOLS_DIR (custom)"
  elif [[ -n "$TOOLS_DIR" ]]; then
    echo "  [tools] WARNING: TOOLS_DIR set but not found: $TOOLS_DIR" >&2
  fi

  # ── Strategy 2: importlib probe (works when openjiuwen is pip-installed) ─
  if [[ "${#TOOL_DIR_ARGS[@]}" -eq 0 ]]; then
    _probe_importlib() {
      python3 -c "
import importlib.util, pathlib, sys
spec = importlib.util.find_spec('$1')
if spec is None: sys.exit(1)
locs = getattr(spec, 'submodule_search_locations', None)
if locs:
    print(list(locs)[0])
elif spec.origin:
    print(str(pathlib.Path(spec.origin).parent))
else:
    sys.exit(1)
" 2>/dev/null
    }

    _ac=$(  _probe_importlib "openjiuwen.harness.tools"                    || true)
    _jw=$(  _probe_importlib "jiuwenswarm.agents.harness.code.tools"       || true)
    [[ -n "$_ac" ]] && _add_tool_dir "$_ac" "importlib → openjiuwen.harness.tools"
    [[ -n "$_jw" ]] && _add_tool_dir "$_jw" "importlib → jiuwenswarm tools"
  fi

  # ── Strategy 3: pip show fallback ────────────────────────────────────────
  if [[ "${#TOOL_DIR_ARGS[@]}" -eq 0 ]]; then
    _pip_loc=$(python3 -m pip show openjiuwen 2>/dev/null \
               | awk '/^Location:/ {print $2}')
    if [[ -n "$_pip_loc" ]]; then
      _add_tool_dir "$_pip_loc/openjiuwen/harness/tools" \
                    "pip show openjiuwen → harness/tools"
    fi
  fi

  # ── Nothing found ─────────────────────────────────────────────────────────
  if [[ "${#TOOL_DIR_ARGS[@]}" -eq 0 ]]; then
    echo "  [tools] No tool directories found — tool scoring will be skipped."
    echo "          To enable, set one of:"
    echo "            AGENT_CORE_DIR=/path/to/agent-core"
    echo "            JIUWENSWARM_DIR=/path/to/jiuwenswarm"
    echo "            TOOLS_DIR=/path/to/custom/tools"
  fi

fi  # end: TOOLS_DIR != "none"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 1-2: Component Scoring ==="
echo "  Skills dir : $SKILLS_DIR"
echo "  Project dir: $PROJECT_DIR"
echo "  Oracle dir : $ORACLE_DIR"
echo "  Model      : $MODEL"
echo "  Tool dirs  : ${#TOOL_DIR_ARGS[@]} dir(s) queued"
echo ""

# ── Score skills ──────────────────────────────────────────────────────────────
python3 -m thalamus.scoring build \
  --type skills \
  --skills-dir  "$SKILLS_DIR" \
  --project-dir "$PROJECT_DIR" \
  --matrix-dir  "$ORACLE_DIR" \
  --model       "$MODEL" \
  --api-key     "$API_KEY" \
  --api-base    "$API_BASE" \
  --n-examples  "$N_EXAMPLES" \
  --parallel    "$PARALLEL"

# ── Score memory sections ─────────────────────────────────────────────────────
python3 -m thalamus.scoring build \
  --type memory \
  --project-dir "$PROJECT_DIR" \
  --matrix-dir  "$ORACLE_DIR" \
  --model       "$MODEL" \
  --api-key     "$API_KEY" \
  --api-base    "$API_BASE" \
  --n-examples  "$N_EXAMPLES" \
  --parallel    "$PARALLEL"

# ── Score tools ───────────────────────────────────────────────────────────────
if [[ "${#TOOL_DIR_ARGS[@]}" -gt 0 ]]; then
  echo "Scoring tool components (${#TOOL_DIR_ARGS[@]} dir arg(s))..."
  python3 -m thalamus.scoring build \
    --type tools \
    "${TOOL_DIR_ARGS[@]}" \
    --matrix-dir  "$ORACLE_DIR" \
    --model       "$MODEL" \
    --api-key     "$API_KEY" \
    --api-base    "$API_BASE" \
    --n-examples  "$N_EXAMPLES" \
    --parallel    "$PARALLEL"
else
  echo "Tool scoring skipped — no tool directories available."
fi

echo ""
echo "Done. Scoring matrices written to: $ORACLE_DIR"
