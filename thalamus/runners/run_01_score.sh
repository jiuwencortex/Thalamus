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

set -euo pipefail

# ── Configure these paths ────────────────────────────────────────────────────
SKILLS_DIR="${SKILLS_DIR:-~/.jiuwenswarm/agent/workspace/skills}"
PROJECT_DIR="${PROJECT_DIR:-~/.jiuwenswarm/agent/workspace}"
TOOLS_DIR="${TOOLS_DIR:-~/.jiuwenswarm/agent/workspace/tools}"
ORACLE_DIR="${ORACLE_DIR:-~/.jiuwenswarm/agent/workspace/oracle}"

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

echo "=== Phase 1-2: Component Scoring ==="
echo "  Skills dir : $SKILLS_DIR"
echo "  Project dir: $PROJECT_DIR"
echo "  Tools dir  : $TOOLS_DIR"
echo "  Oracle dir : $ORACLE_DIR"
echo "  Model      : $MODEL"
echo ""

# Score skills and memory sections (always)
python -m thalamus.scoring build \
  --type skills \
  --skills-dir  "$SKILLS_DIR" \
  --project-dir "$PROJECT_DIR" \
  --matrix-dir  "$ORACLE_DIR" \
  --model       "$MODEL" \
  --api-key     "$API_KEY" \
  --api-base    "$API_BASE" \
  --n-examples  "$N_EXAMPLES" \
  --parallel    "$PARALLEL"

python -m thalamus.scoring build \
  --type memory \
  --project-dir "$PROJECT_DIR" \
  --matrix-dir  "$ORACLE_DIR" \
  --model       "$MODEL" \
  --api-key     "$API_KEY" \
  --api-base    "$API_BASE" \
  --n-examples  "$N_EXAMPLES" \
  --parallel    "$PARALLEL"

# Score tools only if TOOLS_DIR exists and contains Python files
if [[ -d "$TOOLS_DIR" ]] && compgen -G "$TOOLS_DIR"/**/*.py > /dev/null 2>&1; then
  echo ""
  echo "Tools dir found — scoring tool components..."
  python -m thalamus.scoring build \
    --type tools \
    --tools-dir   "$TOOLS_DIR" \
    --matrix-dir  "$ORACLE_DIR" \
    --model       "$MODEL" \
    --api-key     "$API_KEY" \
    --api-base    "$API_BASE" \
    --n-examples  "$N_EXAMPLES" \
    --parallel    "$PARALLEL"
else
  echo ""
  echo "NOTE: No tools dir found at $TOOLS_DIR — skipping tool scoring."
  echo "      To score tools, create $TOOLS_DIR with Python files containing *Tool classes,"
  echo "      then re-run this script."
fi

echo ""
echo "Done. Scoring matrices written to: $ORACLE_DIR"
