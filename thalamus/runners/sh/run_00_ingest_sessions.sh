#!/usr/bin/env bash
# Phase 0: Ingest jiuwenswarm session trajectories → TurnLogger format.
#
# Reads agent session histories from SESSIONS_DIR, converts each
# user-request turn into a TurnLogger-compatible JSONL record in LOG_DIR.
# Run this before run_03_classifier.sh to populate online_logs/.
#
# Config (override via environment variables):
#   SESSIONS_DIR    jiuwenswarm sessions root      (default: ~/.jiuwenswarm/agent/sessions)
#   ORACLE_DIR      Oracle directory               (required, for LOG_DIR default)
#   LOG_DIR         Where to write turns_*.jsonl   (default: ORACLE_DIR/online_logs)
#   MAX_FEATURES    TF-IDF vocabulary size         (default: 2000)
#   FORCE           Set to 1 to re-ingest all      (default: 0)

set -euo pipefail

SESSIONS_DIR="${SESSIONS_DIR:-$HOME/.jiuwenswarm/agent/sessions}"
ORACLE_DIR="${ORACLE_DIR:-$HOME/.jiuwenswarm/agent/workspace/oracle}"
LOG_DIR="${LOG_DIR:-$ORACLE_DIR/online_logs}"
MAX_FEATURES="${MAX_FEATURES:-2000}"
FORCE="${FORCE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_RUNNER="$SCRIPT_DIR/../py/run_00_ingest_sessions.py"

export SESSIONS_DIR ORACLE_DIR LOG_DIR MAX_FEATURES FORCE

python3 "$PY_RUNNER"
