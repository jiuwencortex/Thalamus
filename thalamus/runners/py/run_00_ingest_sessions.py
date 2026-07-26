#!/usr/bin/env python3
"""
Phase 0: Ingest jiuwenswarm session trajectories → TurnLogger format.

Reads agent session histories from SESSIONS_DIR, converts each user-request
turn into a TurnLogger-compatible JSONL record in LOG_DIR, and writes a
state file so re-runs only process new sessions.

Config via environment variables:
  SESSIONS_DIR    jiuwenswarm sessions root      (default: ~/.jiuwenswarm/agent/sessions)
  ORACLE_DIR      Oracle directory               (required, for LOG_DIR default)
  LOG_DIR         Where to write turns_*.jsonl   (default: ORACLE_DIR/online_logs)
  MAX_FEATURES    TF-IDF vocabulary size         (default: 2000)
  FORCE           Set to "1" to re-ingest all    (default: 0)

Output: turns_YYYY-WNN.jsonl files compatible with TurnLogger.load_turns().
Run this before run_03_classifier.py.
"""
from __future__ import annotations

import json
import os
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path


# ── session parsing ───────────────────────────────────────────────────────────

def _parse_session(history_path: Path) -> list[dict]:
    """Parse one history.jsonl → list of turn dicts, one per user request_id."""
    events: list[dict] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Group events by request_id
    by_request: dict[str, list[dict]] = {}
    request_order: list[str] = []
    for ev in events:
        rid = ev.get("request_id", "")
        if not rid:
            continue
        if rid not in by_request:
            by_request[rid] = []
            request_order.append(rid)
        by_request[rid].append(ev)

    # Total user-request count in session (for conversation_length)
    n_requests = len(request_order)

    turns: list[dict] = []
    for rid in request_order:
        evs = by_request[rid]

        # User query text
        user_text = ""
        for ev in evs:
            if ev.get("role") == "user" and not ev.get("event_type"):
                content = ev.get("content", "")
                if isinstance(content, str) and content.strip():
                    user_text = content.strip()
                    break

        if not user_text:
            continue  # skip assistant-only requests (no user message)

        # Timestamp: earliest event in this request
        timestamps = [ev["timestamp"] for ev in evs if isinstance(ev.get("timestamp"), (int, float))]
        ts = min(timestamps) if timestamps else datetime.now(timezone.utc).timestamp()

        # Skills accessed via skill_tool calls
        skills_in_context: list[str] = []
        skills_used: list[str] = []
        tools_called: list[str] = []

        tool_call_ids_with_result: set[str] = set()
        for ev in evs:
            if ev.get("event_type") == "chat.tool_result":
                tc_id = ev.get("tool_call_id") or ev.get("id", "")
                if tc_id:
                    tool_call_ids_with_result.add(tc_id)

        for ev in evs:
            if ev.get("event_type") != "chat.tool_call":
                continue
            tc = ev.get("tool_call", {})
            name = tc.get("name", "")
            args_raw = tc.get("arguments", "{}")
            tc_id = tc.get("tool_call_id", "")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}

            if name == "skill_tool":
                skill_name = args.get("skill_name", "")
                if skill_name:
                    skills_in_context.append(skill_name)
                    # Consider skill "used" if its tool_call got a result
                    if tc_id in tool_call_ids_with_result:
                        skills_used.append(skill_name)
            elif name:
                tools_called.append(name)

        # Task completed: leader produced a final response
        task_completed = any(
            ev.get("role") == "leader" and ev.get("event_type") == "chat.final"
            for ev in evs
        )

        turns.append({
            "request_id": rid,
            "timestamp": ts,
            "query_text": user_text,
            "skills_in_context": list(dict.fromkeys(skills_in_context)),   # deduped, ordered
            "skills_used": list(dict.fromkeys(skills_used)),
            "tools_called": list(dict.fromkeys(tools_called)),
            "task_completed": task_completed,
            "conversation_length": n_requests,
        })

    return turns


# ── TF-IDF embedding ──────────────────────────────────────────────────────────

def _build_vectorizer(query_texts: list[str], max_features: int):
    """Fit a TF-IDF vectorizer on the given texts; return vectorizer."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    vec.fit(query_texts)
    return vec


def _load_vectorizer(vec_path: Path):
    with open(vec_path, "rb") as f:
        return pickle.load(f)


def _save_vectorizer(vec, vec_path: Path) -> None:
    with open(vec_path, "wb") as f:
        pickle.dump(vec, f)


# ── JSONL writing ─────────────────────────────────────────────────────────────

def _week_tag(ts: float) -> str:
    """Return ISO-week tag 'YYYY-WNN' for a Unix timestamp."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-W%W")


def _ts_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_oracle_clusterer(oracle_dir: Path):
    """Load QueryClusterer from oracle if available; return None otherwise."""
    pkl = oracle_dir / "context_configs.pkl"
    if not pkl.exists():
        return None
    try:
        from thalamus._shared.query_clusterer import QueryClusterer
        return QueryClusterer.load(pkl)
    except Exception:
        return None


def _write_turns(turns: list[dict], embeddings, log_dir: Path, clusterer=None) -> int:
    """Write turn records to weekly-rotated JSONL files; return count written."""
    log_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, turn in enumerate(turns):
        week = _week_tag(turn["timestamp"])
        dest = log_dir / f"turns_{week}.jsonl"
        # Flat component_set for OutcomeDataset (R4)
        component_set = list(dict.fromkeys(turn["skills_in_context"]))

        # Pre-compute outcome_quality for OutcomeDataset (R4).
        # Mirrors compute_outcome_quality() from thalamus._shared.outcome_scorer:
        #   base 0.5 + 0.2 if task_completed, - 0.3 if follow_up_correction,
        #   + max(0, 0.1 - 0.02 * conversation_length), clamped to [0, 1].
        _q = 0.5
        if turn["task_completed"]:
            _q += 0.2
        _q += max(0.0, 0.1 - 0.02 * turn["conversation_length"])
        outcome_quality = max(0.0, min(1.0, _q))

        # Cluster ID: assign using the oracle's QueryClusterer while we still have
        # the raw query text. This avoids the feature-space mismatch that occurs
        # when OutcomeDataset tries to re-embed a stored TF-IDF vector using the
        # oracle's different TF-IDF vocabulary.
        cluster_id: int | None = None
        if clusterer is not None:
            try:
                cluster_id = clusterer.predict(turn["query_text"])
            except Exception:
                cluster_id = None

        record = {
            "turn_id": str(uuid.uuid4()),
            "timestamp": _ts_iso(turn["timestamp"]),
            "query_embedding": embeddings[i].tolist(),
            # Flat fields consumed by OutcomeDataset (run_05_set_quality)
            "component_set": component_set,
            "outcome_quality": outcome_quality,
            "cluster_id": cluster_id,  # None if oracle not yet built
            # Nested fields consumed by ComponentClassifierTrainer (run_03_classifier)
            "context_config": {
                "skills":          turn["skills_in_context"],
                "memory_sections": [],
                "tools":           [],   # tools are agent internals, not oracle-managed
            },
            "outcome": {
                "explicit_rating": None,
                "implicit_signals": {
                    "follow_up_correction": False,
                    "task_completed": turn["task_completed"],
                    "conversation_length": turn["conversation_length"],
                },
                "component_usage": {
                    "skills_used": turn["skills_used"],
                    "tools_called": turn["tools_called"],
                },
            },
            "_ingested_from": turn["request_id"],   # provenance (non-standard field, ignored by trainer)
        }
        with open(dest, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
    return written


# ── state ─────────────────────────────────────────────────────────────────────

def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"processed": {}}   # {session_id: message_count}


def _save_state(state: dict, state_path: Path) -> None:
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sessions_dir = Path(os.path.expanduser(
        os.environ.get("SESSIONS_DIR", "~/.jiuwenswarm/agent/sessions")
    ))
    oracle_dir = Path(os.path.expanduser(
        os.environ.get("ORACLE_DIR", "~/.jiuwenswarm/agent/workspace/oracle")
    ))
    log_dir = Path(os.path.expanduser(
        os.environ.get("LOG_DIR", str(oracle_dir / "online_logs"))
    ))
    max_features = int(os.environ.get("MAX_FEATURES", "2000"))
    force = os.environ.get("FORCE", "0").strip() == "1"

    print("=== Phase 0: Ingest jiuwenswarm Sessions ===")
    print(f"  Sessions dir : {sessions_dir}")
    print(f"  Log dir      : {log_dir}")
    print()

    if not sessions_dir.exists():
        print(f"  ERROR: sessions directory not found: {sessions_dir}", flush=True)
        return

    state_path = log_dir / "ingestion_state.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(state_path)
    if force:
        state = {"processed": {}}

    # Discover session directories
    session_dirs = sorted(
        d for d in sessions_dir.iterdir()
        if d.is_dir() and d.name.startswith("sess_")
    )
    print(f"  Found {len(session_dirs)} session directory(ies).")

    # Determine which sessions have new data
    to_ingest: list[tuple[str, Path, int]] = []   # (session_id, history_path, message_count)
    for sdir in session_dirs:
        history = sdir / "history.jsonl"
        if not history.exists():
            continue
        try:
            n_lines = sum(1 for ln in history.read_text(encoding="utf-8").splitlines() if ln.strip())
        except OSError:
            continue
        prev = state["processed"].get(sdir.name, 0)
        if n_lines > prev:
            to_ingest.append((sdir.name, history, n_lines))

    if not to_ingest:
        print("  All sessions already ingested. Nothing to do.")
        print(f"  (Use FORCE=1 to re-ingest everything.)")
        return

    print(f"  Ingesting {len(to_ingest)} new/updated session(s)...")
    print()

    # Parse all sessions and collect turns
    all_turns: list[dict] = []
    session_turn_counts: dict[str, int] = {}
    for session_id, history_path, n_lines in to_ingest:
        turns = _parse_session(history_path)
        n = len(turns)
        session_turn_counts[session_id] = n
        all_turns.extend(turns)
        print(f"  {session_id}: {n} turn(s) extracted from {n_lines} events")

    print()

    if not all_turns:
        print("  No usable turns found (no user query text in any session).")
        # Still mark as processed so we don't retry empty sessions
        for session_id, _, n_lines in to_ingest:
            state["processed"][session_id] = n_lines
        _save_state(state, state_path)
        return

    # Build or update TF-IDF vectorizer
    vec_path = log_dir / "ingestion_vectorizer.pkl"
    query_texts = [t["query_text"] for t in all_turns]

    if vec_path.exists() and not force:
        print(f"  Loading existing vectorizer from {vec_path.name}...")
        vec = _load_vectorizer(vec_path)
        # Re-fit to include new vocabulary (safe: same params, larger corpus)
        # Load existing queries to ensure consistent vocabulary if possible
        # For simplicity, re-fit on current corpus only (classifier re-trains each time)
        existing_queries = _collect_existing_queries(log_dir)
        all_fit_texts = existing_queries + query_texts
        vec = _build_vectorizer(all_fit_texts, max_features)
    else:
        print(f"  Fitting TF-IDF vectorizer on {len(query_texts)} query text(s)...")
        vec = _build_vectorizer(query_texts, max_features)

    _save_vectorizer(vec, vec_path)

    # Load oracle clusterer for cluster_id assignment (if oracle already built)
    clusterer = _load_oracle_clusterer(oracle_dir)
    if clusterer is not None:
        print(f"  Oracle clusterer loaded ({clusterer.n_clusters} clusters) — cluster_id will be stored.")
    else:
        print("  Oracle not yet built — cluster_id will be null (run run_02_oracle first).")

    # Embed and write
    import numpy as np
    X = vec.transform(query_texts)
    embeddings = np.asarray(X.todense(), dtype=np.float32)

    written = _write_turns(all_turns, embeddings, log_dir, clusterer=clusterer)

    print(f"  Wrote {written} turn record(s) to {log_dir}")
    print(f"  Vectorizer saved to {vec_path.name}")
    print()

    # Update state
    for session_id, _, n_lines in to_ingest:
        state["processed"][session_id] = n_lines
    _save_state(state, state_path)

    total = sum(1 for f in log_dir.glob("turns_*.jsonl")
                for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"  Total logged turns in {log_dir}: {total}")
    print()
    if total >= 10:
        print("  Classifier training is now possible (>= 10 turns).")
        print("  Run run_03_classifier.py to train.")
    else:
        print(f"  Need at least 10 turns to train classifier (have {total}).")
        print("  Use more agent sessions or wait for live interaction data.")


def _collect_existing_queries(log_dir: Path) -> list[str]:
    """Read existing turn JSONL files and extract ingested query texts (provenance field)."""
    # We don't store raw query text in TurnLogger records (privacy).
    # We can't recover them from existing logs. Just return empty.
    # The vectorizer will be re-fitted on current batch only.
    return []


if __name__ == "__main__":
    main()
