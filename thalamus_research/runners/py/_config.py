"""
Shared configuration for thalamus_research Python runners.

Import this module at the top of each runner:

    from _config import cfg, require_var, section

Override any value by exporting the corresponding environment variable before
running. Same variable names as _config.sh.
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Required ──────────────────────────────────────────────────────────────
    # Produced by `thalamus-oracle evolve` — context_configs.json, context_configs.pkl,
    # and (after training) classifier_current.pkl.
    oracle_dir: str = field(
        default_factory=lambda: os.environ.get(
            "ORACLE_DIR", "~/.jiuwenswarm/agent/workspace/oracle"
        )
    )

    # ── Results output ────────────────────────────────────────────────────────
    results_dir: str = ""  # resolved below

    # ── Queries ───────────────────────────────────────────────────────────────
    # JSON file with [{"id": str, "query": str}] objects, or plain-text with
    # one query per line. Empty → each runner uses its own inline defaults.
    queries_file: str = field(
        default_factory=lambda: os.environ.get("QUERIES_FILE", "")
    )

    # ── Turn logs ─────────────────────────────────────────────────────────────
    # Directory containing turns_YYYY-WNN.jsonl. Required for R3b, R4, R5.
    turn_log_dir: str = ""  # resolved below

    # ── R3a cross-path ────────────────────────────────────────────────────────
    co_inclusion_lam: str = field(
        default_factory=lambda: os.environ.get("CO_INCLUSION_LAM", "0.2")
    )
    top_pairs: str = field(
        default_factory=lambda: os.environ.get("TOP_PAIRS", "20")
    )

    # ── R3b bandit ────────────────────────────────────────────────────────────
    bandit_n_min:    str = field(default_factory=lambda: os.environ.get("BANDIT_N_MIN",    "10"))
    bandit_t_target: str = field(default_factory=lambda: os.environ.get("BANDIT_T_TARGET", "500"))
    bandit_window:   str = field(default_factory=lambda: os.environ.get("BANDIT_WINDOW",   "50"))
    bandit_budget:   str = field(default_factory=lambda: os.environ.get("BANDIT_BUDGET",   "medium"))

    # ── R5 meta-learning ──────────────────────────────────────────────────────
    kb_path: str = field(
        default_factory=lambda: os.environ.get(
            "KB_PATH", os.path.expanduser("~/.jiuwenswarm/thalamus_knowledge_base.json")
        )
    )
    new_oracle_dir: str = field(
        default_factory=lambda: os.environ.get("NEW_ORACLE_DIR", "")
    )

    def __post_init__(self) -> None:
        if not self.results_dir:
            self.results_dir = os.environ.get(
                "RESULTS_DIR", os.path.join(self.oracle_dir, "research_results")
            )
        if not self.turn_log_dir:
            self.turn_log_dir = os.environ.get(
                "TURN_LOG_DIR", os.path.join(self.oracle_dir, "online_logs")
            )
        # Ensure results directory exists
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)


# Module-level singleton — importers get a fresh Config from env vars
cfg = Config()


# ── Helpers ───────────────────────────────────────────────────────────────────

def require_var(value: str, env_name: str, label: str) -> None:
    """Exit with an informative error if a required variable is empty."""
    if not value:
        print(f"ERROR: {label} is not set. Export {env_name} before running.", file=sys.stderr)
        sys.exit(1)


def section(title: str) -> None:
    """Print a section header matching the shell _section() helper."""
    print()
    print("══════════════════════════════════════════════════════")
    print(f"  {title}")
    print("══════════════════════════════════════════════════════")
    print()
