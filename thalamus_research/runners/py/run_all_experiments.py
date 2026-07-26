#!/usr/bin/env python3
"""
Full THALAMUS research pipeline — R1 through R3b (pre-data phases).

Python equivalent of run_all_experiments.sh — calls each runner's main() directly,
so the entire pipeline is debuggable in one process.

Phases that run immediately (no logged turns required):
  R1  Baseline evaluation       (run_01_baselines.py)
  R2  Ablation study            (run_02_ablation.py)
  R3a Cross-path analysis       (run_03_cross_path.py)
  R3b ε* estimation             (run_04_bandit.py MODE=estimate)

Phases that require logged agent turns (run separately after collecting data):
  R3b Convergence analysis   →  MODE=convergence python run_04_bandit.py
  R4  Set-level quality      →  python run_05_set_quality.py
  R5  Meta-learning          →  python run_06_meta_learning.py

Skip individual phases by setting SKIP_R1=1, SKIP_R2=1, SKIP_R3A=1, SKIP_R3B=1.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_01_baselines   # noqa: E402
import run_02_ablation    # noqa: E402
import run_03_cross_path  # noqa: E402
import run_04_bandit      # noqa: E402
from _config import cfg   # noqa: E402


def main() -> None:
    skip_r1  = os.environ.get("SKIP_R1",  "0") == "1"
    skip_r2  = os.environ.get("SKIP_R2",  "0") == "1"
    skip_r3a = os.environ.get("SKIP_R3A", "0") == "1"
    skip_r3b = os.environ.get("SKIP_R3B", "0") == "1"

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   THALAMUS Research Pipeline — R1 through R3b        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"Oracle:      {cfg.oracle_dir}")
    print(f"Results dir: {cfg.results_dir}")
    print()
    print("Phases requiring turn logs (run separately after data collection):")
    print("  R3b convergence → MODE=convergence python run_04_bandit.py")
    print("  R4 set quality  → python run_05_set_quality.py")
    print("  R5 meta-learning → python run_06_meta_learning.py")
    print()

    # R1 — Baseline evaluation
    if not skip_r1:
        run_01_baselines.main()
    else:
        print("[SKIP] R1 baselines")

    print()

    # R2 — Ablation study
    if not skip_r2:
        run_02_ablation.main()
    else:
        print("[SKIP] R2 ablation")

    print()

    # R3a — Cross-path analysis
    if not skip_r3a:
        run_03_cross_path.main()
    else:
        print("[SKIP] R3a cross-path")

    print()

    # R3b — Bandit exploration rate (estimate mode only — no logs needed)
    if not skip_r3b:
        os.environ["MODE"] = "estimate"
        run_04_bandit.main()
    else:
        print("[SKIP] R3b bandit")

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Pre-data phases complete.                          ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Results in: {cfg.results_dir}")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Next steps (after collecting agent turn logs):      ║")
    print("║    1. Set exploration rate from R3b epsilon report   ║")
    print("║    2. MODE=convergence python run_04_bandit.py       ║")
    print("║    3. python run_05_set_quality.py  (needs ~20+ turns)║")
    print("║    4. python run_06_meta_learning.py  (multi-deploy) ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
