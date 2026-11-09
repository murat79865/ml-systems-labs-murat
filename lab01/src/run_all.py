"""Run every step of Lab 1 in order. From the repo root, with the venv active:

    python lab01/src/run_all.py

Steps: versions -> baselines -> time/size -> memory -> budget fit + plot -> report.md
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
RESULTS = SRC.parent / "results"


def run(script: str, stdout_file: Path | None = None) -> None:
    print(f"\n=== {script} ===", flush=True)
    cmd = [sys.executable, str(SRC / script)]
    if stdout_file is None:
        subprocess.run(cmd, check=True, cwd=SRC)
    else:
        out = subprocess.run(cmd, check=True, cwd=SRC, capture_output=True, text=True).stdout
        stdout_file.write_text(out, encoding="utf-8")
        print(out)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    run("print_versions.py", RESULTS / "versions.txt")
    run("train_baselines.py")
    run("measure_cost.py")
    run("measure_memory.py")
    run("budget_fit.py")
    run("make_report.py")
    print("\nDone. Review lab01/report.md, then commit.")


if __name__ == "__main__":
    main()
