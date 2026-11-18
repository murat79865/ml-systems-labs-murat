"""Run the whole of Lab 6 in order (about 30 minutes on a laptop CPU). From the repo root, venv active:

    python lab06/src/run_all.py

versions -> step 1 base run -> step 2 profile -> step 3 num_workers -> step 4 batch size ->
step 5 schedules/optimizers -> step 6 gradient accumulation -> step 7 interrupt/resume ->
MLflow export -> report.md
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
RESULTS = SRC.parent / "results"


def run(*args: str, stdout_file: Path | None = None) -> None:
    print(f"\n=== {' '.join(args)} ===", flush=True)
    cmd = [sys.executable, str(SRC / args[0]), *args[1:]]
    if stdout_file is None:
        subprocess.run(cmd, check=True, cwd=SRC)
    else:
        out = subprocess.run(cmd, check=True, cwd=SRC, capture_output=True, text=True).stdout
        stdout_file.write_text(out, encoding="utf-8")
        print(out)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    run("print_versions.py", stdout_file=RESULTS / "versions.txt")
    run("base_run.py")
    run("profile_epoch.py")
    run("workers_study.py")
    run("studies.py", "batch")
    run("studies.py", "schedules")
    run("studies.py", "accum")
    run("studies.py", "resume")
    run("export_mlflow.py")
    run("make_report.py")
    print("\nDone. Review lab06/report.md, then commit.")


if __name__ == "__main__":
    main()
