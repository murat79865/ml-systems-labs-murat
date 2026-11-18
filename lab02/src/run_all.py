"""Run every step of Lab 2 in order. From the repo root, with the venv active:

    python lab02/src/run_all.py

Steps: versions -> data/design/forward/loss checks -> training -> gradient check -> timing -> report.md
(the gradient check uses the trained parameters, so it runs after training).
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
    run("check_forward.py")
    run("train.py")
    run("grad_check.py")
    run("time_passes.py")
    run("make_report.py")
    print("\nDone. Review lab02/report.md, then commit.")


if __name__ == "__main__":
    main()
