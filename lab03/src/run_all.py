"""Run every step of Lab 3 in order. From the repo root, with the venv active:

    python lab03/src/run_all.py

Steps: versions -> parameters & FLOPs (torchinfo, hand, thop) -> training (5 epochs) -> latency ->
trade-off plots -> report.md. Fashion-MNIST is downloaded to data/ on the first run.
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
    run("cost.py")
    run("train.py")
    run("latency.py")
    run("plots.py")
    run("make_report.py")
    print("\nDone. Review lab03/report.md, then commit.")


if __name__ == "__main__":
    main()
