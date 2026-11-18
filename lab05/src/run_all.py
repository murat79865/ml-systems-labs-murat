"""Run every step of Lab 5 in order. From the repo root, with the venv active:

    python lab05/src/run_all.py

versions -> verify engine (step 4) -> make_moons training + speed (step 5) -> PyTorch graph, execution
modes and tensor layout (step 6) -> report.md. The engine itself (steps 1-3) is src/engine.py.
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
    run("verify.py")
    run("train_moons.py")
    run("torch_inspect.py")
    run("make_report.py")
    print("\nDone. Review lab05/report.md, then commit.")


if __name__ == "__main__":
    main()
