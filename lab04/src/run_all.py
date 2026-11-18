"""One command for the whole Lab 4 pipeline. From the repo root, with the venv active:

    python lab04/src/run_all.py

ingest -> validate -> split/clean/transform (+ hash) -> derived-feature evaluation -> label audit ->
report.md + data_card.md. All parameters come from lab04/config.yaml.
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
    run("ingest.py")
    run("validate.py")
    run("clean_transform.py")
    run("features_eval.py")
    run("label_audit.py")
    run("make_report.py")
    print("\nDone. Review lab04/report.md and lab04/data_card.md, then commit.")


if __name__ == "__main__":
    main()
