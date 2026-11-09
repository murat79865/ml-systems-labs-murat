"""Print the Python version and the installed version of every package in requirements.txt.

Usage (from the repo root):
    python lab01/src/print_versions.py > lab01/results/versions.txt

Each package is compared with its pin; any mismatch is flagged so a non-reproducible
environment is visible immediately.
"""

from __future__ import annotations

import platform
import re
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

from packaging.version import InvalidVersion, Version

REQUIRED_PYTHON = "3.11.8"
REQ_FILE = Path(__file__).resolve().parents[2] / "requirements.txt"


def parse_requirements(path: Path) -> list[tuple[str, str]]:
    pins = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s;]+)", line)
        if m:
            pins.append((m.group(1), m.group(2)))
    return pins


def same_version(installed: str, pinned: str) -> bool:
    try:
        return Version(installed) == Version(pinned)
    except InvalidVersion:
        return installed == pinned


def main() -> int:
    py = platform.python_version()
    print(f"Generated : {datetime.now().isoformat(timespec='seconds')}")
    print(f"Python    : {py} ({platform.python_implementation()}) "
          f"[required {REQUIRED_PYTHON}: {'OK' if py == REQUIRED_PYTHON else 'MISMATCH'}]")
    print(f"Executable: {sys.executable}")
    print(f"Platform  : {platform.platform()}")
    print(f"In venv   : {sys.prefix != getattr(sys, 'base_prefix', sys.prefix)}")
    print()
    print(f"{'package':<22}{'pinned':<26}{'installed':<26}status")
    print("-" * 80)

    problems = 0
    for name, pinned in parse_requirements(REQ_FILE):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = "NOT INSTALLED"
        status = "OK" if same_version(installed, pinned) else "MISMATCH"
        problems += status != "OK"
        print(f"{name:<22}{pinned:<26}{installed:<26}{status}")

    print("-" * 80)
    print(f"Packages checked: {len(parse_requirements(REQ_FILE))}, problems: {problems}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
