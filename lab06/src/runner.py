"""Launch train.py in a fresh process with config overrides (clean peak-memory measurement per run)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from common import BASE_CONFIG, RUNS_DIR, read_json

SRC = Path(__file__).resolve().parent


def train(run_name: str, **overrides) -> dict:
    sets = [f"run_name={run_name}"] + [f"{k.replace('__', '.')}={_fmt(v)}" for k, v in overrides.items()]
    out = RUNS_DIR / f"{run_name}.json"
    cmd = [sys.executable, str(SRC / "train.py"), "--config", str(BASE_CONFIG), "--out", str(out), "--set", *sets]
    print(f"\n--- {run_name}: {' '.join(sets[1:]) or '(base config)'}", flush=True)
    subprocess.run(cmd, check=True, cwd=SRC)
    return read_json(out)


def _fmt(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
