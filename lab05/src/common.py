"""Shared helpers for Lab 5: paths, seeds, timing, hardware info, json."""

from __future__ import annotations

import json
import os
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np

SEED = 42
LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# step 5
N_SAMPLES, NOISE = 200, 0.1
HIDDEN, LR, EPOCHS = 8, 2.0, 100
SPEED_RUNS = 5


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def time_call(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def median_ms(times_s) -> float:
    return statistics.median(times_s) * 1000.0


def hardware_info() -> dict:
    cpu = platform.processor() or ""
    if not cpu and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    info = {"os": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
            "cpu": cpu or "unknown", "logical_cpus": os.cpu_count(), "python": platform.python_version()}
    try:
        import torch

        info.update({"torch": torch.__version__, "torch_threads": torch.get_num_threads()})
    except ImportError:
        pass
    return info


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
