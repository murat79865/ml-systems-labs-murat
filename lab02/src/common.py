"""Shared helpers for Lab 2: paths, seeds, data, timing, hardware info."""

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
REPO_DIR = LAB_DIR.parent
RESULTS_DIR = LAB_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment grid and training settings from the lab manual.
WIDTHS = (16, 64)
LEARNING_RATES = (0.001, 0.01, 0.1)
BATCH_SIZE = 32
EPOCHS = 50
INIT_STD = 0.01  # N(0, 0.01): mean 0, standard deviation 0.01 (np.random.normal(0, 0.01))

# Timing protocol (global rules + step 7).
TIMING_WARMUP = 1
TIMING_REPEATS = 100


def set_seeds(seed: int = SEED) -> None:
    """Seed every RNG named in the course rules."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is pinned in requirements.txt; guard only for safety
        pass


def one_hot(y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def load_data():
    """Step 1: load_digits, X / 16, one-hot labels, 70/30 stratified split (random_state=42).

    Returns X_train, X_test, Y_train, Y_test (one-hot) and y_train, y_test (integer labels).
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    return X_train, X_test, one_hot(y_train), one_hot(y_test), y_train, y_test


def run_name(width: int, lr: float) -> str:
    return f"W{width}_lr{lr:g}"


def time_call(fn, warmup: int = TIMING_WARMUP, repeats: int = TIMING_REPEATS) -> list[float]:
    """Run fn() `warmup` times (discarded), then `repeats` times; return seconds per run."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def summarize_us(times_s: list[float]) -> dict:
    us = sorted(t * 1e6 for t in times_s)
    return {
        "n": len(us),
        "median_us": statistics.median(us),
        "p95_us": us[max(0, int(round(0.95 * len(us))) - 1)],
        "min_us": us[0],
        "max_us": us[-1],
    }


def hardware_info() -> dict:
    cpu = platform.processor() or ""
    if not cpu and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": cpu or "unknown",
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import psutil

        info["total_ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        pass
    return info


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
