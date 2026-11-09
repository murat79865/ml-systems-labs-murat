"""Shared helpers for Lab 1: paths, seeds, data split, model definitions, timing."""

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
DATA_DIR = REPO_DIR / "data"  # git-ignored; Lab 1 uses a built-in sklearn dataset
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ("logistic_regression", "random_forest")
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest (100 trees)",
}

# Number of repetitions required by the lab manual.
TRAIN_WARMUP = 1
TRAIN_REPEATS = 5
INFER_WARMUP = 1
INFER_REPEATS = 100


def set_seeds(seed: int = SEED, include_torch: bool = True) -> None:
    """Seed every RNG named in the lab manual.

    include_torch=False is used only inside the memory-measurement processes: importing torch
    adds hundreds of MB of RSS that a scikit-learn model never uses, which would make the
    memory footprint of these models look far larger than it is.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if not include_torch:
        return
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is pinned in requirements.txt; guard only for safety
        pass


def load_split():
    """Breast Cancer Wisconsin, 70/30 stratified split, random_state=42."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)


def make_model(name: str):
    """Return a fresh, unfitted model exactly as specified in the lab manual."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=SEED)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=SEED)
    raise ValueError(f"Unknown model: {name}")


def model_path(name: str) -> Path:
    return RESULTS_DIR / f"model_{name}.joblib"


def time_call(fn, warmup: int, repeats: int) -> list[float]:
    """Run fn() `warmup` times (discarded), then `repeats` times; return seconds per run."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def summarize(times_s: list[float]) -> dict:
    ms = [t * 1000.0 for t in times_s]
    ms_sorted = sorted(ms)
    p95_idx = max(0, int(round(0.95 * len(ms_sorted))) - 1)
    return {
        "n": len(ms),
        "median_ms": statistics.median(ms),
        "min_ms": ms_sorted[0],
        "max_ms": ms_sorted[-1],
        "p95_ms": ms_sorted[p95_idx],
        "stdev_ms": statistics.stdev(ms) if len(ms) > 1 else 0.0,
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
