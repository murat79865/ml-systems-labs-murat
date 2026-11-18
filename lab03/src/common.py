"""Shared helpers for Lab 3: paths, seeds, Fashion-MNIST loaders, timing, hardware info."""

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
DATA_DIR = REPO_DIR / "data"  # git-ignored; torchvision downloads Fashion-MNIST here
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ("mlp", "cnn", "attention")
MODEL_LABELS = {"mlp": "MLP", "cnn": "CNN", "attention": "Attention"}

# Step 1
MEAN, STD = 0.2860, 0.3530
N_TRAIN, N_VAL = 55_000, 5_000
BATCH_SIZE = 128

# Step 4
EPOCHS = 5
LR = 1e-3

# Step 3 latency protocol: median of 100 runs (batch 1) and 20 runs (batch 128), after warm-up
LATENCY_RUNS = {1: 100, 128: 20}
LATENCY_WARMUP = 3


def set_seeds(seed: int = SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_datasets():
    """Step 1: FashionMNIST, ToTensor + Normalize((0.2860,), (0.3530,)), 55k/5k split of the official
    60k training set (seeded), official 10k test set."""
    import torch
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    full_train = datasets.FashionMNIST(root=str(DATA_DIR), train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root=str(DATA_DIR), train=False, download=True, transform=tf)
    train, val = torch.utils.data.random_split(
        full_train, [N_TRAIN, N_VAL], generator=torch.Generator().manual_seed(SEED))
    return train, val, test


def get_loaders():
    import torch
    from torch.utils.data import DataLoader

    train, val, test = get_datasets()
    g = torch.Generator().manual_seed(SEED)
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=0),
            DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
            DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0))


def test_inputs():
    """A batch of 1 and a batch of 128 real test images (normalised), used for profiling/latency."""
    import torch

    _, _, test = get_datasets()
    x128 = torch.stack([test[i][0] for i in range(128)])
    return x128[:1].clone(), x128


def time_call(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def summarize_ms(times_s: list[float]) -> dict:
    ms = sorted(t * 1000.0 for t in times_s)
    return {"n": len(ms), "median_ms": statistics.median(ms), "min_ms": ms[0], "max_ms": ms[-1],
            "p95_ms": ms[max(0, int(round(0.95 * len(ms))) - 1)]}


def hardware_info() -> dict:
    import torch

    cpu = platform.processor() or ""
    if not cpu and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    info = {"os": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
            "cpu": cpu or "unknown", "logical_cpus": os.cpu_count(),
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "device": "cpu",
            "cuda_available": torch.cuda.is_available()}
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
