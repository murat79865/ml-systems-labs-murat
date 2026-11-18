"""Shared helpers for Lab 6: paths, YAML config (+ overrides), seeds, data, model, memory sampler."""

from __future__ import annotations

import copy
import json
import os
import platform
import random
import threading
import time
from pathlib import Path

import numpy as np
import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = LAB_DIR.parent
RESULTS_DIR = LAB_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"          # one JSON summary per training run
BASE_CONFIG = LAB_DIR / "configs" / "base.yaml"
for d in (RESULTS_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------------------------- config
def load_config(path: Path = BASE_CONFIG, overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, value = item.split("=", 1)
        set_key(cfg, key, yaml.safe_load(value))
    return cfg


def set_key(cfg: dict, dotted: str, value) -> None:
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"unknown config key: {dotted}")
    node[parts[-1]] = value


def with_overrides(cfg: dict, **kv) -> dict:
    new = copy.deepcopy(cfg)
    for k, v in kv.items():
        set_key(new, k.replace("__", "."), v)
    return new


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def repo_path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_DIR / q


# ---------------------------------------------------------------------------------------------- seeds
def set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------------------------- data / model
def get_datasets(cfg: dict):
    import torch
    from torchvision import datasets, transforms

    d = cfg["data"]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((d["mean"],), (d["std"],))])
    root = str(repo_path(d["root"]))
    full = datasets.FashionMNIST(root=root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root=root, train=False, download=True, transform=tf)
    train, val = torch.utils.data.random_split(full, [d["train_size"], d["val_size"]],
                                               generator=torch.Generator().manual_seed(cfg["seed"]))
    return train, val, test


def make_loaders(cfg: dict, datasets_=None, num_workers: int | None = None, batch_size: int | None = None):
    import torch
    from torch.utils.data import DataLoader

    train, val, test = datasets_ or get_datasets(cfg)
    nw = cfg["data"]["num_workers"] if num_workers is None else num_workers
    bs = cfg["data"]["batch_size"] if batch_size is None else batch_size
    g = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train, batch_size=bs, shuffle=True, generator=g, num_workers=nw)
    val_loader = DataLoader(val, batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, g


def make_model(cfg: dict):
    from torch import nn

    assert cfg["model"]["name"] == "cnn"
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 10),
    )


# ---------------------------------------------------------------------------------------------- memory
class PeakRSS:
    """Samples this process's resident memory every 5 ms in a background thread."""

    def __init__(self, interval: float = 0.005):
        import psutil

        self.proc = psutil.Process()
        self.interval = interval
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


# ---------------------------------------------------------------------------------------------- misc
def hardware_info() -> dict:
    import torch

    cpu = platform.processor() or ""
    if not cpu and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    info = {"os": f"{platform.system()} {platform.release()}", "cpu": cpu or "unknown",
            "logical_cpus": os.cpu_count(), "python": platform.python_version(), "torch": torch.__version__,
            "torch_threads": torch.get_num_threads()}
    try:
        import mlflow

        info["mlflow"] = mlflow.__version__
        import psutil

        info["total_ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        pass
    return info


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
