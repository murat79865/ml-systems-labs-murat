"""Step 3 (memory): peak resident memory (RSS) during training and during inference.

Each (model, phase, tool) is measured in a FRESH Python process, so one measurement never
inherits memory from another. Two independent tools are used, as the manual allows:

  * memory_profiler.memory_usage(..., max_usage=True) - samples the process every 1 ms.
  * psutil.Process().memory_info().rss - sampled every 0.5 ms from a background thread, then
    on Linux combined with the kernel high-water mark (VmHWM, reset right before the work) so
    that short spikes between samples are not missed.

For every measurement we record:
  baseline_mb  RSS after imports and data loading, just before the measured work
  peak_mb      highest RSS while the work runs  -> the real footprint of the process
  delta_mb     peak - baseline                   -> memory added by the model work itself
1 MB here = 1024 * 1024 bytes.

Training phase : one fit() of a fresh model.
Inference phase: joblib.load() of the saved model + 1 warm-up + 100 single-sample predict() calls
                 (what a serving process would actually do).

Run without arguments to measure everything and write results/memory.csv and memory_raw.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

MB = 1024 * 1024
SRC_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- worker side
def _reset_os_peak() -> bool:
    """Linux only: reset the kernel's RSS high-water mark (VmHWM) so it covers just the work."""
    try:
        Path("/proc/self/clear_refs").write_text("5")
        return True
    except OSError:
        return False


def _os_peak_rss_bytes() -> int:
    """Linux VmHWM since the last reset (0 elsewhere: Windows/macOS peaks cannot be reset,
    so there the 0.5 ms sampler alone is used to avoid counting import-time peaks)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _measure_psutil(work) -> tuple[float, float]:
    import psutil

    proc = psutil.Process()
    hwm_reset = _reset_os_peak()
    baseline = proc.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.0005)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    work()
    stop.set()
    thread.join()
    peak_bytes = max(peak[0], proc.memory_info().rss, _os_peak_rss_bytes() if hwm_reset else 0)
    return baseline / MB, peak_bytes / MB


def _measure_memory_profiler(work) -> tuple[float, float]:
    import psutil
    from memory_profiler import memory_usage

    baseline = psutil.Process().memory_info().rss / MB
    peak = memory_usage((work, (), {}), interval=0.001, max_usage=True, include_children=False)
    if isinstance(peak, (list, tuple)):  # older memory_profiler versions return a list
        peak = max(peak)
    return baseline, float(peak)


def worker(model_name: str, phase: str, tool: str) -> dict:
    sys.path.insert(0, str(SRC_DIR))
    import joblib

    from common import INFER_REPEATS, INFER_WARMUP, load_split, make_model, model_path, set_seeds

    set_seeds(include_torch=False)  # torch is not used here and would inflate RSS (see common.py)
    X_train, X_test, y_train, _ = load_split()
    x_one = X_test[:1]

    if phase == "train":
        def work():
            make_model(model_name).fit(X_train, y_train)
    else:
        def work():
            model = joblib.load(model_path(model_name))
            for _ in range(INFER_WARMUP + INFER_REPEATS):
                model.predict(x_one)

    measure = _measure_memory_profiler if tool == "memory_profiler" else _measure_psutil
    baseline, peak = measure(work)
    return {"model": model_name, "phase": phase, "tool": tool,
            "baseline_mb": baseline, "peak_mb": peak, "delta_mb": max(0.0, peak - baseline)}


# ----------------------------------------------------------------------------- driver side
def run_all() -> None:
    sys.path.insert(0, str(SRC_DIR))
    from common import MODEL_NAMES, RESULTS_DIR, write_json

    records = []
    for model_name in MODEL_NAMES:
        for phase in ("train", "inference"):
            for tool in ("memory_profiler", "psutil"):
                cmd = [sys.executable, str(Path(__file__).resolve()),
                       "--model", model_name, "--phase", phase, "--tool", tool]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    print(f"[warn] {model_name}/{phase}/{tool} failed:\n{proc.stderr.strip()[-600:]}")
                    continue
                rec = json.loads(proc.stdout.strip().splitlines()[-1])
                records.append(rec)
                print(f"{model_name:<22}{phase:<10}{tool:<16}"
                      f"baseline {rec['baseline_mb']:7.1f} MB  peak {rec['peak_mb']:7.1f} MB  "
                      f"delta {rec['delta_mb']:6.2f} MB")

    write_json(RESULTS_DIR / "memory_raw.json", records)
    with open(RESULTS_DIR / "memory.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "phase", "tool", "baseline_mb", "peak_mb", "delta_mb"])
        writer.writeheader()
        for r in records:
            writer.writerow({**r, **{k: f"{r[k]:.2f}" for k in ("baseline_mb", "peak_mb", "delta_mb")}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model")
    parser.add_argument("--phase", choices=["train", "inference"])
    parser.add_argument("--tool", choices=["memory_profiler", "psutil"], default="psutil")
    args = parser.parse_args()
    if args.model and args.phase:
        print(json.dumps(worker(args.model, args.phase, args.tool)))
    else:
        run_all()


if __name__ == "__main__":
    main()
