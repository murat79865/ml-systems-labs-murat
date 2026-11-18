"""Step 3 (latency): inference latency of the trained models for batch 1 and batch 128 on CPU.

Protocol: model.eval(), torch.no_grad(), real normalised test images; 3 warm-up calls (discarded),
then 100 timed calls for batch 1 and 20 timed calls for batch 128; the median is reported (p95, min,
max in the CSV, every run in latency_raw.json). Timed with time.perf_counter(). The default PyTorch
inference path is used (including the fused Transformer-encoder fast path).

Outputs: results/latency.csv, results/latency_raw.json, results/hardware.json
"""

from __future__ import annotations

import csv

import torch

import models
from common import (LATENCY_RUNS, LATENCY_WARMUP, MODEL_NAMES, RESULTS_DIR, hardware_info,
                    set_seeds, summarize_ms, test_inputs, time_call, write_json)


def main() -> None:
    set_seeds()
    write_json(RESULTS_DIR / "hardware.json", hardware_info())
    x1, x128 = test_inputs()
    inputs = {1: x1, 128: x128}
    rows, raw = [], {}
    for name in MODEL_NAMES:
        model = models.build(name)
        model.load_state_dict(torch.load(RESULTS_DIR / f"model_{name}.pt", map_location="cpu", weights_only=True))
        model.eval()
        raw[name] = {}
        for bs, runs in LATENCY_RUNS.items():
            x = inputs[bs]
            with torch.no_grad():
                t = time_call(lambda: model(x), warmup=LATENCY_WARMUP, repeats=runs)
            s = summarize_ms(t)
            raw[name][f"batch_{bs}"] = [v * 1000 for v in t]
            rows.append({"model": name, "batch": bs, "runs": s["n"], "warmup": LATENCY_WARMUP,
                         "median_ms": f"{s['median_ms']:.4f}", "p95_ms": f"{s['p95_ms']:.4f}",
                         "min_ms": f"{s['min_ms']:.4f}", "max_ms": f"{s['max_ms']:.4f}",
                         "median_ms_per_image": f"{s['median_ms'] / bs:.5f}",
                         "images_per_second": f"{bs / (s['median_ms'] / 1000):.0f}"})
            print(f"{name:<10} batch {bs:>3}: median {s['median_ms']:8.3f} ms over {s['n']} runs "
                  f"({s['median_ms'] / bs:.4f} ms/image)")
    write_json(RESULTS_DIR / "latency_raw.json", {"unit": "ms", "runs": raw})
    with open(RESULTS_DIR / "latency.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
