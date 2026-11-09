"""Step 3 (time + size): training time, single-sample inference latency and model size.

Protocol (lab manual, Global rules):
  * Training time: 1 warm-up fit (discarded), then 5 timed fits on a fresh model; report median.
  * Single-sample inference: 1 warm-up call, then 100 timed predict() calls on one test row; median.
  * Model size: joblib.dump of the fitted model; bytes and KB (1 KB = 1024 bytes).

Peak memory is measured separately in fresh processes by measure_memory.py so that the
numbers of one model are not polluted by the other.

Outputs: results/model_<name>.joblib, results/model.joblib (both models in one dict),
         results/timing_raw.json, results/cost_time_size.csv, results/hardware.json
"""

from __future__ import annotations

import csv
import warnings

import joblib
from sklearn.exceptions import ConvergenceWarning

from common import (INFER_REPEATS, INFER_WARMUP, MODEL_NAMES, RESULTS_DIR, TRAIN_REPEATS,
                    TRAIN_WARMUP, hardware_info, load_split, make_model, model_path, set_seeds,
                    summarize, time_call, write_json)


def main() -> None:
    set_seeds()
    X_train, X_test, y_train, _ = load_split()
    x_one = X_test[:1]  # one sample, kept 2-D as predict() expects

    write_json(RESULTS_DIR / "hardware.json", hardware_info())

    raw, rows, fitted = {}, [], {}
    for name in MODEL_NAMES:
        # --- training time -------------------------------------------------------
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            train_times = time_call(lambda: make_model(name).fit(X_train, y_train),
                                    warmup=TRAIN_WARMUP, repeats=TRAIN_REPEATS)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

        # --- final model used for inference + size -------------------------------
        set_seeds()
        model = make_model(name).fit(X_train, y_train)
        fitted[name] = model

        infer_times = time_call(lambda: model.predict(x_one),
                                warmup=INFER_WARMUP, repeats=INFER_REPEATS)

        path = model_path(name)
        joblib.dump(model, path)
        size_bytes = path.stat().st_size

        train_s, infer_s = summarize(train_times), summarize(infer_times)
        raw[name] = {
            "train_times_s": train_times,
            "infer_times_s": infer_times,
            "train_summary": train_s,
            "infer_summary": infer_s,
            "size_bytes": size_bytes,
            "converged": converged,
            "n_iter": int(model.n_iter_[0]) if hasattr(model, "n_iter_") else None,
        }
        rows.append({
            "model": name,
            "train_time_median_ms": f"{train_s['median_ms']:.3f}",
            "train_runs": train_s["n"],
            "inference_latency_median_ms": f"{infer_s['median_ms']:.4f}",
            "inference_latency_p95_ms": f"{infer_s['p95_ms']:.4f}",
            "inference_runs": infer_s["n"],
            "model_size_bytes": size_bytes,
            "model_size_kb": f"{size_bytes / 1024:.2f}",
        })
        print(f"{name:<22} train median {train_s['median_ms']:9.2f} ms | "
              f"inference median {infer_s['median_ms']:7.4f} ms | size {size_bytes / 1024:9.2f} KB")

    # The manual names results/model.joblib: store both fitted models there as one dict.
    joblib.dump(fitted, RESULTS_DIR / "model.joblib")

    write_json(RESULTS_DIR / "timing_raw.json", raw)
    with open(RESULTS_DIR / "cost_time_size.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
