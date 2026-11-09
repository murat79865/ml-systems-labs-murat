"""Step 4: combine measurements and decide where each model can be deployed.

Budgets (exact values from the lab manual):
  Cloud : memory >= 1 GB       latency <= 100 ms   model size <= 500 MB
  Edge  : memory 256-1024 MB   latency <=  50 ms   model size <=  50 MB
  Mobile: memory 64-256 MB     latency <=  20 ms   model size <=  10 MB
  TinyML: memory <= 256 KB     latency <=  10 ms   model size <= 100 KB

How the memory budget is read: it describes how much RAM a device in that tier has.
  * Cloud : the process must fit in 1 GB (the guaranteed minimum).
  * Edge / Mobile: if peak <= lower bound, it fits on EVERY device of the tier ("Yes");
                   if lower < peak <= upper bound, it fits only on the larger devices ("Partly");
                   above the upper bound it does not fit ("No").
  * TinyML: peak must be <= 256 KB.
The memory used for the decision is the peak RSS of the inference process (a deployed model
serves predictions; it is not retrained on the device). The larger of the two tools' readings
is used, so the decision is conservative.

Outputs: results/measurements.csv, results/deployment_fit.csv, results/model_footprint.csv,
         results/system_cost.png
"""

from __future__ import annotations

import csv
import statistics

import joblib

from common import MODEL_LABELS, MODEL_NAMES, RESULTS_DIR, model_path, read_json

KB, MB = 1024, 1024 * 1024

BUDGETS = {
    #         mem_min_mb  mem_max_mb      latency_ms  size_bytes
    "Cloud":  (1024.0,    None,           100.0,      500 * MB),
    "Edge":   (256.0,     1024.0,         50.0,       50 * MB),
    "Mobile": (64.0,      256.0,          20.0,       10 * MB),
    "TinyML": (None,      256 / 1024,     10.0,       100 * KB),
}


def memory_verdict(tier: str, peak_mb: float) -> str:
    lo, hi, _, _ = BUDGETS[tier]
    if tier == "Cloud":
        return "Yes" if peak_mb <= lo else "No"
    if tier == "TinyML":
        return "Yes" if peak_mb <= hi else "No"
    if peak_mb <= lo:
        return "Yes"
    if peak_mb <= hi:
        return "Partly"
    return "No"


def load_measurements() -> dict:
    timing = read_json(RESULTS_DIR / "timing_raw.json")
    memory = read_json(RESULTS_DIR / "memory_raw.json")
    acc = {r["model"]: r["test_accuracy"] for r in csv.DictReader(open(RESULTS_DIR / "baseline_accuracy.csv"))}

    out = {}
    for name in MODEL_NAMES:
        mem = {(m["phase"], m["tool"]): m for m in memory if m["model"] == name}

        def worst(phase, key):
            vals = [v[key] for (p, _), v in mem.items() if p == phase]
            return max(vals) if vals else float("nan")

        t = timing[name]
        out[name] = {
            "accuracy": acc[name],
            "train_ms": t["train_summary"]["median_ms"],
            "infer_ms": t["infer_summary"]["median_ms"],
            "infer_p95_ms": t["infer_summary"]["p95_ms"],
            "size_bytes": t["size_bytes"],
            "train_peak_mb": worst("train", "peak_mb"),
            "train_delta_mb": worst("train", "delta_mb"),
            "infer_peak_mb": worst("inference", "peak_mb"),
            "infer_delta_mb": worst("inference", "delta_mb"),
            "converged": t["converged"],
        }
    return out


def parameter_footprint(name: str) -> dict:
    """Bytes needed for the learned parameters alone if the model were exported to C (float32)."""
    model = joblib.load(model_path(name))
    if name == "logistic_regression":
        n_params = int(model.coef_.size + model.intercept_.size)
        return {"model": name, "structure": f"{model.coef_.shape[1]} weights + 1 bias",
                "parameters": n_params, "float32_bytes": n_params * 4}
    nodes = sum(est.tree_.node_count for est in model.estimators_)
    leaves = sum(est.tree_.n_leaves for est in model.estimators_)
    depth = statistics.median(est.tree_.max_depth for est in model.estimators_)
    # compact node: feature (int16) + threshold (float32) + left/right child (int16 each) = 10 B;
    # leaves additionally store one float32 class score.
    est_bytes = nodes * 10 + leaves * 4
    return {"model": name, "structure": f"{len(model.estimators_)} trees, {nodes} nodes, "
                                        f"{leaves} leaves, median depth {depth:g}",
            "parameters": nodes, "float32_bytes": est_bytes}


def plot(meas: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, AXIS, SURFACE = "#1f1f1d", "#6b6b66", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    COLORS = ["#2a78d6", "#eb6834"]  # categorical slots 1-2, validated (CVD dE 24.7)
    labels = ["Logistic\nRegression", "Random\nForest"]

    panels = [
        ("Single-sample inference latency (median, ms)", [meas[n]["infer_ms"] for n in MODEL_NAMES],
         {t: BUDGETS[t][2] for t in BUDGETS}, "ms", "{:.3f} ms"),
        ("Model size on disk (KB)", [meas[n]["size_bytes"] / KB for n in MODEL_NAMES],
         {t: BUDGETS[t][3] / KB for t in BUDGETS}, "KB", "{:,.1f} KB"),
        ("Peak RSS of inference process (MB)", [meas[n]["infer_peak_mb"] for n in MODEL_NAMES],
         {"Cloud (min)": 1024, "Edge (min)": 256, "Mobile (min)": 64, "TinyML (max)": 0.25}, "MB",
         "{:,.1f} MB"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), facecolor=SURFACE)
    for ax, (title, values, budgets, unit, fmt) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        bars = ax.bar(labels, values, color=COLORS, width=0.5, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_yscale("log")
        lo = min(min(values), min(budgets.values())) / 3
        hi = max(max(values), max(budgets.values())) * 3
        ax.set_ylim(lo, hi)
        for tier, limit in budgets.items():
            ax.hlines(limit, -0.5, 1.38, color=MUTED, linestyles=(0, (4, 3)), linewidth=1, zorder=2)
            ax.text(1.42, limit, tier,
                    va="center", ha="left", fontsize=8, color=MUTED)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.25, fmt.format(v),
                    ha="center", va="bottom", fontsize=9, color=INK, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none"))
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlim(-0.5, 1.95)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.set_ylabel(f"{unit} (log scale)", fontsize=8, color=MUTED)
    fig.suptitle("Lab 1 - measured system cost vs. deployment budgets (dashed = tier limit)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "system_cost.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    meas = load_measurements()

    with open(RESULTS_DIR / "measurements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "test_accuracy", "train_time_median_ms", "inference_latency_median_ms",
                    "inference_latency_p95_ms", "model_size_bytes", "model_size_kb",
                    "train_peak_rss_mb", "train_delta_mb", "inference_peak_rss_mb", "inference_delta_mb"])
        for n, m in meas.items():
            w.writerow([n, m["accuracy"], f"{m['train_ms']:.3f}", f"{m['infer_ms']:.4f}",
                        f"{m['infer_p95_ms']:.4f}", m["size_bytes"], f"{m['size_bytes'] / KB:.2f}",
                        f"{m['train_peak_mb']:.2f}", f"{m['train_delta_mb']:.2f}",
                        f"{m['infer_peak_mb']:.2f}", f"{m['infer_delta_mb']:.2f}"])

    rows = []
    for n, m in meas.items():
        for tier, (_, _, lat_max, size_max) in BUDGETS.items():
            mem_v = memory_verdict(tier, m["infer_peak_mb"])
            lat_v = "Yes" if m["infer_ms"] <= lat_max else "No"
            size_v = "Yes" if m["size_bytes"] <= size_max else "No"
            verdicts = (mem_v, lat_v, size_v)
            overall = ("No" if "No" in verdicts else
                       "Partly (larger devices only)" if "Partly" in verdicts else "Yes")
            rows.append({"model": n, "tier": tier, "memory_ok": mem_v, "latency_ok": lat_v,
                         "size_ok": size_v, "fits": overall})
            print(f"{n:<22}{tier:<8} memory={mem_v:<7} latency={lat_v:<4} size={size_v:<4} -> {overall}")

    with open(RESULTS_DIR / "deployment_fit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fp = [parameter_footprint(n) for n in MODEL_NAMES]
    with open(RESULTS_DIR / "model_footprint.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fp[0].keys()))
        w.writeheader()
        w.writerows(fp)

    plot(meas)


if __name__ == "__main__":
    main()
