"""Step 5: trade-off plots - test accuracy against parameter count and against inference latency.

Outputs: results/accuracy_vs_params.png, results/accuracy_vs_latency.png, results/tradeoffs.csv
"""

from __future__ import annotations

import csv

from common import MODEL_LABELS, MODEL_NAMES, RESULTS_DIR

COLORS = {"mlp": "#2a78d6", "cnn": "#eb6834", "attention": "#1baf7a"}  # validated palette slots 1-3
INK, MUTED, GRID, AXIS, SURFACE = "#1f1f1d", "#6b6b66", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def rows(name):
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load():
    params = {r["model"]: int(r["torchinfo_total_params"]) for r in rows("params.csv")}
    acc = {r["model"]: float(r["test_acc"]) for r in rows("test_accuracy.csv")}
    lat = {(r["model"], int(r["batch"])): float(r["median_ms"]) for r in rows("latency.csv")}
    flops = {r["model"]: int(r["hand_flops"]) for r in rows("flops_models.csv")}
    return params, acc, lat, flops


def style(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel("test accuracy", fontsize=8, color=MUTED)


def scatter(ax, xs, acc, fmt):
    for m in MODEL_NAMES:
        ax.scatter([xs[m]], [acc[m]], s=90, color=COLORS[m], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{MODEL_LABELS[m]}\n{fmt(xs[m])}, {acc[m]:.4f}", (xs[m], acc[m]), xytext=(8, -2),
                    textcoords="offset points", fontsize=8, color=INK, va="center")
    lo, hi = min(acc.values()), max(acc.values())
    pad = max(0.01, (hi - lo) * 0.35)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xscale("log")
    xmin, xmax = min(xs.values()), max(xs.values())
    lo_x, hi_x = xmin / 1.8, xmax * 4
    ax.set_xlim(lo_x, hi_x)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    cands = [m * 10 ** e for e in range(-4, 7) for m in (1, 2, 5)]
    ax.xaxis.set_major_locator(FixedLocator([c for c in cands if lo_x <= c <= hi_x]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: (f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")))
    ax.tick_params(axis="x", which="minor", length=0)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params, acc, lat, flops = load()

    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    style(ax, "Test accuracy vs. parameters (torchinfo)", "parameters (log scale)")
    scatter(ax, params, acc, lambda v: f"{v:,} params")
    fig.suptitle("Lab 3 - accuracy against model size", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS_DIR / "accuracy_vs_params.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    for ax, bs in zip(axes, (1, 128)):
        xs = {m: lat[(m, bs)] for m in MODEL_NAMES}
        unit = "ms per image" if bs == 1 else "ms per batch of 128"
        style(ax, f"Batch {bs}: median latency ({'100' if bs == 1 else '20'} runs)", f"{unit} (log scale)")
        scatter(ax, xs, acc, lambda v: f"{v:.3f} ms" if v < 1 else f"{v:.2f} ms")
    fig.suptitle("Lab 3 - accuracy against CPU inference latency", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "accuracy_vs_latency.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    out = []
    for m in MODEL_NAMES:
        out.append({"model": m, "test_acc": f"{acc[m]:.4f}", "params": params[m],
                    "hand_flops_per_image": flops[m],
                    "flops_per_param": f"{flops[m] / params[m]:.2f}",
                    "latency_b1_ms": f"{lat[(m, 1)]:.4f}", "latency_b128_ms": f"{lat[(m, 128)]:.3f}",
                    "latency_b128_ms_per_image": f"{lat[(m, 128)] / 128:.5f}",
                    "acc_per_100k_params": f"{acc[m] / (params[m] / 1e5):.3f}"})
    with open(RESULTS_DIR / "tradeoffs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("wrote accuracy_vs_params.png, accuracy_vs_latency.png, tradeoffs.csv")


if __name__ == "__main__":
    main()
