"""Steps 4-7: batch-size study, LR schedules x optimizers, gradient accumulation, interrupt/resume.

Every run is a separate train.py process (clean memory measurement) driven by the base YAML config
plus overrides, and every run is logged to MLflow.

  step 4  batch size [32, 64, 128, 256, 512], Adam 1e-3, 5 epochs: throughput, peak memory, test acc
  step 5  {SGD+momentum 0.9 lr 0.05, Adam lr 1e-3} x {StepLR(4, 0.1), CosineAnnealingLR(T=10),
          ReduceLROnPlateau(0.5, patience 1)}, equal 10-epoch budgets
  step 6  batch 256 vs batch 64 x accumulation 4 (+ batch 64 without accumulation as a contrast), 5 epochs
  step 7  the base run (5 epochs, uninterrupted) vs the same config interrupted after epoch 3 and resumed

Usage: python lab06/src/studies.py [batch|schedules|accum|resume]   (no argument = all)
Outputs: results/batch_size.csv/.png, results/schedules.csv/.png, results/grad_accum.csv/.png,
         results/resume.csv/.png
"""

from __future__ import annotations

import csv
import sys

from common import RESULTS_DIR, RUNS_DIR, read_json
from runner import train

INK, MUTED, GRID, AXIS, SURFACE = "#1f1f1d", "#6b6b66", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#7d5fd1", "#d4a20b"]


def write_csv(name, rows):
    with open(RESULTS_DIR / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)


def figure(ncols, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.2), facecolor=SURFACE)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    return plt, fig, axes


def save(plt, fig, name):
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS_DIR / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def curve(s, key):
    return [h["epoch"] for h in s["history"]], [h[key] for h in s["history"]]


# ------------------------------------------------------------------------------------------- step 4
def batch_study():
    rows, runs = [], {}
    for bs in [32, 64, 128, 256, 512]:
        s = train(f"bs-{bs}", data__batch_size=bs, mlflow__experiment="lab06-batch-size")
        runs[bs] = s
        rows.append({"batch_size": bs, "optimizer_steps_per_epoch": s["history"][-1]["optimizer_steps"],
                     "throughput_samples_per_s": s["mean_throughput_samples_per_s"],
                     "epoch_seconds_mean": sum(h["epoch_seconds"] for h in s["history"]) / len(s["history"]),
                     "peak_rss_mb": s["peak_rss_mb"], "peak_minus_baseline_mb": s["peak_minus_baseline_mb"],
                     "final_train_loss": s["history"][-1]["train_loss"], "final_val_acc": s["history"][-1]["val_acc"],
                     "test_acc_after_5_epochs": s["test_acc"]})
    write_csv("batch_size.csv", rows)
    plt, fig, axes = figure(3, "Lab 6 - batch-size study (CNN, Adam 1e-3, 5 epochs, CPU)")
    bss = [r["batch_size"] for r in rows]
    for ax, key, lab in zip(axes, ["throughput_samples_per_s", "peak_rss_mb", "test_acc_after_5_epochs"],
                            ["training samples / s", "peak RSS of the process (MB)", "test accuracy after 5 epochs"]):
        ax.plot(bss, [r[key] for r in rows], "-o", color=COLORS[0], linewidth=2, markersize=7, zorder=3)
        for b, r in zip(bss, rows):
            ax.annotate(f"{r[key]:.0f}" if key != "test_acc_after_5_epochs" else f"{r[key]:.4f}", (b, r[key]),
                        xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.set_xscale("log", base=2)
        ax.set_xticks(bss, [str(b) for b in bss])
        style(ax, lab, "batch size", lab)
    save(plt, fig, "batch_size.png")


# ------------------------------------------------------------------------------------------- step 5
OPTS = {"sgd": {"optimizer__name": "sgd", "optimizer__lr": 0.05, "optimizer__momentum": 0.9},
        "adam": {"optimizer__name": "adam", "optimizer__lr": 0.001}}
SCHEDS = ["step", "cosine", "plateau"]


def schedule_study():
    rows, runs = [], {}
    for o, okw in OPTS.items():
        for sc in SCHEDS:
            name = f"{o}-{sc}"
            s = train(name, epochs=10, scheduler__name=sc, mlflow__experiment="lab06-schedules", **okw)
            runs[name] = s
            h = s["history"]
            rows.append({"optimizer": "SGD+momentum 0.9, lr 0.05" if o == "sgd" else "Adam, lr 0.001",
                         "scheduler": {"step": "StepLR(4, 0.1)", "cosine": "CosineAnnealingLR(T_max=10)",
                                       "plateau": "ReduceLROnPlateau(0.5, patience 1)"}[sc],
                         "run": name, "final_lr": h[-1]["lr"], "best_val_acc": max(x["val_acc"] for x in h),
                         "final_val_acc": h[-1]["val_acc"], "final_train_loss": h[-1]["train_loss"],
                         "test_acc": s["test_acc"],
                         "train_seconds": sum(x["epoch_seconds"] for x in h)})
    write_csv("schedules.csv", rows)
    plt, fig, axes = figure(3, "Lab 6 - LR schedules x optimizers, equal 10-epoch budgets")
    for k, o in enumerate(OPTS):
        ax = axes[k]
        for c, sc in zip(COLORS, SCHEDS):
            ax.plot(*curve(runs[f"{o}-{sc}"], "val_acc"), "-o", color=c, linewidth=2, markersize=5, label=sc)
        style(ax, f"{'SGD+momentum' if o == 'sgd' else 'Adam'}: validation accuracy", "epoch", "val accuracy")
        ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    for c, sc in zip(COLORS, SCHEDS):
        ep, lr = curve(runs[f"sgd-{sc}"], "lr")
        ax.plot(ep, lr, "-o", color=c, linewidth=2, markersize=5, label=f"{sc} (SGD)")
    ax.set_yscale("log")
    style(ax, "Learning rate per epoch (SGD runs)", "epoch", "learning rate (log)")
    ax.legend(frameon=False, fontsize=8)
    save(plt, fig, "schedules.png")


# ------------------------------------------------------------------------------------------- step 6
def accum_study():
    runs = {
        "bs256": train("accum-bs256", data__batch_size=256, grad_accum_steps=1, mlflow__experiment="lab06-grad-accum"),
        "bs64x4": train("accum-bs64x4", data__batch_size=64, grad_accum_steps=4, mlflow__experiment="lab06-grad-accum"),
        "bs64": train("accum-bs64-no-accum", data__batch_size=64, grad_accum_steps=1,
                      mlflow__experiment="lab06-grad-accum"),
    }
    rows = []
    for e in range(len(runs["bs256"]["history"])):
        a, b, c = (runs[k]["history"][e] for k in ("bs256", "bs64x4", "bs64"))
        rows.append({"epoch": a["epoch"], "train_loss_bs256": a["train_loss"], "train_loss_bs64x4": b["train_loss"],
                     "train_loss_bs64": c["train_loss"], "val_acc_bs256": a["val_acc"], "val_acc_bs64x4": b["val_acc"],
                     "val_acc_bs64": c["val_acc"], "steps_bs256": a["optimizer_steps"],
                     "steps_bs64x4": b["optimizer_steps"], "steps_bs64": c["optimizer_steps"],
                     "abs_diff_train_loss_256_vs_64x4": abs(a["train_loss"] - b["train_loss"]),
                     "abs_diff_val_acc_256_vs_64x4": abs(a["val_acc"] - b["val_acc"])})
    write_csv("grad_accum.csv", rows)
    write_csv("grad_accum_summary.csv", [{"run": k, "test_acc": v["test_acc"],
                                          "throughput_samples_per_s": v["mean_throughput_samples_per_s"],
                                          "peak_rss_mb": v["peak_rss_mb"]} for k, v in runs.items()])
    plt, fig, axes = figure(2, "Lab 6 - gradient accumulation: batch 64 x 4 emulates batch 256")
    for key, ax, lab in (("train_loss", axes[0], "training loss"), ("val_acc", axes[1], "validation accuracy")):
        ax.plot(*curve(runs["bs256"], key), "-", color=COLORS[0], linewidth=5, label="batch 256")
        ax.plot(*curve(runs["bs64x4"], key), "--o", color=COLORS[1], linewidth=2, markersize=5,
                label="batch 64 x accumulation 4")
        ax.plot(*curve(runs["bs64"], key), ":", color=COLORS[2], linewidth=2, label="batch 64, no accumulation")
        style(ax, lab, "epoch", lab)
        ax.legend(frameon=False, fontsize=8)
    save(plt, fig, "grad_accum.png")


# ------------------------------------------------------------------------------------------- step 7
def resume_study():
    ref_path = RUNS_DIR / "base-cnn-adam.json"
    ref = read_json(ref_path) if ref_path.exists() else train("base-cnn-adam")
    part = train("resume-test", checkpoint__stop_after_epoch=3, mlflow__experiment="lab06-resume")
    full = train("resume-test-continued", checkpoint__resume_from=part["checkpoint"].replace("\\", "/"),
                 mlflow__experiment="lab06-resume")
    rows = []
    for r, f in zip(ref["history"], full["history"]):
        rows.append({"epoch": r["epoch"], "segment": "before interruption" if f["epoch"] <= 3 else "after resume",
                     "train_loss_uninterrupted": r["train_loss"], "train_loss_resumed": f["train_loss"],
                     "val_loss_uninterrupted": r["val_loss"], "val_loss_resumed": f["val_loss"],
                     "val_acc_uninterrupted": r["val_acc"], "val_acc_resumed": f["val_acc"],
                     "abs_diff_train_loss": abs(r["train_loss"] - f["train_loss"]),
                     "abs_diff_val_loss": abs(r["val_loss"] - f["val_loss"])})
    write_csv("resume.csv", rows)
    write_csv("resume_summary.csv", [{
        "uninterrupted_test_acc": ref["test_acc"], "resumed_test_acc": full["test_acc"],
        "max_abs_diff_train_loss": max(r["abs_diff_train_loss"] for r in rows),
        "max_abs_diff_val_loss": max(r["abs_diff_val_loss"] for r in rows),
        "same_mlflow_run": part["mlflow_run_id"] == full["mlflow_run_id"],
        "mlflow_run_id": full["mlflow_run_id"]}])
    plt, fig, axes = figure(2, "Lab 6 - interrupt after epoch 3, resume from checkpoint")
    for key, ax, lab in (("train_loss", axes[0], "training loss"), ("val_loss", axes[1], "validation loss")):
        ax.plot(*curve(ref, key), "-", color=COLORS[0], linewidth=5, label="uninterrupted run")
        ep, v = curve(full, key)
        ax.plot([e for e in ep if e <= 3], [x for e, x in zip(ep, v) if e <= 3], "--o", color=COLORS[1],
                linewidth=2, markersize=6, label="epochs 1-3 (then interrupted)")
        ax.plot([e for e in ep if e >= 3], [x for e, x in zip(ep, v) if e >= 3], "--s", color=COLORS[2],
                linewidth=2, markersize=6, label="resumed from checkpoint (epoch 4-5)")
        ax.axvline(3.5, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
        style(ax, lab, "epoch", lab)
        ax.legend(frameon=False, fontsize=8)
    save(plt, fig, "resume.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["batch", "schedules", "accum", "resume"]
    for w in which:
        {"batch": batch_study, "schedules": schedule_study, "accum": accum_study, "resume": resume_study}[w]()
