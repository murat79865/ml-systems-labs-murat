"""Step 6: mini-batch SGD training for every (width, learning rate) pair.

Protocol (lab manual): shuffle every epoch, mini-batch size 32, 50 epochs, per-epoch logging,
learning rates [0.001, 0.01, 0.1], widths [16, 64]. Every run starts from the same seed (42), so
runs differ only in width and learning rate.

After every epoch the full train and test sets are evaluated (loss + accuracy) and logged; epoch 0
is the untrained network.

Main grid: weights initialised from N(0, 0.01) read as standard deviation 0.01 (the literal
np.random.normal(0, 0.01)). Because N(0, 0.01) is also commonly read as variance 0.01, the same grid
is repeated as a supplementary experiment with standard deviation 0.1 (variance 0.01).

Outputs: results/training_log.csv, results/final_accuracy.csv, results/params_<run>.npz,
         results/loss_curves.png, results/accuracy_curves.png
         supplementary (std 0.1): results/training_log_init_std0.1.csv,
         results/final_accuracy_init_std0.1.csv, results/loss_curves_init_std0.1.png
"""

from __future__ import annotations

import csv
import time

import numpy as np

import nn
from common import (BATCH_SIZE, EPOCHS, INIT_STD, LEARNING_RATES, RESULTS_DIR, SEED, WIDTHS,
                    load_data, run_name, set_seeds)

SUPPLEMENTARY_STD = 0.1

# Categorical colours (validated palette, slots 1-3) - one per learning rate.
LR_COLORS = {0.001: "#2a78d6", 0.01: "#eb6834", 0.1: "#1baf7a"}
INK, MUTED, GRID, AXIS, SURFACE = "#1f1f1d", "#6b6b66", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def evaluate(params, X, Y, y) -> tuple[float, float]:
    P, cache = nn.forward(params, X)
    return nn.cross_entropy(cache, Y), float((P.argmax(axis=1) == y).mean())


def train_one(width, lr, data, log_rows, std=INIT_STD, verbose=True, save=True) -> dict:
    X_train, X_test, Y_train, Y_test, y_train, y_test = data
    rng = np.random.default_rng(SEED)
    params = nn.init_params(width, rng, std)
    name = run_name(width, lr)

    def log(epoch, seconds):
        tr_loss, tr_acc = evaluate(params, X_train, Y_train, y_train)
        te_loss, te_acc = evaluate(params, X_test, Y_test, y_test)
        log_rows.append({"run": name, "init_std": std, "width": width, "lr": lr, "epoch": epoch,
                         "train_loss": tr_loss, "train_acc": tr_acc,
                         "test_loss": te_loss, "test_acc": te_acc, "epoch_seconds": seconds})
        if verbose:
            print(f"{name:<12} epoch {epoch:>2}/{EPOCHS}  train loss {tr_loss:.4f}  acc {tr_acc:.4f} | "
                  f"test loss {te_loss:.4f}  acc {te_acc:.4f}")
        return te_loss, te_acc

    log(0, 0.0)
    n = X_train.shape[0]
    total = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        order = rng.permutation(n)  # shuffle every epoch
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            _, cache = nn.forward(params, X_train[idx])
            grads = nn.backward(params, cache, Y_train[idx])
            nn.sgd_step(params, grads, lr)
        seconds = time.perf_counter() - t0
        total += seconds
        te_loss, te_acc = log(epoch, seconds)

    if save:
        np.savez(RESULTS_DIR / f"params_{name}.npz", **params)
    tr_loss, tr_acc = evaluate(params, X_train, Y_train, y_train)
    return {"run": name, "init_std": std, "width": width, "lr": lr, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "final_train_loss": f"{tr_loss:.4f}", "final_train_acc": f"{tr_acc:.4f}",
            "final_test_loss": f"{te_loss:.4f}", "final_test_acc": f"{te_acc:.4f}",
            "train_seconds": f"{total:.2f}"}


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.set_xlabel("epoch", fontsize=8, color=MUTED)


def plot(log_rows, metric, ylabel, fname, suptitle):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, len(WIDTHS), figsize=(12, 4.6), sharey=True, facecolor=SURFACE)
    for ax, width in zip(axes, WIDTHS):
        style(ax, f"Hidden width W = {width}")
        ends = []
        for k, lr in enumerate(LEARNING_RATES):
            rows = [r for r in log_rows if r["width"] == width and r["lr"] == lr]
            ep = [r["epoch"] for r in rows]
            # earlier learning rates are drawn wider underneath, so identical curves stay visible
            lw = 2 + 0.9 * (len(LEARNING_RATES) - 1 - k)
            ax.plot(ep, [r[f"train_{metric}"] for r in rows], color=LR_COLORS[lr], linewidth=lw,
                    solid_capstyle="butt", zorder=2 + k)
            ax.plot(ep, [r[f"test_{metric}"] for r in rows], color=LR_COLORS[lr], linewidth=2,
                    linestyle=(0, (4, 2)), zorder=5 + k)
            ends.append([rows[-1][f"test_{metric}"], lr])
        # direct labels: merge the ones that sit on top of each other
        lo, hi = min(r[f"test_{metric}"] for r in log_rows), max(r[f"test_{metric}"] for r in log_rows)
        gap = 0.06 * (hi - lo if hi > lo else 1.0)
        ends.sort()
        merged = []
        for y, lr in ends:
            if merged and abs(y - merged[-1][0]) < gap:
                merged[-1][1].append(lr)
            else:
                merged.append([y, [lr]])
        for y, lrs in merged:
            ax.annotate("lr=" + ", ".join(f"{v:g}" for v in sorted(lrs)), (EPOCHS, y), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK)
        ax.set_xlim(0, EPOCHS + 12)
    axes[0].set_ylabel(ylabel, fontsize=8, color=MUTED)
    handles = [Line2D([], [], color=LR_COLORS[lr], linewidth=2, label=f"lr = {lr:g}") for lr in LEARNING_RATES]
    handles += [Line2D([], [], color=MUTED, linewidth=2, label="train"),
                Line2D([], [], color=MUTED, linewidth=2, linestyle=(0, (4, 2)), label="test")]
    fig.legend(handles=handles, loc="upper right", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(suptitle, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS_DIR / fname, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_csvs(log_rows, finals, suffix=""):
    with open(RESULTS_DIR / f"training_log{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        for r in log_rows:
            w.writerow({**r, **{k: f"{r[k]:.6f}" for k in ("train_loss", "train_acc", "test_loss",
                                                             "test_acc", "epoch_seconds")}})
    with open(RESULTS_DIR / f"final_accuracy{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(finals[0].keys()))
        w.writeheader()
        w.writerows(finals)


def main() -> None:
    set_seeds()
    data = load_data()

    # ---- main grid: N(0, 0.01) with standard deviation 0.01 --------------------------------
    log_rows, finals = [], []
    for width in WIDTHS:
        for lr in LEARNING_RATES:
            finals.append(train_one(width, lr, data, log_rows))
    write_csvs(log_rows, finals)
    plot(log_rows, "loss", "cross-entropy loss", "loss_curves.png",
         "Lab 2 - loss per epoch, init N(0, 0.01) (solid = train, dashed = test), batch 32, 50 epochs")
    plot(log_rows, "acc", "accuracy", "accuracy_curves.png",
         "Lab 2 - accuracy per epoch, init N(0, 0.01) (solid = train, dashed = test)")
    print("\nFinal test accuracy (init std 0.01):")
    for r in finals:
        print(f"  {r['run']:<12} {r['final_test_acc']}")

    # ---- supplementary: standard deviation 0.1 (N(0, 0.01) read as variance) ---------------
    sup_rows, sup_finals = [], []
    for width in WIDTHS:
        for lr in LEARNING_RATES:
            sup_finals.append(train_one(width, lr, data, sup_rows, std=SUPPLEMENTARY_STD,
                                        verbose=False, save=False))
    write_csvs(sup_rows, sup_finals, suffix="_init_std0.1")
    plot(sup_rows, "loss", "cross-entropy loss", "loss_curves_init_std0.1.png",
         "Supplementary - loss per epoch with init std 0.1 (variance 0.01) (solid = train, dashed = test)")
    print("\nSupplementary final test accuracy (init std 0.1):")
    for r in sup_finals:
        print(f"  {r['run']:<12} {r['final_test_acc']}")


if __name__ == "__main__":
    main()
