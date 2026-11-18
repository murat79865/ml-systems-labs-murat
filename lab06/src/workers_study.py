"""Step 3: vary DataLoader num_workers over [0, 1, 2, 4, 8] and find the bottleneck.

For every setting (batch 128, same seed):
  loader-only  - iterate over the whole training set without any model: pure data-pipeline speed,
                 plus the time to the first batch (worker start-up)
  train epoch  - one full training epoch of the CNN (fresh model): epoch time, throughput and the time
                 the training loop spent waiting for the next batch (data wait)
If the loader alone is much faster than training and the data wait is a small share, training is
compute-bound and extra workers cannot help; workers only pay off when the data wait is large.

Outputs: results/workers.csv, results/workers.png, MLflow experiment lab06-num-workers
"""

from __future__ import annotations

import csv
import time

from torch import nn

from common import RESULTS_DIR, flatten, get_datasets, load_config, make_loaders, make_model, repo_path, set_seeds
from train import make_optimizer, train_one_epoch

WORKERS = [0, 1, 2, 4, 8]


def main() -> None:
    import mlflow

    cfg = load_config()
    ds = get_datasets(cfg)
    n_train = len(ds[0])
    tracking = repo_path(cfg["mlflow"]["tracking_dir"])
    mlflow.set_tracking_uri(tracking.resolve().as_uri())
    mlflow.set_experiment("lab06-num-workers")
    rows = []
    for nw in WORKERS:
        set_seeds(cfg["seed"])
        loader, _, _, _ = make_loaders(cfg, ds, num_workers=nw)
        t0 = time.perf_counter()
        it = iter(loader)
        next(it)
        first = time.perf_counter() - t0
        for _ in it:
            pass
        loader_s = time.perf_counter() - t0

        set_seeds(cfg["seed"])
        loader, _, _, _ = make_loaders(cfg, ds, num_workers=nw)
        model = make_model(cfg)
        opt = make_optimizer(cfg, model.parameters())
        t0 = time.perf_counter()
        _, acc, wait, _ = train_one_epoch(model, loader, opt, nn.CrossEntropyLoss(), 1, n_train,
                                          cfg["data"]["batch_size"])
        epoch_s = time.perf_counter() - t0
        row = {"num_workers": nw, "loader_only_s": loader_s, "loader_samples_per_s": n_train / loader_s,
               "time_to_first_batch_s": first, "train_epoch_s": epoch_s, "train_samples_per_s": n_train / epoch_s,
               "data_wait_s": wait, "data_wait_share": wait / epoch_s, "train_acc_epoch1": acc}
        rows.append(row)
        with mlflow.start_run(run_name=f"workers-{nw}"):
            mlflow.log_params({k: ("null" if v is None else v) for k, v in flatten(cfg).items()
                               if k != "data.num_workers"})
            mlflow.log_param("data.num_workers", nw)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "num_workers"})
        print(f"num_workers {nw}: loader-only {loader_s:6.2f} s ({n_train / loader_s:7.0f}/s, first batch "
              f"{first:5.2f} s) | train epoch {epoch_s:6.2f} s ({n_train / epoch_s:6.0f}/s), data wait "
              f"{wait:5.2f} s ({wait / epoch_s:.0%})", flush=True)

    with open(RESULTS_DIR / "workers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    plot(rows)


def plot(rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, AXIS, SURFACE = "#1f1f1d", "#6b6b66", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    x = [r["num_workers"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    ax = axes[0]
    ax.plot(x, [r["train_epoch_s"] for r in rows], "-o", color=BLUE, linewidth=2, markersize=7, label="training epoch")
    ax.plot(x, [r["loader_only_s"] for r in rows], "-o", color=ORANGE, linewidth=2, markersize=7,
            label="data loader alone")
    ax.set_ylabel("seconds per epoch", fontsize=8, color=MUTED)
    ax.set_title("Epoch time vs num_workers", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.bar([str(v) for v in x], [100 * r["data_wait_share"] for r in rows], color=AQUA, edgecolor=SURFACE,
           linewidth=2, zorder=3)
    for i, r in enumerate(rows):
        ax.text(i, 100 * r["data_wait_share"] + 0.5, f"{r['data_wait_share']:.0%}", ha="center", fontsize=8, color=INK)
    ax.set_ylabel("% of training epoch waiting for data", fontsize=8, color=MUTED)
    ax.set_title("Data wait share of the training epoch", fontsize=10, color=INK, loc="left")
    for a in axes:
        a.set_facecolor(SURFACE)
        a.set_xlabel("num_workers", fontsize=8, color=MUTED)
        a.grid(color=GRID, linewidth=0.6, zorder=0)
        a.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(AXIS)
    fig.suptitle("Lab 6 - DataLoader workers (CNN, batch 128, CPU)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS_DIR / "workers.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
