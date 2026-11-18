"""Step 2: profile one training epoch with torch.profiler.

Every training step is wrapped in four torch.profiler.record_function regions -
data_loading (fetching the next batch from the DataLoader), forward (model + loss), backward
(loss.backward()) and optimizer_step (Adam step + zero_grad). The profiler's CPU time per region gives
the breakdown; plain time.perf_counter() timers around the same regions are recorded as a cross-check.
The epoch runs once without the profiler first (warm-up), then the profiled epoch follows.

Outputs: results/profile_breakdown.csv, results/profile_table.txt, results/profile.png,
         MLflow run 'profile-one-epoch' in experiment lab06-profile
"""

from __future__ import annotations

import csv
import time

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from common import RESULTS_DIR, flatten, load_config, make_loaders, make_model, repo_path, set_seeds, write_json
from train import make_optimizer

PHASES = ["data_loading", "forward", "backward", "optimizer_step"]


def epoch(model, loader, opt, loss_fn, timers=None, use_labels=False):
    model.train()
    it = iter(loader)
    while True:
        t0 = time.perf_counter()
        if use_labels:
            with record_function("data_loading"):
                batch = next(it, None)
        else:
            batch = next(it, None)
        t1 = time.perf_counter()
        if batch is None:
            break
        x, y = batch
        with record_function("forward") if use_labels else _null():
            loss = loss_fn(model(x), y)
        t2 = time.perf_counter()
        with record_function("backward") if use_labels else _null():
            loss.backward()
        t3 = time.perf_counter()
        with record_function("optimizer_step") if use_labels else _null():
            opt.step()
            opt.zero_grad(set_to_none=True)
        t4 = time.perf_counter()
        if timers is not None:
            for k, a, b in (("data_loading", t0, t1), ("forward", t1, t2), ("backward", t2, t3),
                            ("optimizer_step", t3, t4)):
                timers[k] += b - a


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main() -> None:
    import mlflow

    cfg = load_config()
    set_seeds(cfg["seed"])
    train_loader, _, _, _ = make_loaders(cfg)
    model = make_model(cfg)
    opt = make_optimizer(cfg, model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    epoch(model, train_loader, opt, loss_fn)  # warm-up epoch (not profiled)

    timers = {k: 0.0 for k in PHASES}
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        t0 = time.perf_counter()
        epoch(model, train_loader, opt, loss_fn, timers, use_labels=True)
        wall = time.perf_counter() - t0   # measured before the profiler post-processes its events

    ka = {e.key: e for e in prof.key_averages()}
    prof_ms = {k: ka[k].cpu_time_total / 1000.0 if k in ka else 0.0 for k in PHASES}
    total_prof = sum(prof_ms.values())
    total_wall = sum(timers.values())
    rows = []
    for k in PHASES:
        rows.append({"phase": k, "profiler_cpu_ms": f"{prof_ms[k]:.1f}",
                     "profiler_share": f"{prof_ms[k] / total_prof:.4f}",
                     "wall_ms": f"{timers[k] * 1000:.1f}", "wall_share": f"{timers[k] / total_wall:.4f}",
                     "calls": ka[k].count if k in ka else 0})
    rows.append({"phase": "other (loop overhead)", "profiler_cpu_ms": "", "profiler_share": "",
                 "wall_ms": f"{(wall - total_wall) * 1000:.1f}", "wall_share": "", "calls": ""})
    with open(RESULTS_DIR / "profile_breakdown.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=25)
    (RESULTS_DIR / "profile_table.txt").write_text(
        f"torch.profiler, one training epoch ({len(train_loader)} steps, batch {cfg['data']['batch_size']}, "
        f"num_workers {cfg['data']['num_workers']}), sorted by CPU total time\n\n" + table, encoding="utf-8")
    write_json(RESULTS_DIR / "profile_summary.json", {"epoch_wall_s": wall, "steps": len(train_loader),
                                                      "profiler_ms": prof_ms, "wall_ms": {k: v * 1000 for k, v in timers.items()}})
    for r in rows[:-1]:
        print(f"{r['phase']:<15} profiler {float(r['profiler_cpu_ms']):9.1f} ms ({float(r['profiler_share']):6.1%})  "
              f"wall {float(r['wall_ms']):9.1f} ms ({float(r['wall_share']):6.1%})")
    plot(rows[:-1])

    tracking = repo_path(cfg["mlflow"]["tracking_dir"])
    mlflow.set_tracking_uri(tracking.resolve().as_uri())
    mlflow.set_experiment("lab06-profile")
    for old in mlflow.search_runs(filter_string="tags.mlflow.runName = 'profile-one-epoch'", output_format="list"):
        mlflow.delete_run(old.info.run_id)  # keep a single profile run when the step is re-run
    with mlflow.start_run(run_name="profile-one-epoch"):
        mlflow.log_params({k: ("null" if v is None else v) for k, v in flatten(cfg).items()})
        for r in rows[:-1]:
            mlflow.log_metric(f"share_{r['phase']}", float(r["profiler_share"]))
            mlflow.log_metric(f"ms_{r['phase']}", float(r["profiler_cpu_ms"]))
        mlflow.log_metric("epoch_wall_s", wall)
        for f in ("profile_breakdown.csv", "profile_table.txt", "profile.png"):
            mlflow.log_artifact(str(RESULTS_DIR / f))


def plot(rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, SURFACE = "#1f1f1d", "#6b6b66", "#e1e0d9", "#fcfcfb"
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#7d5fd1"]
    fig, ax = plt.subplots(figsize=(9, 2.6), facecolor=SURFACE)
    left = 0.0
    for r, c in zip(rows, cols):
        share = float(r["profiler_share"])
        ax.barh([0], [share], left=left, color=c, edgecolor=SURFACE, linewidth=2)
        if share > 0.04:
            ax.text(left + share / 2, 0, f"{r['phase']}\n{share:.0%}", ha="center", va="center", fontsize=9,
                    color="white")
        left += share
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Share of one training epoch's CPU time (torch.profiler)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "profile.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
