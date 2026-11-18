"""Step 1: YAML-driven training script for the best Lab 3 model (the CNN).

Everything is controlled by ONE config object (lab06/configs/base.yaml, optionally changed with
--set key=value): seed, data, optimizer, scheduler, gradient accumulation, checkpointing, MLflow.

  * Seed       - random / numpy / torch seeded; the DataLoader shuffles with its own seeded generator.
  * Checkpoint - after every epoch: model, optimizer, scheduler, epoch, history, all RNG states and the
                 DataLoader generator state, plus the MLflow run id -> results are identical after resume.
  * Resume     - checkpoint.resume_from continues the SAME MLflow run from the next epoch.
  * Interrupt  - checkpoint.stop_after_epoch stops after that epoch (simulated crash).
  * MLflow     - params (the flattened config), metrics per epoch (step = epoch), artifacts (config,
                 history CSV, final checkpoint) in a local file store at mlflow.tracking_dir.
  * Grad accum - grad_accum_steps micro-batches per optimizer step; every micro-batch loss is weighted
                 by its share of the group, so the update equals that of one large batch.

Usage:  python lab06/src/train.py --config lab06/configs/base.yaml --set data.batch_size=256 --out x.json
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from common import (BASE_CONFIG, RUNS_DIR, PeakRSS, flatten, load_config, make_loaders, make_model,
                    repo_path, set_seeds, write_json)


def make_optimizer(cfg, params):
    o = cfg["optimizer"]
    if o["name"] == "adam":
        return torch.optim.Adam(params, lr=o["lr"], weight_decay=o["weight_decay"])
    if o["name"] == "sgd":
        return torch.optim.SGD(params, lr=o["lr"], momentum=o["momentum"], weight_decay=o["weight_decay"])
    raise ValueError(o["name"])


def make_scheduler(cfg, opt):
    s = cfg["scheduler"]
    if s["name"] == "none":
        return None
    if s["name"] == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=s["step_size"], gamma=s["gamma"])
    if s["name"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    if s["name"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=s["plateau_factor"],
                                                          patience=s["plateau_patience"])
    raise ValueError(s["name"])


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    n, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        out = model(x)
        loss_sum += loss_fn(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


def group_sizes(n: int, micro: int, accum: int) -> list[int]:
    """Total number of samples in each accumulation group (the last group may be smaller)."""
    sizes = [micro] * (n // micro) + ([n % micro] if n % micro else [])
    return [sum(sizes[i:i + accum]) for i in range(0, len(sizes), accum)]


def train_one_epoch(model, loader, opt, loss_fn, accum: int, n_train: int, micro_bs: int):
    model.train()
    groups = group_sizes(n_train, micro_bs, accum)
    seen, correct, loss_sum, data_wait = 0, 0, 0.0, 0.0
    it = iter(loader)
    step = 0
    t_end = time.perf_counter()
    while True:
        try:
            x, y = next(it)
        except StopIteration:
            break
        t_got = time.perf_counter()
        data_wait += t_got - t_end
        g = step // accum
        if step % accum == 0:
            opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        (loss * (y.size(0) / groups[g])).backward()   # weight = share of this micro-batch in its group
        last_in_group = (step % accum == accum - 1) or (seen + y.size(0) == n_train)
        if last_in_group:
            opt.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += y.size(0)
        step += 1
        t_end = time.perf_counter()
    return loss_sum / seen, correct / seen, data_wait, step


def rng_state(gen):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "loader_generator": gen.get_state()}


def set_rng_state(state, gen):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    gen.set_state(state["loader_generator"])


def run(cfg: dict, out_json: Path | None = None) -> dict:
    import mlflow

    set_seeds(cfg["seed"])
    train_loader, val_loader, test_loader, gen = make_loaders(cfg)
    n_train = len(train_loader.dataset)
    model = make_model(cfg)
    opt = make_optimizer(cfg, model.parameters())
    sched = make_scheduler(cfg, opt)
    loss_fn = nn.CrossEntropyLoss()
    accum = int(cfg["grad_accum_steps"])
    ckpt_dir = repo_path(cfg["checkpoint"]["dir"]) / cfg["run_name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, history, run_id = 1, [], None
    if cfg["checkpoint"]["resume_from"]:
        ck = torch.load(repo_path(cfg["checkpoint"]["resume_from"]), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if sched is not None and ck["scheduler"] is not None:
            sched.load_state_dict(ck["scheduler"])
        set_rng_state(ck["rng"], gen)
        start_epoch, history, run_id = ck["epoch"] + 1, ck["history"], ck["mlflow_run_id"]
        print(f"resumed from {cfg['checkpoint']['resume_from']} at epoch {start_epoch}", flush=True)

    tracking = repo_path(cfg["mlflow"]["tracking_dir"])
    tracking.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking.resolve().as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    status = "completed"
    with PeakRSS() as mem, mlflow.start_run(run_id=run_id, run_name=None if run_id else cfg["run_name"]) as active:
        run_id = active.info.run_id
        if start_epoch == 1:
            mlflow.log_params({k: ("null" if v is None else v) for k, v in flatten(cfg).items()})
            mlflow.log_param("model.parameters", sum(p.numel() for p in model.parameters()))
        else:
            mlflow.set_tag("resumed_at_epoch", start_epoch)
        train_time = 0.0
        for epoch in range(start_epoch, cfg["epochs"] + 1):
            t0 = time.perf_counter()
            tr_loss, tr_acc, wait, steps = train_one_epoch(model, train_loader, opt, loss_fn, accum, n_train,
                                                            cfg["data"]["batch_size"])
            secs = time.perf_counter() - t0
            train_time += secs
            va_loss, va_acc = evaluate(model, val_loader, loss_fn)
            lr = opt.param_groups[0]["lr"]
            if sched is not None:
                sched.step(va_loss) if cfg["scheduler"]["name"] == "plateau" else sched.step()
            row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss,
                   "val_acc": va_acc, "lr": lr, "epoch_seconds": secs, "data_wait_seconds": wait,
                   "throughput_samples_per_s": n_train / secs, "optimizer_steps": -(-steps // accum)}
            history.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            print(f"[{cfg['run_name']}] epoch {epoch}/{cfg['epochs']}  train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {lr:.2e} | {secs:.1f}s "
                  f"({n_train / secs:.0f} samples/s)", flush=True)
            ck_path = ckpt_dir / "last.pt"
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict() if sched is not None else None, "epoch": epoch,
                        "history": history, "rng": rng_state(gen), "mlflow_run_id": run_id, "config": cfg},
                       ck_path)
            if cfg["checkpoint"]["stop_after_epoch"] == epoch:
                status = "interrupted"
                print(f"[{cfg['run_name']}] simulated interruption after epoch {epoch}; checkpoint {ck_path}",
                      flush=True)
                break
        summary = {"run_name": cfg["run_name"], "status": status, "mlflow_run_id": run_id,
                   "experiment": cfg["mlflow"]["experiment"], "checkpoint": str(ck_path), "history": history}
        if status == "completed":
            te_loss, te_acc = evaluate(model, test_loader, loss_fn)
            summary.update({"test_loss": te_loss, "test_acc": te_acc})
            mlflow.log_metrics({"test_loss": te_loss, "test_acc": te_acc})
        mlflow.set_tag("status", status)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(history[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(history)
        mlflow.log_text(buf.getvalue(), "history.csv")
        mlflow.log_text(yaml.safe_dump(cfg, sort_keys=False), "config.yaml")
        mlflow.log_artifact(str(ck_path), artifact_path="checkpoint")
    eps = [h for h in history if h["epoch"] >= start_epoch]
    summary.update({
        "train_seconds_this_session": train_time,
        "mean_throughput_samples_per_s": float(np.mean([h["throughput_samples_per_s"] for h in eps])) if eps else None,
        "peak_rss_mb": mem.peak / 2**20, "baseline_rss_mb": mem.baseline / 2**20,
        "peak_minus_baseline_mb": (mem.peak - mem.baseline) / 2**20,
        "config": cfg,
    })
    if out_json:
        write_json(out_json, summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(BASE_CONFIG))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", default=None, help="summary JSON (default results/runs/<run_name>.json)")
    a = ap.parse_args()
    cfg = load_config(Path(a.config), a.set)
    out = Path(a.out) if a.out else RUNS_DIR / f"{cfg['run_name']}.json"
    s = run(cfg, out)
    print(f"[{cfg['run_name']}] {s['status']}; test acc {s.get('test_acc', float('nan')):.4f}; "
          f"peak RSS {s['peak_rss_mb']:.0f} MB; summary {out}")


if __name__ == "__main__":
    main()
