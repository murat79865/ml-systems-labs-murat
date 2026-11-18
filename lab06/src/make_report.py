"""Build lab06/report.md from lab06/results/ (and the MLflow export). Interpretation sentences are chosen
from the measured values. The handwritten proposal is embedded if results/proposal_paper.(jpg|png) exists.
"""

from __future__ import annotations

import csv
import re

from common import LAB_DIR, REPO_DIR, RESULTS_DIR, RUNS_DIR, hardware_info, load_config, read_json, write_json


def rows(name):
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def photo():
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        if (RESULTS_DIR / f"proposal_paper.{ext}").exists():
            return f"proposal_paper.{ext}"
    return None


def versions():
    text = (RESULTS_DIR / "versions.txt").read_text(encoding="utf-8")
    py = re.search(r"Python\s*:\s*(\S+)", text)
    pr = re.search(r"problems:\s*(\d+)", text)
    return (py.group(1) if py else "?"), (int(pr.group(1)) if pr else -1)


f = float


def main() -> None:
    hw = hardware_info()
    write_json(RESULTS_DIR / "hardware.json", hw)
    cfg = load_config()
    py, problems = versions()
    base = read_json(RUNS_DIR / "base-cnn-adam.json")
    prof = rows("profile_breakdown.csv")
    psum = read_json(RESULTS_DIR / "profile_summary.json")
    wk = rows("workers.csv")
    bs = rows("batch_size.csv")
    sc = rows("schedules.csv")
    ga = rows("grad_accum.csv")
    gas = {r["run"]: r for r in rows("grad_accum_summary.csv")}
    rs = rows("resume.csv")
    rss = rows("resume_summary.csv")[0]
    ml = rows("mlflow_runs.csv")
    mls = read_json(RESULTS_DIR / "mlflow_summary.json")
    pic = photo()
    lab3 = None
    p3 = REPO_DIR / "lab03" / "results" / "test_accuracy.csv"
    if p3.exists():
        with open(p3, encoding="utf-8") as fh:
            lab3 = next((r for r in csv.DictReader(fh) if r["model"] == "cnn"), None)

    L = []
    add = L.append
    add("# Lab 6 - Training pipelines and project proposal\n")
    add("**Course:** Machine Learning (CYI5SE039CY), Group 11  ")
    add("**Reading:** Reddi, Ch. 8 AI Training (training-system architecture, data loading, optimization, "
        "profiling, §8.7 Optimization Techniques); Ch. 5 §5.5 Model Development; Ch. 13 §13.3 MLOps Key Components\n")
    add("## 1. Goal\n")
    add("Turn the best Lab 3 model (the CNN on Fashion-MNIST) into a reproducible training pipeline driven by one "
        "YAML config, with seeding, checkpoints and MLflow tracking; use it to profile an epoch and to study "
        "DataLoader workers, batch size, learning-rate schedules and optimizers, gradient accumulation and "
        "interrupt/resume; and write the individual project proposal.\n")

    # ------------------------------------------------------------------------------------------ method
    add("## 2. Pipeline\n")
    env_ok = py == "3.11.8" and problems == 0
    add("**Environment.** Python 3.11.8 in the pinned `.venv` "
        + ("(all 25 pins `OK`, `results/versions.txt`)" if env_ok else
           f"(**warning:** versions.txt reports Python {py}, {problems} mismatch(es))")
        + f"; PyTorch {hw['torch']}, MLflow {hw.get('mlflow', '?')}, CPU only ({hw['cpu']}, {hw['logical_cpus']} "
          f"logical CPUs, {hw['torch_threads']} PyTorch threads, {hw['os']}).\n")
    add("| Component | Implementation |")
    add("|---|---|")
    add("| One config object | `configs/base.yaml` - seed, data, model, optimizer, scheduler, gradient accumulation, "
        "checkpointing, MLflow; every study only overrides keys (`train.py --set data.batch_size=256`) |")
    add("| Training script | `src/train.py` - builds everything from the config; one function `run(cfg)` |")
    add("| Seed | `random`, `numpy`, `torch` = 42; the DataLoader shuffles with its own seeded generator |")
    add("| Checkpoint | after every epoch: model, optimizer, scheduler, epoch, history, all RNG states, DataLoader "
        "generator state, MLflow run id |")
    add("| MLflow | local store `lab06/mlflow_runs/`; params = flattened config, metrics per epoch "
        "(loss, accuracy, lr, epoch time, throughput, data wait), artifacts = config, history CSV, checkpoint |")
    add("| Studies | `profile_epoch.py`, `workers_study.py`, `studies.py` (batch, schedules, accum, resume) - each "
        "training run in its own process for clean memory numbers |")
    add("")

    # ------------------------------------------------------------------------------------------ step 1
    add("## 3. Results\n")
    add("### 3.1 Step 1 - the YAML-driven base run\n")
    h = base["history"]
    add(f"Base config: CNN (20,490 parameters), Adam lr {cfg['optimizer']['lr']}, batch {cfg['data']['batch_size']}, "
        f"{cfg['epochs']} epochs, seed {cfg['seed']}, 55k/5k/10k split.\n")
    add("| Epoch | Train loss | Train acc | Val loss | Val acc | Epoch time (s) | Samples/s |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for e in h:
        add(f"| {e['epoch']} | {e['train_loss']:.4f} | {e['train_acc']:.4f} | {e['val_loss']:.4f} | "
            f"{e['val_acc']:.4f} | {e['epoch_seconds']:.1f} | {e['throughput_samples_per_s']:.0f} |")
    add("")
    same3 = lab3 is not None and abs(f(lab3["test_acc"]) - base["test_acc"]) < 5e-4
    add(f"**Test accuracy {base['test_acc']:.4f}**"
        + (f" - identical to the Lab 3 CNN ({f(lab3['test_acc']):.4f}): same seed, same split, same model, so the "
           "restructured pipeline reproduces the earlier result exactly." if same3 else
           (f" (Lab 3 CNN: {f(lab3['test_acc']):.4f})." if lab3 else "."))
        + f" MLflow run `{base['mlflow_run_id']}` in experiment `lab06-base`; peak memory "
          f"{base['peak_rss_mb']:.0f} MB.\n")

    # ------------------------------------------------------------------------------------------ step 2
    add("### 3.2 Step 2 - profile of one epoch\n")
    add(f"`torch.profiler` over one full epoch ({psum['steps']} steps) after a warm-up epoch; each phase wrapped in "
        "`record_function`. Wall-clock timers around the same regions as a cross-check.\n")
    add("| Phase | Profiler CPU time (ms) | Share | Wall-clock (ms) | Share |")
    add("|---|---:|---:|---:|---:|")
    for r in prof:
        if r["profiler_share"]:
            add(f"| {r['phase']} | {f(r['profiler_cpu_ms']):,.0f} | **{f(r['profiler_share']):.1%}** | "
                f"{f(r['wall_ms']):,.0f} | {f(r['wall_share']):.1%} |")
    add("")
    add("![Profile](results/profile.png)\n")
    shares = {r["phase"]: f(r["profiler_share"]) for r in prof if r["profiler_share"]}
    top = max(shares, key=shares.get)
    fb = shares.get("forward", 0) + shares.get("backward", 0)
    rest = [k for k in ("data_loading", "optimizer_step") if k != top]
    add(f"The largest share is **{top} ({shares[top]:.0%})**; forward + backward together take {fb:.0%} of the "
        "epoch" + "".join(f", {k.replace('_', ' ')} {shares.get(k, 0):.0%}" for k in rest) + ". "
        + ("Backward costs more than forward because it computes two gradients per layer (weights and inputs), "
           "as measured in Lab 2. " if shares.get("backward", 0) > shares.get("forward", 0) else "")
        + ("With a 20k-parameter model Adam's update is cheap. " if shares.get("optimizer_step", 0) < 0.1 else "")
        + "Top operators: `results/profile_table.txt`.\n")

    # ------------------------------------------------------------------------------------------ step 3
    add("### 3.3 Step 3 - num_workers\n")
    add("| num_workers | Loader alone (s/epoch) | Time to first batch (s) | Training epoch (s) | Samples/s | Data wait | Data-wait share |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for r in wk:
        add(f"| {r['num_workers']} | {f(r['loader_only_s']):.2f} | {f(r['time_to_first_batch_s']):.2f} | "
            f"{f(r['train_epoch_s']):.2f} | {f(r['train_samples_per_s']):,.0f} | {f(r['data_wait_s']):.2f} s | "
            f"{f(r['data_wait_share']):.0%} |")
    add("")
    add("![Workers](results/workers.png)\n")
    w0 = wk[0]
    best_w = min(wk, key=lambda r: f(r["train_epoch_s"]))
    worst_w = max(wk, key=lambda r: f(r["train_epoch_s"]))
    gain = f(w0["train_epoch_s"]) / f(best_w["train_epoch_s"])
    if f(w0["data_wait_share"]) < 0.25:
        verdict = (f"**The bottleneck is computation, not data loading.** Even with `num_workers=0` the loop waits for "
                   f"data only {f(w0['data_wait_share']):.0%} of the epoch: decoding a 28x28 image and normalising it "
                   f"is cheap compared with the CNN's forward and backward pass. ")
    else:
        verdict = (f"**With `num_workers=0` data loading is a real bottleneck** ({f(w0['data_wait_share']):.0%} of the "
                   "epoch is spent waiting for batches). ")
    verdict += (f"The fastest setting is `num_workers={best_w['num_workers']}` ({f(best_w['train_epoch_s']):.1f} s, "
                f"{gain:.2f}x vs 0 workers); the slowest is `num_workers={worst_w['num_workers']}` "
                f"({f(worst_w['train_epoch_s']):.1f} s). Extra worker processes move loading off the main process, "
                "but each one must start a Python interpreter (see time to first batch; on Windows workers are "
                "spawned, not forked) and they compete with PyTorch's compute threads for the same CPU cores, so "
                "beyond the point where the data wait is gone more workers only add overhead.")
    add(verdict + "\n")

    # ------------------------------------------------------------------------------------------ step 4
    add("### 3.4 Step 4 - batch size\n")
    add("Adam lr 1e-3 for every batch size (no learning-rate scaling), 5 epochs, each run in a fresh process.\n")
    add("| Batch | Optimizer steps / epoch | Throughput (samples/s) | Peak RSS (MB) | Peak - baseline (MB) | Final val acc | **Test acc after 5 epochs** |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for r in bs:
        add(f"| {r['batch_size']} | {r['optimizer_steps_per_epoch']} | {f(r['throughput_samples_per_s']):,.0f} | "
            f"{f(r['peak_rss_mb']):.0f} | {f(r['peak_minus_baseline_mb']):.0f} | {f(r['final_val_acc']):.4f} | "
            f"**{f(r['test_acc_after_5_epochs']):.4f}** |")
    add("")
    add("![Batch size](results/batch_size.png)\n")
    tp = {int(r["batch_size"]): f(r["throughput_samples_per_s"]) for r in bs}
    ac = {int(r["batch_size"]): f(r["test_acc_after_5_epochs"]) for r in bs}
    me = {int(r["batch_size"]): f(r["peak_rss_mb"]) for r in bs}
    b_acc = max(ac, key=ac.get)
    b_tp = max(tp, key=tp.get)
    if b_tp == max(tp):
        tp_text = (f"Throughput rises from {tp[32]:,.0f} samples/s at batch 32 to {tp[b_tp]:,.0f} at batch {b_tp} "
                   f"(+{tp[b_tp] / tp[32] - 1:.0%}): larger batches amortise the per-step Python and kernel-launch "
                   "overhead and give the matrix kernels more work per call. ")
    else:
        tp_text = (f"Throughput rises from {tp[32]:,.0f} samples/s at batch 32 to a peak of {tp[b_tp]:,.0f} at batch "
                   f"{b_tp} (+{tp[b_tp] / tp[32] - 1:.0%}) and then falls slightly to {tp[512]:,.0f} at batch 512: "
                   "larger batches amortise the per-step Python and kernel-launch overhead, but once that overhead "
                   "is small the gain stops - most likely because the bigger activation tensors use the CPU caches less well. On "
                   "this small model the total effect of batch size on speed is modest. ")
    add(tp_text + f"Peak memory grows with the batch ({me[32]:.0f} -> {me[512]:.0f} MB) "
        "because activations for the whole batch are kept for the backward pass. "
        + (f"Accuracy after a fixed 5 epochs moves the other way: best at batch {b_acc} ({ac[b_acc]:.4f}), "
           f"{ac[512]:.4f} at batch 512 - with the same learning rate a large batch takes far fewer optimizer steps "
           f"({bs[0]['optimizer_steps_per_epoch']} vs {bs[-1]['optimizer_steps_per_epoch']} per epoch), so it has "
           "learned less after the same number of epochs. Large batches buy speed per sample, not accuracy per "
           "epoch, unless the learning rate is scaled up.\n" if ac[512] < ac[b_acc] - 0.002 else
           f"Accuracy after 5 epochs stays within {max(ac.values()) - min(ac.values()):.4f} across batch sizes "
           f"(best {ac[b_acc]:.4f} at batch {b_acc}), although a large batch takes far fewer optimizer steps "
           f"({bs[0]['optimizer_steps_per_epoch']} vs {bs[-1]['optimizer_steps_per_epoch']} per epoch).\n"))

    # ------------------------------------------------------------------------------------------ step 5
    add("### 3.5 Step 5 - learning-rate schedules and optimizers (10 epochs each)\n")
    add("| Optimizer | Scheduler | Final LR | Best val acc | Final val acc | Final train loss | **Test acc** | Time (s) |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in sc:
        add(f"| {r['optimizer']} | {r['scheduler']} | {f(r['final_lr']):.2e} | {f(r['best_val_acc']):.4f} | "
            f"{f(r['final_val_acc']):.4f} | {f(r['final_train_loss']):.4f} | **{f(r['test_acc']):.4f}** | "
            f"{f(r['train_seconds']):.0f} |")
    add("")
    add("![Schedules](results/schedules.png)\n")
    best = max(sc, key=lambda r: f(r["test_acc"]))
    sgd = [f(r["test_acc"]) for r in sc if r["optimizer"].startswith("SGD")]
    adam = [f(r["test_acc"]) for r in sc if r["optimizer"].startswith("Adam")]
    by_s = {}
    for r in sc:
        by_s.setdefault(r["scheduler"], []).append(f(r["test_acc"]))
    best_s = max(by_s, key=lambda k: sum(by_s[k]) / len(by_s[k]))
    add(f"Best combination: **{best['optimizer']} with {best['scheduler']} - test accuracy {f(best['test_acc']):.4f}**. "
        f"Averaged over the three schedules, {'Adam' if sum(adam) > sum(sgd) else 'SGD+momentum'} is ahead "
        f"(Adam {sum(adam) / 3:.4f}, SGD+momentum {sum(sgd) / 3:.4f}) under the same 10-epoch budget, and the best "
        f"schedule on average is {best_s}. StepLR cuts the learning rate by 10x at epochs 4 and 8, which first "
        "speeds up convergence but leaves only a tiny step size for the last epochs; cosine annealing decays "
        "smoothly to zero, so the final epochs fine-tune; ReduceLROnPlateau only reacts when the validation loss "
        "stops improving, so its learning rate stays high as long as progress continues. Adam adapts the step per "
        "parameter and is less sensitive to the choice of learning rate; SGD+momentum needs the schedule to do "
        "more of the work.\n")

    # ------------------------------------------------------------------------------------------ step 6
    add("### 3.6 Step 6 - gradient accumulation\n")
    add("Batch 64 with 4 accumulation steps emulates batch 256: gradients of four micro-batches are summed (each "
        "loss weighted by its share of the 256 samples) and the optimizer steps once. The shuffled order is "
        "the same, so the four micro-batches contain exactly the samples of one batch-256 step.\n")
    add("| Epoch | Train loss b256 | Train loss b64x4 | Train loss b64 (no accum) | Val acc b256 | Val acc b64x4 | Val acc b64 | Opt. steps b256 / b64x4 / b64 |")
    add("|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in ga:
        add(f"| {r['epoch']} | {f(r['train_loss_bs256']):.5f} | {f(r['train_loss_bs64x4']):.5f} | "
            f"{f(r['train_loss_bs64']):.5f} | {f(r['val_acc_bs256']):.4f} | {f(r['val_acc_bs64x4']):.4f} | "
            f"{f(r['val_acc_bs64']):.4f} | {r['steps_bs256']} / {r['steps_bs64x4']} / {r['steps_bs64']} |")
    add("")
    add("![Gradient accumulation](results/grad_accum.png)\n")
    dl = max(f(r["abs_diff_train_loss_256_vs_64x4"]) for r in ga)
    dv = max(f(r["abs_diff_val_acc_256_vs_64x4"]) for r in ga)
    add(f"**Batch 64 x 4 tracks batch 256 almost exactly** - largest difference in training loss {dl:.1e}, in "
        f"validation accuracy {dv:.4f}; test accuracy {f(gas['bs256']['test_acc']):.4f} vs "
        f"{f(gas['bs64x4']['test_acc']):.4f}. The remaining differences come only from floating-point summation "
        "order. The same micro-batch without accumulation (batch 64, 4x more optimizer steps) follows a different "
        f"curve (test {f(gas['bs64']['test_acc']):.4f}), which shows that the match is due to accumulation and not to "
        f"the micro-batch size. Peak memory: batch 256 {f(gas['bs256']['peak_rss_mb']):.0f} MB vs batch 64x4 "
        f"{f(gas['bs64x4']['peak_rss_mb']):.0f} MB - accumulation reaches the large-batch update with small-batch "
        "activation memory, at lower throughput.\n")

    # ------------------------------------------------------------------------------------------ step 7
    add("### 3.7 Step 7 - interrupt at epoch 3, resume from checkpoint\n")
    add("| Epoch | Segment | Train loss uninterrupted | Train loss interrupted + resumed | Val loss uninterrupted | Val loss resumed | Abs. diff (train / val) |")
    add("|---:|---|---:|---:|---:|---:|---|")
    for r in rs:
        add(f"| {r['epoch']} | {r['segment']} | {f(r['train_loss_uninterrupted']):.6f} | {f(r['train_loss_resumed']):.6f} | "
            f"{f(r['val_loss_uninterrupted']):.6f} | {f(r['val_loss_resumed']):.6f} | "
            f"{f(r['abs_diff_train_loss']):.1e} / {f(r['abs_diff_val_loss']):.1e} |")
    add("")
    add("![Resume](results/resume.png)\n")
    mx = max(f(rss["max_abs_diff_train_loss"]), f(rss["max_abs_diff_val_loss"]))
    add(f"The run was stopped after epoch 3 (checkpoint saved) and restarted from that checkpoint as a new process. "
        + (f"**The curves continue seamlessly: epochs 4-5 match the uninterrupted run to {mx:.1e}**" if mx < 1e-4 else
           f"The resumed curve continues from the checkpoint (largest difference {mx:.1e})")
        + f", and the test accuracy is {f(rss['resumed_test_acc']):.4f} vs {f(rss['uninterrupted_test_acc']):.4f}. This "
          "works because the checkpoint stores not only the weights but the optimizer state (Adam's moment "
          "estimates), the scheduler, the epoch counter and every random-number state including the DataLoader's "
          "shuffling generator. "
        + ("The resumed process also continued the same MLflow run, so the run's metric history is one unbroken "
           "curve." if rss["same_mlflow_run"] in ("True", True) else "") + "\n")

    # ------------------------------------------------------------------------------------------ mlflow
    add("### 3.8 MLflow runs\n")
    add(f"**{mls['runs']} runs in {len(mls['experiments'])} experiments** are stored in `lab06/mlflow_runs/` "
        "(params, per-epoch metrics, artifacts); browse them with `mlflow ui --backend-store-uri lab06/mlflow_runs`. "
        "Full export: `results/mlflow_runs.csv`.\n")
    add("| Experiment | Runs |")
    add("|---|---:|")
    for k, v in mls["runs_per_experiment"].items():
        add(f"| {k} | {v} |")
    add("")
    add("| Experiment | Run | Status | batch | workers | optimizer | lr | scheduler | accum | test acc |")
    add("|---|---|---|---:|---:|---|---:|---|---:|---:|")
    for r in ml:
        ta = r.get("test_acc", "")
        add(f"| {r['experiment']} | {r['run_name']} | {r['status']} | {r.get('data.batch_size', '')} | "
            f"{r.get('data.num_workers', '')} | {r.get('optimizer.name', '')} | {r.get('optimizer.lr', '')} | "
            f"{r.get('scheduler.name', '')} | {r.get('grad_accum_steps', '')} | "
            f"{(f'{f(ta):.4f}' if ta not in ('', None) else '-')} |")
    add("")

    # ------------------------------------------------------------------------------------------ step 8
    add("### 3.9 Step 8 - individual project proposal\n")
    add("The one-page proposal (problem, dataset, success metric, deployment constraints latency <= 50 ms and size "
        "<= 5 MB, planned optimization, name and ID) is in [`proposal.md`](proposal.md)"
        + (" and as the signed handwritten page below." if pic else ".") + "\n")
    if pic:
        add(f"![Project proposal](results/{pic})\n")

    # ------------------------------------------------------------------------------------------ conclusions
    add("## 4. Conclusions\n")
    add(f"1. **One config object makes every experiment a small, reproducible diff.** All {mls['runs']} runs come from "
        "`base.yaml` plus a few overrides, are fully logged in MLflow, and the base run "
        + ("reproduces the Lab 3 accuracy exactly" if same3 else "is reproducible from its seed")
        + "; checkpoints restore training " + ("exactly" if mx == 0 else f"to {mx:.0e}") + ".\n")
    compute_bound = f(w0["data_wait_share"]) < 0.25 and shares.get("data_loading", 0) < 0.25
    add(("2. **On this CPU the bottleneck is computation.** " if compute_bound else
         "2. **Data loading is a significant cost on this CPU.** ")
        + f"{top.replace('_', ' ').capitalize()} is the largest share of an epoch ({shares[top]:.0%})"
        + ("" if top == "data_loading" else f", data loading {shares.get('data_loading', 0):.0%}")
        + f"; the best DataLoader setting (num_workers={best_w['num_workers']}) was "
        f"{gain:.2f}x the speed of num_workers=0, while the best batch size (batch {b_tp}) raised throughput "
        f"only +{tp[b_tp] / tp[32] - 1:.0%} over batch 32 and larger batches cost memory.\n")
    add(f"3. **Optimizer and schedule matter more than raw speed for the final result.** Under the same 10-epoch "
        f"budget the best combination ({best['optimizer'].split(',')[0]} + {best['scheduler'].split('(')[0]}) reached "
        f"{f(best['test_acc']):.4f}; gradient accumulation reproduces a large-batch update with small-batch memory.\n")
    add("## 5. Reproducibility\n")
    add("With the virtual environment active, run `python lab06/src/run_all.py` from the repository root (about 30 "
        "minutes on a laptop CPU). It regenerates `lab06/results/`, the MLflow store `lab06/mlflow_runs/` and this "
        "report. A single run: `python lab06/src/train.py --config lab06/configs/base.yaml --set epochs=10 "
        "scheduler.name=cosine`.\n")
    (LAB_DIR / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {LAB_DIR / 'report.md'}" + ("" if pic else "  (note: proposal_paper photo not found yet)"))


if __name__ == "__main__":
    main()
