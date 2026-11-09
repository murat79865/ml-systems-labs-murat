"""Step 5: build lab01/report.md from the files in results/.

Every number in the report is read from results/, so the report always matches the run that
produced it. Read the generated text, check it against your own run, and rewrite the
conclusions in your own words before committing (the manual requires your own wording).
"""

from __future__ import annotations

import csv
import re

from budget_fit import BUDGETS, load_measurements
from common import LAB_DIR, MODEL_LABELS, MODEL_NAMES, RESULTS_DIR, read_json

KB, MB = 1024, 1024 * 1024


def rows(name: str) -> list[dict]:
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def versions_summary() -> tuple[str, int, str]:
    text = (RESULTS_DIR / "versions.txt").read_text(encoding="utf-8")
    py = re.search(r"Python\s*:\s*(\S+)", text)
    problems = re.search(r"problems:\s*(\d+)", text)
    return (py.group(1) if py else "?"), (int(problems.group(1)) if problems else -1), text


def fmt_bytes(b: int) -> str:
    return f"{b:,} B ({b / KB:,.2f} KB)"


def main() -> None:
    m = load_measurements()
    split = read_json(RESULTS_DIR / "split_info.json")
    hw = read_json(RESULTS_DIR / "hardware.json")
    fit = rows("deployment_fit.csv")
    fp = {r["model"]: r for r in rows("model_footprint.csv")}
    mem_tools = sorted({r["tool"] for r in rows("memory.csv")})
    py, problems, _ = versions_summary()

    lr, rf = m["logistic_regression"], m["random_forest"]
    acc_lr, acc_rf = float(lr["accuracy"]), float(rf["accuracy"])
    lat_ratio = rf["infer_ms"] / lr["infer_ms"]
    size_ratio = rf["size_bytes"] / lr["size_bytes"]
    better = "Logistic Regression" if acc_lr > acc_rf else "Random Forest" if acc_rf > acc_lr else "neither model"
    acc_gap_pp = abs(acc_lr - acc_rf) * 100
    test_n = split["test_size"]
    acc_gap_samples = round(abs(acc_lr - acc_rf) * test_n)

    fit_by = {(r["model"], r["tier"]): r for r in fit}

    def tiers_where(model, value_prefix):
        return [t for t in BUDGETS if fit_by[(model, t)]["fits"].startswith(value_prefix)]

    env_ok = (py == "3.11.8" and problems == 0)
    env_line = ("The environment matches the required stack exactly: Python 3.11.8 and all 25 pinned "
                "packages report `OK` in `results/versions.txt`." if env_ok else
                f"**Warning:** `results/versions.txt` reports Python {py} and {problems} package "
                "mismatch(es). Re-run inside the pinned `.venv` before submitting.")

    L = []
    add = L.append
    add("# Lab 1 - Environment and first system measurements\n")
    add("**Course:** Machine Learning (CYI5SE039CY), Group 11  ")
    add("**Reading:** Reddi, *Introduction to Machine Learning Systems* - Ch. 1 §§1.4-1.6, "
        "Ch. 2 §§2.2-2.5 and §§2.8-2.9, Ch. 12 §§12.4 and 12.7\n")

    add("## 1. Goal\n")
    add("Set up a reproducible, pinned Python environment and measure what two classic baselines "
        "cost as *systems*, not only how accurate they are: training time, single-sample inference "
        "latency, model size and peak memory. Then use these measurements to decide on which "
        "deployment tier (Cloud, Edge, Mobile, TinyML) each model could realistically run.\n")

    add("## 2. Method\n")
    add("**Environment.** Python 3.11.8 in an isolated `.venv`, packages installed from the pinned "
        "`requirements.txt`. `src/print_versions.py` prints the Python version and the installed "
        "version of every pinned package, compares each with its pin, and its output is saved to "
        f"`results/versions.txt`. {env_line}\n")
    add(f"**Hardware.** {hw['cpu']}, {hw['logical_cpus']} logical CPUs"
        + (f", {hw['total_ram_gb']} GB RAM" if "total_ram_gb" in hw else "")
        + f", {hw['os']} ({hw['machine']}). All numbers below are specific to this machine.\n")
    add(f"**Data.** Breast Cancer Wisconsin (Diagnostic), loaded with "
        f"`load_breast_cancer(return_X_y=True)`: {split['n_samples']} samples, "
        f"{split['n_features']} numeric features, binary label. Split 70/30, stratified, "
        f"`random_state=42` -> {split['train_size']} train / {split['test_size']} test samples; "
        f"class counts (0 = malignant / 1 = benign): train {split['train_class_counts']['0']} / "
        f"{split['train_class_counts']['1']}, test {split['test_class_counts']['0']} / "
        f"{split['test_class_counts']['1']}. "
        "The dataset ships with scikit-learn, so nothing is downloaded and `data/` stays empty "
        "(it is git-ignored).\n")
    add("**Models.** Exactly as specified: `LogisticRegression(max_iter=1000, random_state=42)` and "
        "`RandomForestClassifier(n_estimators=100, random_state=42)`, default hyper-parameters "
        "otherwise, no feature scaling. Seeds: `random.seed(42)`, `np.random.seed(42)`, "
        "`torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)` (torch is seeded in every script except the "
        "memory-measurement processes, where importing torch would add hundreds of MB that these "
        "scikit-learn models never use).\n")
    add("**Measurement protocol.**\n")
    add("| Quantity | How it was measured |")
    add("|---|---|")
    add("| Test accuracy | `accuracy_score` on the 30 % test split, 4 decimals -> `results/baseline_accuracy.csv` |")
    add("| Training time | `time.perf_counter()` around `fit()` on a fresh model; 1 warm-up run discarded, then 5 runs; **median** reported |")
    add("| Inference latency | `predict()` on one test sample (shape 1x30); 1 warm-up, then 100 runs; **median** (and p95) reported |")
    add("| Model size | `joblib.dump` of each fitted model (`results/model_<name>.joblib`; both models together in `results/model.joblib`); file size in bytes and KB (1 KB = 1024 B) |")
    add(f"| Peak memory | Each model/phase in a **fresh Python process**. Tools: {', '.join('`' + t + '`' for t in mem_tools)} "
        "(`memory_profiler.memory_usage(max_usage=True)` at 1 ms and a `psutil.Process().memory_info().rss` sampler at 0.5 ms). "
        "Training = one `fit()`; inference = `joblib.load()` + 101 single-sample predictions. The higher reading of the tools is used |")
    add("")
    add("Memory is reported two ways: **peak RSS** (the whole process: interpreter + NumPy/SciPy/"
        "scikit-learn + data + model - what a device really has to provide) and **delta** (peak minus "
        "the RSS just before the measured work - the extra memory the training or inference step itself adds).\n")

    add("## 3. Results\n")
    add("### 3.1 Accuracy and system cost\n")
    add("| Metric | Logistic Regression | Random Forest (100 trees) |")
    add("|---|---:|---:|")
    add(f"| Test accuracy | {lr['accuracy']} | {rf['accuracy']} |")
    add(f"| Training time, median of 5 (ms) | {lr['train_ms']:.2f} | {rf['train_ms']:.2f} |")
    add(f"| Inference latency, 1 sample, median of 100 (ms) | {lr['infer_ms']:.4f} | {rf['infer_ms']:.4f} |")
    add(f"| Inference latency, p95 (ms) | {lr['infer_p95_ms']:.4f} | {rf['infer_p95_ms']:.4f} |")
    add(f"| Model size (joblib) | {fmt_bytes(lr['size_bytes'])} | {fmt_bytes(rf['size_bytes'])} |")
    bundle = (RESULTS_DIR / "model.joblib").stat().st_size
    add(f"| Peak RSS during training (MB) | {lr['train_peak_mb']:.1f} | {rf['train_peak_mb']:.1f} |")
    add(f"| Training delta (MB) | {lr['train_delta_mb']:.2f} | {rf['train_delta_mb']:.2f} |")
    add(f"| Peak RSS during inference (MB) | {lr['infer_peak_mb']:.1f} | {rf['infer_peak_mb']:.1f} |")
    add(f"| Inference delta (MB) | {lr['infer_delta_mb']:.2f} | {rf['infer_delta_mb']:.2f} |")
    add("")
    add(f"`results/model.joblib` (written with `joblib.dump`, both fitted models in one file): "
        f"{fmt_bytes(bundle)}. Per-model sizes in the table come from the individual dumps "
        f"`results/model_logistic_regression.joblib` and `results/model_random_forest.joblib`.\n")
    if lr["train_ms"] > rf["train_ms"]:
        add(f"*Observation:* on this machine Logistic Regression takes longer to train "
            f"({lr['train_ms']:.0f} ms) than the 100-tree Random Forest ({rf['train_ms']:.0f} ms). "
            f"The reason is the unscaled data: lbfgs never converges and runs all 1000 iterations "
            f"(see the note below), while the forest builds shallow trees quickly. The simpler model "
            f"is not automatically the cheaper one to train - it has to be measured.\n")
    if not lr["converged"]:
        add(f"*Note:* with the prescribed `max_iter=1000` and unscaled features, the lbfgs solver of "
            "Logistic Regression stops at the iteration limit and scikit-learn raises a "
            "`ConvergenceWarning`. The model was kept exactly as specified; its accuracy is still "
            f"{lr['accuracy']}. Standardising the features would let it converge in far fewer iterations "
            "(and likely train faster) - a fix for later labs, not a change to this baseline.\n")
    add("![System cost versus deployment budgets](results/system_cost.png)\n")
    add("*Figure 1 - Measured latency, model size and inference-process memory (log scale). "
        "Dashed lines are the tier limits from the lab manual.*\n")
    add("Raw data: `results/timing_raw.json` (every individual run), `results/memory.csv` "
        "(per tool), `results/measurements.csv` (the table above).\n")

    add("### 3.2 Resource budget table\n")
    add("| Tier | Memory budget | Latency budget | Size budget |")
    add("|---|---|---|---|")
    add("| Cloud | >= 1 GB | <= 100 ms | <= 500 MB |")
    add("| Edge | 256-1024 MB | <= 50 ms | <= 50 MB |")
    add("| Mobile | 64-256 MB | <= 20 ms | <= 10 MB |")
    add("| TinyML | <= 256 KB | <= 10 ms | <= 100 KB |")
    add("")
    add("Decision rule: latency and size must be within the limit. The memory budget is the RAM a device "
        "of that tier has, compared with the **peak RSS of the inference process**; for Edge and Mobile, "
        "*Yes* = fits even the smallest device of the range (peak <= lower bound), *Partly* = fits only "
        "the larger devices (lower < peak <= upper bound).\n")
    add("| Model | Tier | Memory | Latency | Size | **Can run?** |")
    add("|---|---|:---:|:---:|:---:|---|")
    for r in fit:
        add(f"| {MODEL_LABELS[r['model']]} | {r['tier']} | {r['memory_ok']} | {r['latency_ok']} | "
            f"{r['size_ok']} | **{r['fits']}** |")
    add("")

    add("## 4. Deployment-fit argument\n")
    for name in MODEL_NAMES:
        d = m[name]
        yes = tiers_where(name, "Yes")
        partly = tiers_where(name, "Partly")
        add(f"**{MODEL_LABELS[name]}.** ")
        parts = []
        if yes:
            parts.append(f"Fits **{', '.join(yes)}** without restriction: median latency "
                         f"{d['infer_ms']:.3f} ms, model {d['size_bytes'] / KB:,.1f} KB and an inference "
                         f"process of {d['infer_peak_mb']:.0f} MB are all inside those budgets with a large margin.")
        if partly:
            parts.append(f"**{', '.join(partly)}**: latency and size pass, but the {d['infer_peak_mb']:.0f} MB "
                         f"Python process is above the {BUDGETS['Mobile'][0]:.0f} MB lower bound, so it runs "
                         f"only on the larger devices of that range. Almost all of that memory is the Python "
                         f"runtime and libraries (baseline before the work), not the model "
                         f"(inference delta {d['infer_delta_mb']:.1f} MB).")
        t = fit_by[(name, "TinyML")]
        reasons = []
        if t["memory_ok"] == "No":
            reasons.append(f"a {d['infer_peak_mb']:.0f} MB process versus a 256 KB budget")
        if t["size_ok"] == "No":
            reasons.append(f"a {d['size_bytes'] / KB:.0f} KB model file versus 100 KB")
        if t["latency_ok"] == "No":
            reasons.append(f"{d['infer_ms']:.2f} ms latency versus 10 ms")
        if reasons:
            parts.append(f"**Not TinyML** as built: {'; '.join(reasons)}.")
        f = fp[name]
        parts.append(f"The learned parameters themselves are small - {f['structure']}, about "
                     f"{int(f['float32_bytes']):,} B if exported as float32 C arrays "
                     f"(`results/model_footprint.csv`)"
                     + (" - so a hand-ported C version would fit a microcontroller easily."
                        if int(f["float32_bytes"]) < 100 * KB else
                        " - still close to the 100 KB / 256 KB TinyML limits, so trees would have to be "
                        "pruned or reduced before porting.")
                     + " That would be a different deployment artifact than the one measured here.")
        add(" ".join(parts) + "\n")
    add(f"**Recommendation.** "
        + (f"Accuracy is practically the same (Logistic Regression ahead by {acc_gap_pp:.2f} percentage points, "
           f"i.e. {acc_gap_samples} of {test_n} test samples), while Logistic Regression is "
           f"about {lat_ratio:.0f}x faster per prediction and about {size_ratio:.0f}x smaller on disk. "
           "For any constrained target (Edge, Mobile, and TinyML after porting) Logistic Regression is the "
           "better choice; the Random Forest offers no accuracy advantage that would pay for its extra cost."
           if acc_lr >= acc_rf else
           f"Random Forest is {acc_gap_pp:.2f} percentage points ({acc_gap_samples} test samples) more "
           f"accurate, but Logistic Regression is about {lat_ratio:.0f}x faster and {size_ratio:.0f}x smaller. "
           "Use the Random Forest on Cloud/Edge where its cost is irrelevant, and Logistic Regression where "
           "latency or size is tight.") + "\n")

    add("## 5. Conclusions\n")
    add(f"1. **Accuracy alone does not separate these models; system cost does.** Test accuracy is "
        f"{lr['accuracy']} vs {rf['accuracy']} - a difference of {acc_gap_samples} test sample(s) out of {test_n}. "
        f"The system cost differs by orders of magnitude: the forest is about {lat_ratio:.0f}x slower per "
        f"prediction and about {size_ratio:.0f}x larger. On this task, {better} gives the better accuracy, "
        "but the cost difference is what decides deployment.\n")
    add(f"2. **For small models the runtime, not the model, dominates memory.** Both inference processes "
        f"peak at roughly {min(lr['infer_peak_mb'], rf['infer_peak_mb']):.0f}-{max(lr['infer_peak_mb'], rf['infer_peak_mb']):.0f} MB, "
        f"while the models are {lr['size_bytes'] / KB:.1f} KB and {rf['size_bytes'] / KB:.0f} KB. "
        "This is why neither fits TinyML and both are only *partly* fit for Mobile as Python programs: "
        "reaching those tiers needs a different runtime (e.g. ONNX Runtime / exported C code), not a "
        "smaller model.\n")
    add("3. **How you measure changes the answer.** A single timing run of a sub-millisecond call is "
        "dominated by noise and first-call effects, so warm-up plus the median of many runs is necessary "
        f"(the RF latency p95 is {rf['infer_p95_ms']:.2f} ms vs a median of {rf['infer_ms']:.2f} ms). "
        "Likewise, memory has to be measured in a fresh process per model and reported both as total peak "
        "and as delta; otherwise one model's allocations or the import cost get charged to the other.\n")

    add("## 6. Reproducibility\n")
    add("From the repository root, with the virtual environment active, run "
        "`python lab01/src/run_all.py`.\n")
    add("This regenerates every file in `lab01/results/` and this report. All randomness is seeded with 42; "
        "timings and memory depend on the machine and on background load, so small differences between runs "
        "are expected - the medians and the deployment decisions are stable.\n")
    add("| File | Content |")
    add("|---|---|")
    for fname, desc in [
        ("versions.txt", "Python and all package versions vs. pins"),
        ("baseline_accuracy.csv", "test accuracy (4 decimals)"),
        ("split_info.json", "split sizes and class balance"),
        ("hardware.json", "machine used for the measurements"),
        ("cost_time_size.csv", "training time, latency, model size"),
        ("timing_raw.json", "every individual timing run"),
        ("memory.csv / memory_raw.json", "peak memory per model, phase and tool"),
        ("measurements.csv", "combined measurement table"),
        ("deployment_fit.csv", "budget check per model and tier"),
        ("model_footprint.csv", "parameter count and C-export size estimate"),
        ("model.joblib, model_<name>.joblib", "saved models"),
        ("system_cost.png", "Figure 1"),
    ]:
        add(f"| `results/{fname}` | {desc} |")
    add("")

    (LAB_DIR / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {LAB_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
