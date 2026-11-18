"""Step 6: build lab03/report.md from the files in lab03/results/.

Every number is read from results/, and the interpretation sentences are chosen from the measured
values, so the report always matches the run that produced it. A photo of the handwritten FLOP
calculation is embedded if results/hand_calculation_paper.(jpg|jpeg|png) exists.
"""

from __future__ import annotations

import csv
import re

from common import (BATCH_SIZE, EPOCHS, LAB_DIR, LATENCY_RUNS, LATENCY_WARMUP, LR, MODEL_LABELS,
                    MODEL_NAMES, N_TRAIN, N_VAL, RESULTS_DIR, read_json)

L = []
add = L.append


def rows(name):
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def by_model(name):
    return {r["model"]: r for r in rows(name)}


def fnum(v) -> str:
    return f"{int(float(v)):,}"


def photo():
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        if (RESULTS_DIR / f"hand_calculation_paper.{ext}").exists():
            return f"hand_calculation_paper.{ext}"
    return None


def versions():
    text = (RESULTS_DIR / "versions.txt").read_text(encoding="utf-8")
    py = re.search(r"Python\s*:\s*(\S+)", text)
    pr = re.search(r"problems:\s*(\d+)", text)
    return (py.group(1) if py else "?"), (int(pr.group(1)) if pr else -1)


def pareto(models, acc, cost):
    """Models not dominated (another model at least as accurate AND at least as cheap, one strictly)."""
    out = []
    for m in models:
        dominated = any((acc[o] >= acc[m] and cost[o] <= cost[m]) and (acc[o] > acc[m] or cost[o] < cost[m])
                        for o in models if o != m)
        if not dominated:
            out.append(m)
    return out


def main() -> None:
    hw = read_json(RESULTS_DIR / "hardware.json")
    py, problems = versions()
    params = by_model("params.csv")
    fm = by_model("flops_models.csv")
    one = {r["type"]: r for r in rows("flops_one_layer.csv")}
    hand_layers = rows("hand_layers.csv")
    test = by_model("test_accuracy.csv")
    log = rows("training_log.csv")
    lat = {(r["model"], int(r["batch"])): r for r in rows("latency.csv")}
    trade = by_model("tradeoffs.csv")
    pic = photo()

    P = {m: int(params[m]["torchinfo_total_params"]) for m in MODEL_NAMES}
    A = {m: float(test[m]["test_acc"]) for m in MODEL_NAMES}
    F = {m: int(fm[m]["hand_flops"]) for m in MODEL_NAMES}
    L1 = {m: float(lat[(m, 1)]["median_ms"]) for m in MODEL_NAMES}
    L128 = {m: float(lat[(m, 128)]["median_ms"]) for m in MODEL_NAMES}
    lab = MODEL_LABELS

    # ------------------------------------------------------------------------------------------
    add("# Lab 3 - Architectures and their costs\n")
    add("**Course:** Machine Learning (CYI5SE039CY), Group 11  ")
    add("**Reading:** Reddi, *Introduction to Machine Learning Systems*, Ch. 4 DNN Architectures - "
        "§4.2 MLPs, §4.3 CNNs, §4.5 Attention, §4.6 Architectural Building Blocks, §4.7 System-Level "
        "Building Blocks\n")

    add("## 1. Goal\n")
    add("Implement an MLP, a CNN and a small attention (Transformer) model for Fashion-MNIST in PyTorch, "
        "and compare them not only by accuracy but by what they cost: parameters, FLOPs (calculated by hand "
        "and measured with a profiler), and CPU inference latency at batch 1 and batch 128. Then use the "
        "access patterns of Ch. 4.6-4.7 to explain which architecture gives the most accuracy per unit of "
        "cost.\n")

    # ------------------------------------------------------------------------------------------
    add("## 2. Method\n")
    env_ok = py == "3.11.8" and problems == 0
    add("**Environment.** Python 3.11.8 in the repository's pinned `.venv` "
        + ("(all 25 pins `OK` in `results/versions.txt`)" if env_ok else
           f"(**warning:** versions.txt reports Python {py}, {problems} mismatch(es))")
        + f"; PyTorch {hw['torch']}. Everything runs on the CPU ({hw['cpu']}, {hw['logical_cpus']} logical "
          f"CPUs, PyTorch using {hw['torch_threads']} threads"
        + (f", {hw['total_ram_gb']} GB RAM" if "total_ram_gb" in hw else "")
        + f", {hw['os']}). Seeds: `random`, `numpy`, `torch` and `torch.cuda` = 42.\n")
    add(f"**Step 1 - data.** `torchvision.datasets.FashionMNIST` with `ToTensor()` and "
        f"`Normalize((0.2860,), (0.3530,))`. The official 60k training set is split with a seeded "
        f"`random_split` into {N_TRAIN:,} training and {N_VAL:,} validation images; the official 10k test "
        f"set is used only for the final accuracy. Batch size {BATCH_SIZE}. The data is downloaded to "
        "`data/`, which is git-ignored.\n")

    add("**Step 2 - models** (layer sizes exactly as specified):\n")
    add("| Model | Architecture | Parameters (torchinfo) | In 100k-150k target? |")
    add("|---|---|---:|:---:|")
    arch = {
        "mlp": "Flatten 784 -> Linear 256 -> ReLU -> Linear 128 -> ReLU -> Linear 10",
        "cnn": "Conv2d(1,16,3,p=1) -> ReLU -> MaxPool2 -> Conv2d(16,32,3,p=1) -> ReLU -> MaxPool2 -> "
               "Flatten -> Linear(1568,10)",
        "attention": "16 patches of 7x7 -> Linear 49->64 -> 2 x TransformerEncoderLayer (d_model 64, "
                     "4 heads, d_ff 128) -> mean pool -> Linear 64->10",
    }
    for m in MODEL_NAMES:
        add(f"| {lab[m]} | {arch[m]} | {P[m]:,} | {'yes' if params[m]['in_target_100k_150k'] == 'True' else 'no'} |")
    add("")
    outside = [m for m in MODEL_NAMES if params[m]["in_target_100k_150k"] != "True"]
    if outside:
        add("The manual gives both a size target (100k-150k parameters) and exact layer sizes, and the exact "
            "layers do not land in the target: "
            + ", ".join(f"{lab[m]} {P[m]:,}" for m in MODEL_NAMES)
            + ". I kept the specified layers unchanged, because changing widths would change the "
              "architectures being compared. The sizes therefore differ by about "
            f"{max(P.values()) / min(P.values()):.0f}x, which is part of the trade-off analysis below. The "
            "attention model has no positional encoding because none is specified, so after mean pooling it "
            "does not know where each patch was in the image.\n")

    add("**Step 3 - cost.** *Parameters:* `torchinfo.summary` with a 1x1x28x28 input (full tables in "
        "`results/torchinfo_<model>.txt`), checked against a hand count. *FLOPs (per image):* hand "
        "calculation with the Ch. 4 formulas - Linear `2 x in x out`, Conv2d `2 x K x K x C_in x C_out x "
        "H_out x W_out`, attention `2 x seq x d x 3d` (QKV) + `2 x seq^2 x d` (scores) + `2 x seq^2 x d` "
        "(weighted sum) + `2 x seq x d x d` (output projection) + the feed-forward Linears; only the "
        "multiply-adds of matrix products are counted. These are compared with `thop.profile`, which "
        "reports multiply-accumulates (MACs), so thop FLOPs = 2 x MACs. *Latency:* trained models, "
        f"`eval()` + `torch.no_grad()`, real test images, {LATENCY_WARMUP} warm-up calls, then the median "
        f"of {LATENCY_RUNS[1]} runs at batch 1 and of {LATENCY_RUNS[128]} runs at batch 128 "
        "(`time.perf_counter`).\n")
    add(f"**Step 4 - training.** Each model from seed 42, {EPOCHS} epochs, Adam, learning rate {LR:g}, "
        f"batch {BATCH_SIZE}, cross-entropy loss; train and validation metrics are logged every epoch "
        "(`results/training_log.csv`) and the test set is evaluated once at the end.\n")

    # ------------------------------------------------------------------------------------------
    add("## 3. Results\n")
    add("### 3.1 Parameters\n")
    add("| Model | torchinfo total | trainable | hand count | sum of `numel()` |")
    add("|---|---:|---:|---:|---:|")
    for m in MODEL_NAMES:
        r = params[m]
        add(f"| {lab[m]} | {fnum(r['torchinfo_total_params'])} | {fnum(r['torchinfo_trainable_params'])} | "
            f"{fnum(r['hand_params'])} | {fnum(r['sum_numel'])} |")
    add("")
    all_match = all(params[m]["hand_matches_torchinfo"] == "True" for m in MODEL_NAMES)
    add(("The hand counts match torchinfo exactly for all three models. " if all_match else
         "**Note:** the hand count differs from torchinfo for: "
         + ", ".join(lab[m] for m in MODEL_NAMES if params[m]["hand_matches_torchinfo"] != "True") + ". ")
        + "Where the parameters sit:\n")
    add("| Model | Layer | Parameters | FLOPs per image (hand) |")
    add("|---|---|---:|---:|")
    for r in hand_layers:
        add(f"| {lab[r['model']]} | {r['layer']} | {fnum(r['params'])} | {fnum(r['flops'])} |")
    add("")
    add(f"The MLP's first layer alone holds {784 * 256 + 256:,} of its {P['mlp']:,} parameters, because "
        "every one of the 784 pixels is connected to every hidden unit. The CNN's convolutions hold only "
        f"{160 + 4640:,} parameters - a 3x3 kernel is shared across all positions - and most of its "
        "parameters are in the final Linear layer.\n")

    add("### 3.2 FLOPs: hand calculation vs. thop (step 3)\n")
    if pic:
        add("The hand calculation was first done on paper:\n")
        add(f"![Handwritten FLOP calculation](results/{pic})\n")
        add("*Figure 1 - Handwritten parameter and FLOP calculation.*\n")
    add("**One layer of each type** (FLOPs for one image):\n")
    add("| Type | Layer | Hand formula | Hand FLOPs | thop default (2 x MACs) | Difference | "
        "thop, no fast path | thop, no fast path + MHA rule |")
    add("|---|---|---|---:|---:|---:|---:|---:|")
    for typ in ("Linear", "Conv2d", "Attention"):
        r = one[typ]
        th = r.get("thop_flops_default", "")
        diff = r.get("thop_default_minus_hand", "")
        add(f"| {typ} | {r['layer']} | `{r['hand_formula']}` | {fnum(r['hand_flops'])} | "
            f"{fnum(th) if th not in ('', None) else 'not reported'} | "
            f"{(('+' if float(diff) > 0 else '') + fnum(diff)) if diff not in ('', None) else '-'} | "
            f"{fnum(r['thop_flops_no_fastpath']) if r.get('thop_flops_no_fastpath') else '-'} | "
            f"{fnum(r['thop_flops_no_fastpath_plus_mha_rule']) if r.get('thop_flops_no_fastpath_plus_mha_rule') else '-'} |")
    add("")
    att = one["Attention"]
    add("Attention layer by hand (seq = 16 tokens, d = 64, d_ff = 128): QKV projections 2·16·64·192 = "
        "393,216; scores QK^T 2·16²·64 = 32,768; weighted sum AV 2·16²·64 = 32,768; output projection "
        "2·16·64·64 = 131,072; feed-forward 2·16·64·128 + 2·16·128·64 = 524,288; total "
        f"{fnum(att['hand_flops'])} FLOPs. Because the sequence is short (16 tokens), the seq² terms are only "
        "6% of the layer; the projections dominate.\n")

    add("**Whole models:**\n")
    add("| Model | Hand FLOPs | thop default (2 x MACs) | thop / hand | torchinfo mult-adds x 2 |")
    add("|---|---:|---:|---:|---:|")
    for m in MODEL_NAMES:
        r = fm[m]
        add(f"| {lab[m]} | {fnum(r['hand_flops'])} | {fnum(r['thop_flops_2x_macs'])} | {float(r['thop_over_hand']):.3f} | "
            f"{fnum(r['torchinfo_flops_2x'])} |")
    a = fm["attention"]
    add("")
    add("| Attention model, thop variant | FLOPs (2 x MACs) | / hand |")
    add("|---|---:|---:|")
    add(f"| default (eval mode, PyTorch encoder fast path) | {fnum(a['thop_flops_2x_macs'])} | {float(a['thop_over_hand']):.3f} |")
    add(f"| regular per-module path (fast path disabled) | {fnum(a['thop_flops_no_fastpath'])} | {float(a['no_fastpath_over_hand']):.3f} |")
    add(f"| + custom thop rule for `nn.MultiheadAttention` | {fnum(a['thop_flops_no_fastpath_plus_mha_rule'])} | {float(a['plus_mha_rule_over_hand']):.3f} |")
    add("")

    # gap explanation, driven by the numbers
    lin, conv = one["Linear"], one["Conv2d"]
    expl = ["**Explaining the gaps.** (1) thop counts multiply-accumulates; one MAC is two FLOPs, so every "
            "thop number is doubled before comparing."]
    if lin.get("thop_default_minus_hand") not in ("", None) and int(float(lin["thop_default_minus_hand"])) == 0:
        expl.append("(2) For the Linear layer thop and the hand formula agree exactly: both count "
                    "in x out multiply-adds and ignore the bias.")
    else:
        expl.append(f"(2) For the Linear layer the difference is {lin.get('thop_default_minus_hand')} FLOPs.")
    cd = conv.get("thop_default_minus_hand")
    bias_ops = 2 * 32 * 14 * 14
    if cd not in ("", None) and int(float(cd)) == 0:
        expl.append("(3) For the convolution thop also agrees exactly: this thop version counts "
                    "K x K x C_in multiply-adds per output value and, like the Ch. 4 formula, ignores the bias.")
    elif cd not in ("", None) and int(float(cd)) == bias_ops:
        expl.append(f"(3) For the convolution thop is higher by exactly {bias_ops:,} = 2 x 32 x 14 x 14: thop "
                    "adds one operation per output value for the bias, which the Ch. 4 formula leaves out.")
    elif cd not in ("", None):
        expl.append(f"(3) For the convolution thop differs by {fnum(cd)} FLOPs "
                    f"({float(cd) / float(conv['hand_flops']):+.1%}); thop also counts the bias additions.")
    ad = float(a["thop_over_hand"])
    ns = float(a["no_fastpath_over_hand"])
    if abs(ad - ns) < 0.005:
        expl.append(f"(4) For the attention model the default thop count is {ad:.0%} of the hand count; here "
                    "disabling PyTorch's fused encoder fast path did not change it.")
    elif ad < 0.5:
        expl.append(f"(4) For the attention model the default thop count is only {ad:.0%} of the hand count. "
                    "In `eval()` mode PyTorch runs each `TransformerEncoderLayer` as one fused kernel "
                    "(the encoder *fast path*), so the forward hooks that thop puts on `Linear` layers never "
                    "fire and the encoder is invisible; only the patch projection and the head are counted.")
    else:
        expl.append(f"(4) For the attention model the default thop count is {ad:.0%} of the hand count.")
    expl.append(f"With the fast path disabled thop sees the feed-forward Linears and reaches {ns:.0%}, but it "
                "still misses the attention block itself: thop has no rule for `nn.MultiheadAttention`, and "
                "that module multiplies by its weight tensors directly instead of calling `Linear` "
                "sub-modules, so the QKV projections, QK^T, AV and the output projection go uncounted.")
    rr = float(a["plus_mha_rule_over_hand"])
    thop_layers = read_json(RESULTS_DIR / "thop_layers.json")
    ln_flops = int(2 * sum(v for k, v in thop_layers["attention"].get("no_fastpath_plus_mha_rule", {}).items()
                           if k.endswith((".norm1", ".norm2"))))
    rest = int(a["thop_flops_no_fastpath_plus_mha_rule"]) - int(a["hand_flops"])
    expl.append(f"Adding a custom thop rule for `nn.MultiheadAttention` closes the gap to {rr:.3f} x the hand "
                f"count. The remaining {rest:+,} FLOPs are "
                + (f"exactly the LayerNorm operations that thop counts ({ln_flops:,} FLOPs for the four "
                   "LayerNorms) and the Ch. 4 formula leaves out." if ln_flops == rest else
                   f"operations the hand formula deliberately ignores (thop counts {ln_flops:,} FLOPs for "
                   "LayerNorm alone).") if abs(rr - 1) > 0.0005 else
                f"Adding a custom thop rule for `nn.MultiheadAttention` closes the gap completely ({rr:.3f}).")
    ti = {m: int(fm[m]["torchinfo_flops_2x"]) / int(fm[m]["hand_flops"]) for m in MODEL_NAMES}
    expl.append("torchinfo's mult-adds estimate behaves similarly: "
                + ", ".join(f"{lab[m]} {ti[m]:.3f} x hand" for m in MODEL_NAMES)
                + " - it adds the bias terms (for the MLP its mult-adds equal the parameter count) and, like "
                  "thop, cannot see inside the fused encoder.")
    expl.append("Lesson: a profiler only counts the operations it has rules for and actually observes; fused "
                "kernels and unsupported modules make it under-count, so a hand calculation is the "
                "reference.")
    add(" ".join(expl) + "\n")

    add("### 3.3 Inference latency (step 3)\n")
    add(f"| Model | Batch 1: median ms ({LATENCY_RUNS[1]} runs) | Batch 128: median ms ({LATENCY_RUNS[128]} runs) | "
        "Batch 128: ms per image | Speed-up per image from batching | Images/s at batch 128 |")
    add("|---|---:|---:|---:|---:|---:|")
    for m in MODEL_NAMES:
        r128 = lat[(m, 128)]
        add(f"| {lab[m]} | {L1[m]:.4f} | {L128[m]:.3f} | {L128[m] / 128:.5f} | "
            f"{L1[m] / (L128[m] / 128):.1f}x | {fnum(r128['images_per_second'])} |")
    add("")
    add("Raw runs: `results/latency_raw.json` (p95/min/max in `results/latency.csv`).\n")

    add("### 3.4 Training and test accuracy (step 4)\n")
    add("| Model | " + " | ".join(f"val acc ep {e}" for e in range(1, EPOCHS + 1)) + " | **Test accuracy** | Train time (s) |")
    add("|---|" + "---:|" * (EPOCHS + 2))
    for m in MODEL_NAMES:
        vals = [r["val_acc"] for r in log if r["model"] == m]
        add(f"| {lab[m]} | " + " | ".join(vals) + f" | **{test[m]['test_acc']}** | {test[m]['train_seconds']} |")
    add("")

    add("### 3.5 Trade-offs (step 5)\n")
    add("![Accuracy vs parameters](results/accuracy_vs_params.png)\n")
    add("*Figure 2 - Test accuracy against parameter count (`results/accuracy_vs_params.png`).*\n")
    add("![Accuracy vs latency](results/accuracy_vs_latency.png)\n")
    add("*Figure 3 - Test accuracy against median CPU latency at batch 1 and batch 128 "
        "(`results/accuracy_vs_latency.png`).*\n")
    add("| Model | Test acc. | Params | FLOPs / image | FLOPs per parameter | Latency b1 (ms) | "
        "Latency b128 (ms/image) |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for m in MODEL_NAMES:
        t = trade[m]
        add(f"| {lab[m]} | {A[m]:.4f} | {P[m]:,} | {F[m]:,} | {float(t['flops_per_param']):.1f} | "
            f"{L1[m]:.4f} | {L128[m] / 128:.5f} |")
    add("")

    best_acc = max(MODEL_NAMES, key=lambda m: A[m])
    fewest = min(MODEL_NAMES, key=lambda m: P[m])
    fastest1 = min(MODEL_NAMES, key=lambda m: L1[m])
    fastest128 = min(MODEL_NAMES, key=lambda m: L128[m])
    par_p = pareto(MODEL_NAMES, A, P)
    par_1 = pareto(MODEL_NAMES, A, L1)
    par_128 = pareto(MODEL_NAMES, A, L128)
    fpp = {m: F[m] / P[m] for m in MODEL_NAMES}

    add("**Access patterns (Ch. 4.6-4.7).** "
        f"*MLP - dense, all-to-all.* Every weight is read once per image and used for a single multiply-add: "
        f"{fpp['mlp']:.1f} FLOPs per parameter. At batch 1 that is about {F['mlp'] / (4 * P['mlp']):.1f} FLOP per "
        f"byte of fp32 weights, so the layer is limited by streaming {P['mlp'] * 4 / 1024:.0f} KB of weights "
        "from memory, not by arithmetic; batching reuses each weight for 128 images and turns the matmuls "
        f"compute-bound (per-image speed-up {L1['mlp'] / (L128['mlp'] / 128):.0f}x). "
        f"*CNN - local, sliding window, shared weights.* Each 3x3 kernel is reused at every position, giving "
        f"{fpp['cnn']:.0f} FLOPs per parameter: few weights, many operations, high arithmetic intensity and "
        "neighbouring-pixel (cache-friendly) access. Its cost is compute, not memory"
        + (f" - which is why batching helps it only {L1['cnn'] / (L128['cnn'] / 128):.1f}x per image versus "
           f"{L1['mlp'] / (L128['mlp'] / 128):.0f}x for the memory-bound MLP. "
           if L1['cnn'] / (L128['cnn'] / 128) < L1['mlp'] / (L128['mlp'] / 128) else ". ")
        + f"*Attention - global, data-dependent.* Every token attends to every other token (seq² scores), the "
        f"QKV and output projections run per token, and each layer adds softmax, LayerNorm, reshapes and "
        f"residual adds - many small, memory-bound steps between the matmuls ({fpp['attention']:.0f} FLOPs per "
        "parameter). With only 16 tokens the quadratic term is cheap, but the many small operations add "
        "fixed overhead per call.\n")

    verdict = []
    verdict.append(f"**Which is most cost-effective?** Highest test accuracy: {lab[best_acc]} ({A[best_acc]:.4f}). "
                   f"Fewest parameters: {lab[fewest]} ({P[fewest]:,}). Fastest at batch 1: {lab[fastest1]} "
                   f"({L1[fastest1]:.4f} ms); fastest at batch 128: {lab[fastest128]} ({L128[fastest128]:.2f} ms).")
    verdict.append("Pareto-optimal models (no other model is both at least as accurate and at least as cheap): "
                   f"vs. parameters {', '.join(lab[m] for m in par_p)}; vs. batch-1 latency "
                   f"{', '.join(lab[m] for m in par_1)}; vs. batch-128 latency {', '.join(lab[m] for m in par_128)}.")
    if best_acc == fewest:
        winner = best_acc
        verdict.append(f"The **{lab[winner]}** is the most cost-effective architecture here: it is the most accurate "
                       "model and also the smallest. Weight sharing gives it the right inductive bias for images "
                       "(local patterns that can appear anywhere) and lets it spend compute instead of memory: "
                       f"{lab[winner]} needs {P['mlp'] / P[winner]:.0f}x fewer parameters than the MLP.")
        if fastest1 != winner:
            verdict.append(f"Its price is latency: it performs {F[winner] / F['mlp']:.1f}x the MLP's FLOPs, so the "
                           f"{lab[fastest1]} answers a single request faster ({L1[fastest1]:.4f} vs "
                           f"{L1[winner]:.4f} ms). If memory is the constraint (edge/mobile), choose the "
                           f"{lab[winner]}; if only single-image latency on a CPU matters, the {lab[fastest1]} "
                           f"is cheaper at a cost of {(A[winner] - A[fastest1]) * 100:.1f} accuracy points.")
    else:
        winner = max(MODEL_NAMES, key=lambda m: A[m] / P[m])
        verdict.append(f"Per parameter the **{lab[winner]}** is the most cost-effective "
                       f"({float(trade[winner]['acc_per_100k_params']):.2f} accuracy per 100k parameters), "
                       f"while {lab[best_acc]} is the most accurate.")
    att_rank = sorted(MODEL_NAMES, key=lambda m: A[m]).index("attention")
    if att_rank == 0:
        slow1 = max(MODEL_NAMES, key=lambda m: L1[m]) == "attention"
        verdict.append(f"The attention model is the least accurate ({A['attention']:.4f}) although it does the "
                       f"most FLOPs ({F['attention']:,} per image): without positional encoding and with only 5 "
                       "epochs, its global mixing gives no advantage on small, centred images"
                       + (", and its many small operations make it the slowest model at batch 1." if slow1 else "."))
    add(" ".join(verdict) + "\n")

    # ------------------------------------------------------------------------------------------
    add("## 4. Conclusions\n")
    add(f"1. **Parameters, FLOPs and latency measure different things.** The MLP has the most parameters "
        f"({P['mlp']:,}) but the fewest FLOPs ({F['mlp']:,} per image); the CNN has the fewest parameters "
        f"({P['cnn']:,}) but {F['cnn'] / F['mlp']:.1f}x the MLP's FLOPs. Which one is \"small\" depends on whether "
        "memory or compute is the limit, so all three numbers must be reported.\n")
    lin_ok = lin.get("thop_default_minus_hand") not in ("", None) and int(float(lin["thop_default_minus_hand"])) == 0
    conv_bias = cd not in ("", None) and int(float(cd)) == bias_ops
    c2 = "2. **Profilers under-count unless you check them by hand.** thop "
    c2 += ("matched the hand formula exactly for the Linear layer" if lin_ok else
           "was close to the hand formula for the Linear layer")
    conv_exact = cd not in ("", None) and int(float(cd)) == 0
    c2 += (" and the convolution, " if conv_exact else
           " and differed only by the bias term for the convolution, " if conv_bias else
           " and was close for the convolution, ")
    c2 += (f"but saw only {ad:.0%} of the attention model's FLOPs by default"
           + (" (fused encoder fast path)" if ad < ns - 0.005 else "")
           + f" and {ns:.0%} with the regular path, because it has no rule for `MultiheadAttention`; a custom rule "
             f"was needed to reach {rr:.2f} x the hand count.\n")
    add(c2)
    add(f"3. **{lab[best_acc] if best_acc == fewest else lab[winner]} is the most cost-effective architecture for this "
        f"task.** " + (f"It reached the best test accuracy ({A[best_acc]:.4f}) with the fewest parameters "
                       f"({P[best_acc]:,}), because shared local kernels match the structure of images; "
                       if best_acc == fewest else
                       f"It gives the most accuracy per parameter; ")
        + (f"the {lab[fastest1]} is the fastest choice at batch 1 when memory is not the constraint" )
        + (", and the attention model gives the lowest accuracy for the most FLOPs.\n" if att_rank == 0 else ".\n"))

    add("## 5. Reproducibility\n")
    add("From the repository root with the virtual environment active, run `python lab03/src/run_all.py` "
        "(it downloads Fashion-MNIST to `data/` on first use). It regenerates every file in `lab03/results/` "
        "and this report; the photo of the handwritten page is kept. Seeds are fixed to 42; latencies depend "
        "on the machine and its load.\n")
    add("| File | Content |")
    add("|---|---|")
    for fname, desc in [
        ("versions.txt / hardware.json", "environment and machine"),
        ("torchinfo_mlp.txt, torchinfo_cnn.txt, torchinfo_attention.txt", "torchinfo summaries"),
        ("params.csv", "parameters: torchinfo vs hand"),
        ("hand_layers.csv", "hand parameters and FLOPs of every layer"),
        ("flops_one_layer.csv / flops_models.csv / thop_layers.json", "hand vs thop FLOPs"),
        ("hand_calculation_paper.jpg", "handwritten calculation"),
        ("latency.csv / latency_raw.json", "inference latency, batch 1 and 128"),
        ("training_log.csv / test_accuracy.csv", "per-epoch log and test accuracy"),
        ("model_mlp.pt, model_cnn.pt, model_attention.pt", "trained weights"),
        ("accuracy_vs_params.png / accuracy_vs_latency.png / tradeoffs.csv", "trade-off plots and table"),
    ]:
        add(f"| `results/{fname}` | {desc} |")
    add("")

    (LAB_DIR / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {LAB_DIR / 'report.md'}" + ("" if pic else "  (note: hand_calculation_paper photo not found yet)"))


if __name__ == "__main__":
    main()
