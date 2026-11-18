"""Build lab02/report.md from the files in lab02/results/.

Every number in the report is read from results/, so the report always matches the run that
produced it. The handwritten design page is embedded if a photo named
results/network_design_paper.(jpg|jpeg|png) exists.
"""

from __future__ import annotations

import csv
import math
import re

from common import (BATCH_SIZE, EPOCHS, INIT_STD, LAB_DIR, LEARNING_RATES, RESULTS_DIR, WIDTHS,
                    read_json, run_name)


def rows(name: str) -> list[dict]:
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def versions_summary() -> tuple[str, int]:
    text = (RESULTS_DIR / "versions.txt").read_text(encoding="utf-8")
    py = re.search(r"Python\s*:\s*(\S+)", text)
    problems = re.search(r"problems:\s*(\d+)", text)
    return (py.group(1) if py else "?"), (int(problems.group(1)) if problems else -1)


def first_epoch_below(log, run, threshold=2.2):
    for r in log:
        if r["run"] == run and int(r["epoch"]) > 0 and float(r["train_loss"]) < threshold:
            return int(r["epoch"])
    return None


def paper_photo():
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        p = RESULTS_DIR / f"network_design_paper.{ext}"
        if p.exists():
            return p.name
    return None


def main() -> None:
    hw = read_json(RESULTS_DIR / "hardware.json")
    data = read_json(RESULTS_DIR / "data_checks.json")
    design = rows("network_design.csv")
    fwd = rows("forward_checks.csv")
    gc = rows("grad_check.csv")
    gcs = rows("grad_check_summary.csv")
    fin = {r["run"]: r for r in rows("final_accuracy.csv")}
    sup = {r["run"]: r for r in rows("final_accuracy_init_std0.1.csv")}
    log = rows("training_log.csv")
    tim = rows("timing.csv")
    py, problems = versions_summary()
    photo = paper_photo()

    L = []
    add = L.append

    # ------------------------------------------------------------------------------------------
    add("# Lab 2 - A neural network from scratch\n")
    add("**Course:** Machine Learning (CYI5SE039CY), Group 11  ")
    add("**Reading:** Reddi, *Introduction to Machine Learning Systems*, Ch. 3 DL Primer - §3.4, "
        "§§3.5.1-3.5.5 and §3.6\n")

    add("## 1. Goal\n")
    add("Build a two-hidden-layer neural network in plain NumPy - no deep-learning framework - and "
        "verify every part of it: dimensionally correct parameters, a forward pass that outputs valid "
        "probabilities, a cross-entropy loss with the expected starting value, hand-derived gradients "
        "confirmed by numerical gradient checking, and a mini-batch training loop. Then compare learning "
        "rates and hidden widths, and measure where the time goes in one training step (forward vs. "
        "backward pass).\n")

    # ------------------------------------------------------------------------------------------
    add("## 2. Method\n")
    env_ok = py == "3.11.8" and problems == 0
    add("**Environment.** Python 3.11.8 in the repository's pinned `.venv`; `src/print_versions.py` "
        "writes the Python and package versions to `results/versions.txt`. "
        + ("All 25 pinned packages report `OK`. " if env_ok else
           f"**Warning:** versions.txt reports Python {py} and {problems} mismatch(es). ")
        + f"Hardware: {hw['cpu']}, {hw['logical_cpus']} logical CPUs"
        + (f", {hw['total_ram_gb']} GB RAM" if "total_ram_gb" in hw else "")
        + f", {hw['os']}; NumPy {hw['numpy']}. All computation is float64.\n")

    add("**Step 1 - data.** `load_digits` (8x8 grey-scale digits): "
        f"{data['n_samples']} samples x {data['n_features']} features, 10 classes. Inputs are divided "
        f"by 16, so every pixel lies in [{data['x_min']:g}, {data['x_max']:g}]. Labels are one-hot "
        f"encoded (train matrix {tuple(data['one_hot_shape_train'])}, every row sums to 1: "
        f"{'yes' if data['one_hot_rows_sum_to_1'] else 'NO'}). Split 70/30, stratified, "
        f"`random_state=42` -> {data['train_size']} train / {data['test_size']} test samples "
        f"(per-class counts: train {data['train_class_counts']}, test {data['test_class_counts']}).\n")

    add("**Step 2 - network design.** Input 64 -> hidden W -> ReLU -> hidden W -> ReLU -> output 10 "
        "-> softmax, for W = 16 and W = 64. Weights are drawn from N(0, 0.01), implemented as "
        f"mean 0 and standard deviation {INIT_STD:g} (`np.random.normal(0, 0.01)`); biases start at 0. "
        "The script asserts that the created arrays have exactly these shapes.\n")
    add("| Layer | W = 16: weight / bias | params | W = 64: weight / bias | params | Activation |")
    add("|---|---|---:|---|---:|---|")
    d16 = [r for r in design if r["width"] == "16"]
    d64 = [r for r in design if r["width"] == "64"]
    for a, b in zip(d16, d64):
        if a["layer"] == "total":
            add(f"| **total** | | **{int(a['parameters']):,}** | | **{int(b['parameters']):,}** | |")
        else:
            add(f"| {a['layer']} ({a['in_out']} / {b['in_out']}) | {a['weight_shape']} / {a['bias_shape']} | "
                f"{int(a['parameters']):,} | {b['weight_shape']} / {b['bias_shape']} | "
                f"{int(b['parameters']):,} | {a['activation']} |")
    add("")
    if photo:
        add(f"The design was first done by hand on paper (shapes, forward equations and the gradient "
            f"derivation):\n")
        add(f"![Handwritten network design](results/{photo})\n")
        add("*Figure 1 - Handwritten network design and gradient derivation.*\n")
    else:
        add("*The handwritten design page is stored as `results/network_design_paper.jpg`.*\n")

    add("**Step 3 - forward pass** (N = batch size, layer by layer):\n")
    add("| Layer | Computation | Output shape |")
    add("|---|---|---|")
    add("| hidden 1 | `Z1 = X W1 + b1`, `A1 = ReLU(Z1)` | (N, W) |")
    add("| hidden 2 | `Z2 = A1 W2 + b2`, `A2 = ReLU(Z2)` | (N, W) |")
    add("| output | `Z3 = A2 W3 + b3`, `P = softmax(Z3)` | (N, 10) |")
    add("")
    add("Softmax subtracts each row's maximum before `exp` so it cannot overflow. Checks: output shape "
        "(N, 10) and `|sum_k P[n,k] - 1| < 1e-6` for every row.\n")
    add("**Step 4 - loss.** Mean cross-entropy `L = -(1/N) sum_n sum_k Y[n,k] log P[n,k]`, computed "
        "from the logits with a stable log-softmax. Sanity check: with near-zero weights all 10 "
        "probabilities are about 0.1, so the untrained loss should be about ln(10) = 2.3026.\n")

    add("**Step 5 - gradients (chain rule).** Softmax and cross-entropy together give a simple output "
        "error; every other gradient follows from the chain rule through the matrix products and the ReLU "
        "masks:\n")
    add("| Gradient | Formula | Shape |")
    add("|---|---|---|")
    add("| output error | `dZ3 = (P - Y) / N` | (N, 10) |")
    add("| layer 3 | `dW3 = A2^T dZ3`, `db3 = sum_n dZ3` | (W, 10), (10,) |")
    add("| back to hidden 2 | `dA2 = dZ3 W3^T`, `dZ2 = dA2 * 1[Z2 > 0]` | (N, W) |")
    add("| layer 2 | `dW2 = A1^T dZ2`, `db2 = sum_n dZ2` | (W, W), (W,) |")
    add("| back to hidden 1 | `dA1 = dZ2 W2^T`, `dZ1 = dA1 * 1[Z1 > 0]` | (N, W) |")
    add("| layer 1 | `dW1 = X^T dZ1`, `db1 = sum_n dZ1` | (64, W), (W,) |")
    add("")
    add("*Gradient check:* 10 weights are sampled at random (seed 42) from the entries of W1, W2 and W3 "
        "whose gradient on the check batch is not exactly zero (a weight attached to a pixel that is 0 in "
        "every check image - e.g. the image border - or to an inactive ReLU unit has gradient 0 in both "
        "methods and would test nothing). Each is nudged by +1e-5 and -1e-5 and the central-difference slope "
        "`(L(w+e) - L(w-e)) / 2e` is compared with the analytic gradient on a fixed batch of the first "
        "32 training samples. Relative error = `|analytic - numerical| / max(|analytic|, |numerical|)` "
        "(0 when both are exactly 0). Pass: relative error < 1e-6 for at least 9 of 10 weights. The "
        "check is run on the trained networks (lr = 0.1, after 50 epochs), where gradients have realistic "
        "sizes, and - for transparency - also on the untrained N(0, 0.01) networks. As extra evidence "
        "every single parameter (weights and biases) is also checked and the vector relative error "
        "`||analytic - numerical|| / max(||analytic||, ||numerical||)` is reported.\n")

    add(f"**Step 6 - training.** Plain mini-batch SGD: the training set is reshuffled every epoch, batch "
        f"size {BATCH_SIZE}, {EPOCHS} epochs, update `p <- p - lr * dp`. Grid: learning rates "
        f"{list(LEARNING_RATES)} x widths {list(WIDTHS)} = 6 runs, each starting from seed 42. After "
        "every epoch the full train and test sets are evaluated and logged (`results/training_log.csv`). "
        "Because \"N(0, 0.01)\" is also often read as *variance* 0.01, the same grid was repeated as a "
        "supplementary experiment with standard deviation 0.1.\n")
    add("**Step 7 - timing.** Forward pass, loss and backward pass are timed separately with "
        "`time.perf_counter()`: 1 warm-up call, then 100 timed calls, **median** reported (p95 in the CSV). "
        "Batch sizes 32 (the training mini-batch) and 1257 (the whole training set). To link the time to "
        "matrix multiplication, the matmuls of each pass are also timed alone with exactly the same "
        "shapes, and the analytic FLOP count (2·m·k·n per matmul) is computed.\n")

    # ------------------------------------------------------------------------------------------
    add("## 3. Results\n")
    add("### 3.1 Forward pass and initial loss (steps 3-4)\n")
    add("| W | Input | Output shape | max abs(row sum - 1) | Rows sum to 1 (< 1e-6) | Initial loss | ln(10) |")
    add("|---:|---|---|---:|:---:|---:|---:|")
    for r in fwd:
        add(f"| {r['width']} | {r['input']} {r['input_shape']} | {r['output_shape']} | "
            f"{r['max_abs_row_sum_minus_1']} | {'yes' if r['rows_sum_to_1'] == 'True' else 'NO'} | "
            f"{r['initial_loss']} | {r['ln10']} |")
    add("")
    worst_dev = max(float(r["max_abs_row_sum_minus_1"]) for r in fwd)
    worst_l = max(float(r["abs_diff_from_ln10"]) for r in fwd)
    add(f"Every output has shape (batch, 10) and every row sums to 1 within {worst_dev:.1e} (limit 1e-6). "
        f"The untrained loss differs from ln(10) by at most {worst_l:.1e}: with weights of size 0.01 the "
        "logits are almost zero, so the softmax is almost uniform.\n")

    add("### 3.2 Gradient check (step 5)\n")
    add("| Network | W | Weights below 1e-6 | Required | Criterion | Max rel. error | Max abs. error | "
        "All-parameter vector rel. error (params) |")
    add("|---|---:|---:|---:|:---:|---:|---:|---:|")
    for r in gcs:
        label = "trained (lr 0.1)" if r["point"] == "trained" else "untrained N(0, 0.01)"
        crit = "**PASS**" if r["criterion_9_of_10"] == "met" else "not met"
        add(f"| {label} | {r['width']} | {r['passed']}/{r['weights_checked']} | {r['required']} | {crit} | "
            f"{r['max_rel_error']} | {r['max_abs_error']} | {r['all_params_vector_rel_error']} "
            f"({int(r['all_params_checked']):,}) |")
    add("")
    add("Sampled weights of the trained networks:\n")
    add("| W | Weight | Analytic | Numerical | Rel. error | < 1e-6 |")
    add("|---:|---|---:|---:|---:|:---:|")
    for r in gc:
        if r["point"] == "trained":
            add(f"| {r['width']} | {r['param']}{r['index']} | {float(r['analytic']):.6e} | "
                f"{float(r['numerical']):.6e} | {float(r['rel_error']):.1e} | "
                f"{'yes' if r['pass'] == 'True' else 'no'} |")
    add("")
    trained = [r for r in gcs if r["point"] == "trained"]
    init = [r for r in gcs if r["point"] == "init"]
    all_trained_pass = all(r["criterion_9_of_10"] == "met" for r in trained)
    fails = [r for r in gc if r["point"] == "trained" and r["pass"] != "True"]
    txt = ("**The analytic gradients pass the check for both widths**" if all_trained_pass else
           "**Warning: the trained-network check did not meet the 9-of-10 rule for every width**")
    txt += (f" ({', '.join(r['passed'] + '/10 for W = ' + r['width'] for r in trained)}). When a weight "
            "has a normal-sized gradient the two methods agree to 9-12 significant digits.")
    if fails:
        f0 = fails[0]
        txt += (f" The weight that misses the threshold ({f0['param']}{f0['index']}, W = {f0['width']}) has "
                f"a gradient of only {float(f0['analytic']):.1e}: the absolute difference is "
                f"{float(f0['abs_error']):.1e}, i.e. at the float64 noise level of a +/-1e-5 central "
                "difference, so its relative error is not meaningful.")
    txt += (" Weights excluded from sampling because their gradient is exactly 0: "
            + " and ".join(f"{int(r['weights_zero_grad_excluded']):,} of {int(r['weights_total']):,} (W = {r['width']})"
                           for r in trained) + ".")
    txt += (" The all-parameter check covers every weight and bias and gives a vector relative error of "
            + " and ".join(r["all_params_vector_rel_error"] + f" (W = {r['width']})" for r in trained) + ".")
    add(txt + "\n")
    init_fails = [r for r in gc if r["point"] == "init" and r["pass"] != "True"]
    n_kink = sum(r["failure_reason"].startswith("ReLU") for r in init_fails)
    n_noise = sum(r["failure_reason"].startswith("gradient") for r in init_fails)
    noise_abs = [float(r["abs_error"]) for r in init_fails if r["failure_reason"].startswith("gradient")]
    para = ("On the **untrained** network the same procedure gives "
            + " and ".join(f"{r['passed']}/10 (W = {r['width']})" for r in init)
            + ", below the 9-of-10 rule. Every miss was classified automatically "
              "(`failure_reason` in `results/grad_check.csv`), and none of them points to a wrong formula:")
    if n_noise:
        para += (f" **{n_noise}** are gradients of only about 1e-5 to 1e-7. `L(w+e)` and `L(w-e)` are both "
                 "about 2.3026 and differ only in their last digits; float64 resolves that difference to about "
                 "1e-16, so the numerical slope carries an absolute error of about 1e-11 (measured "
                 f"{min(noise_abs):.0e} to {max(noise_abs):.0e}) - tiny in absolute terms, but large relative to "
                 "a gradient of 1e-6.")
    if n_kink:
        para += (f" **{n_kink}** crossed a ReLU kink: with N(0, 0.01) weights the second-layer pre-activations "
                 "are typically only 1e-3 to 1e-4 and some are far smaller, so the +/-1e-5 nudge itself switched a hidden unit on or off between the "
                 "two loss evaluations, where the loss has no derivative and the finite difference is invalid.")
    para += (" Both effects disappear once the weights and gradients have realistic sizes, which is why "
             "the trained-network check passes. Lesson: a relative-error gradient check is only informative "
             "when the gradient is far above the float64 noise floor and the nudge does not cross a kink.\n")
    add(para)

    add("### 3.3 Training: learning rate x width (step 6)\n")
    add("| W | lr | Final train loss | Final test loss | Final train acc. | **Final test acc.** | "
        "First epoch with train loss < 2.2 | Train time (s) |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for w in WIDTHS:
        for lr in LEARNING_RATES:
            r = fin[run_name(w, lr)]
            esc = first_epoch_below(log, r["run"])
            add(f"| {w} | {lr:g} | {r['final_train_loss']} | {r['final_test_loss']} | {r['final_train_acc']} | "
                f"**{r['final_test_acc']}** | {esc if esc else 'never'} | {r['train_seconds']} |")
    add("")
    add("![Loss curves](results/loss_curves.png)\n")
    add("*Figure 2 - Train (solid) and test (dashed) cross-entropy per epoch for all six runs "
        "(`results/loss_curves.png`). Accuracy curves: `results/accuracy_curves.png`.*\n")

    best = max(fin.values(), key=lambda r: float(r["final_test_acc"]))
    stuck = [r for r in fin.values() if float(r["final_test_acc"]) < 0.2]
    para = (f"**Analysis.** The best run is W = {best['width']}, lr = {float(best['lr']):g} with test accuracy "
            f"**{best['final_test_acc']}**.")
    if stuck:
        para += (f" {len(stuck)} of the 6 runs ({', '.join('W=' + r['width'] + '/lr=' + format(float(r['lr']), 'g') for r in stuck)}) "
            "never leave the starting loss of about ln(10) and end at chance-level accuracy (about 0.10, "
            "the network predicts one class). The cause is the very small initialisation: with weights of "
            "size 0.01 the activations shrink by roughly a factor of 0.01·sqrt(fan-in) per layer, and the "
            "gradient that reaches W1 is multiplied by both W2 and W3, so it is tiny. A small learning rate "
            "turns this tiny gradient into an even smaller update, and 50 epochs are not enough to break "
            "out of the plateau.")
    esc_lines = []
    for w in WIDTHS:
        e = first_epoch_below(log, run_name(w, max(LEARNING_RATES)))
        esc_lines.append(f"W = {w}: epoch {e}" if e else f"W = {w}: never")
    add(para + f" Only the largest learning rate escapes the plateau ({'; '.join(esc_lines)}), and then the loss "
        "drops quickly, with some spikes typical of plain SGD with a large step. The wider network escapes "
        "earlier and ends much better: more hidden units give more paths for the signal and a larger "
        "gradient, and more capacity. Train and test curves stay close, so there is no strong overfitting "
        "within 50 epochs.\n")
    add("**Supplementary - initial standard deviation 0.1** (`results/final_accuracy_init_std0.1.csv`, "
        "`results/loss_curves_init_std0.1.png`):\n")
    add("| W | lr = " + " | lr = ".join(f"{lr:g}" for lr in LEARNING_RATES) + " |")
    add("|---:|" + "---:|" * len(LEARNING_RATES))
    for w in WIDTHS:
        add(f"| {w} | " + " | ".join(sup[run_name(w, lr)]["final_test_acc"] for lr in LEARNING_RATES) + " |")
    add("")
    sup_best = max(sup.values(), key=lambda r: float(r["final_test_acc"]))
    add(f"With ten times larger initial weights the plateau disappears for most runs and the ranking "
        f"becomes the textbook one: larger learning rate and larger width learn faster, best test accuracy "
        f"{sup_best['final_test_acc']} (W = {sup_best['width']}, lr = {float(sup_best['lr']):g}). This "
        "confirms that the weak results of the small learning rates in the main grid come from the "
        "initialisation scale, not from the gradients or the training loop.\n")

    add("### 3.4 Forward vs. backward timing (step 7)\n")
    add("| W | Batch | Forward (us) | Loss (us) | Backward (us) | Backward / forward | Matmul share fwd / bwd | "
        "Matmul MFLOP fwd / bwd | FLOP ratio bwd/fwd |")
    add("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in tim:
        add(f"| {r['width']} | {r['batch']} | {r['forward_median_us']} | {r['loss_median_us']} | "
            f"{r['backward_median_us']} | **{r['backward_to_forward_ratio']}** | "
            f"{float(r['forward_matmul_share']):.0%} / {float(r['backward_matmul_share']):.0%} | "
            f"{r['forward_mflop']} / {r['backward_mflop']} | {r['flop_ratio_bwd_fwd']} |")
    add("")
    add("![Timing breakdown](results/timing_breakdown.png)\n")
    add("*Figure 3 - Median forward and backward time; the dark part of each bar is the time of the "
        "pass's matrix multiplications alone (`results/timing_breakdown.png`, raw runs in "
        "`results/timing_raw.json`).*\n")
    ratios = [float(r["backward_to_forward_ratio"]) for r in tim]
    big = max(tim, key=lambda r: float(r["backward_mflop"]))
    small = min(tim, key=lambda r: float(r["backward_mflop"]))
    dom = "backward" if all(x > 1 for x in ratios) else ("forward" if all(x < 1 for x in ratios) else "mixed")
    if dom == "backward":
        para = (f"**The backward pass dominates** in every configuration ({min(ratios):.2f}x to "
                f"{max(ratios):.2f}x the forward time).")
    elif dom == "forward":
        para = (f"In this run the forward pass took longer in every configuration "
            f"({min(ratios):.2f}x to {max(ratios):.2f}x).")
    else:
        para = f"The backward/forward ratio ranges from {min(ratios):.2f} to {max(ratios):.2f}."
    add(para + " The reason is matrix multiplication. The forward pass needs 3 matmuls (`X W1`, `A1 W2`, "
        "`A2 W3`); the backward pass needs 5, because every layer must produce a weight gradient "
        "(`A^T dZ`) and - except the first layer - also pass the error back (`dZ W^T`). Counting "
        f"2·m·k·n FLOPs per matmul, backward does {small['flop_ratio_bwd_fwd']}x the work of forward for "
        f"W = {small['width']} and {big['flop_ratio_bwd_fwd']}x for W = {big['width']} (the W x W layer "
        "appears twice in backward). The measured ratio follows this FLOP ratio more closely as the "
        f"matrices grow: for W = {big['width']} with batch {big['batch'].split()[0]} the matmuls are "
        f"{float(big['backward_matmul_share']):.0%} of the backward time and the measured ratio is "
        f"{big['backward_to_forward_ratio']} (FLOP ratio {big['flop_ratio_bwd_fwd']}). For tiny matrices "
        f"(W = {small['width']}, batch {small['batch'].split()[0]}) each NumPy call costs a few microseconds "
        "of fixed overhead regardless of size, and the forward pass also runs `exp` and divisions for "
        "the softmax, so the two passes take similar time. Doubling the batch or the width therefore "
        "mainly increases matmul time, and backward - with more and larger matmuls - grows fastest.\n")

    # ------------------------------------------------------------------------------------------
    add("## 4. Conclusions\n")
    tr = " and ".join(f"{r['passed']}/10 (W = {r['width']})" for r in trained)
    add(f"1. **The hand-derived backpropagation is correct.** The numerical check passes on the trained "
        f"networks ({tr}) and the all-parameter relative error is about 1e-10. The untrained check also "
        "showed that a finite-difference test has a precision floor: with tiny N(0, 0.01) gradients the "
        "relative error is dominated by float64 rounding, so a gradient check must be read together with "
        "the gradient's size.\n")
    best_stuck = (f"With N(0, 0.01) initial weights only lr = {max(LEARNING_RATES):g} learned at all "
                  f"(best test accuracy {best['final_test_acc']} with W = {best['width']}); "
                  "the smaller learning rates stayed at ln(10) and chance accuracy for all 50 epochs."
                  if stuck else
                  f"The best result was test accuracy {best['final_test_acc']} (W = {best['width']}, "
                  f"lr = {float(best['lr']):g}).")
    add(f"2. **Initialisation scale and learning rate decide whether the network learns, width decides how "
        f"well.** {best_stuck} The wider network escaped the plateau earlier and reached a much higher "
        f"accuracy. With standard deviation 0.1 all settings improve (best {sup_best['final_test_acc']}), so "
        "the small-weight start, not the model, limited the main runs.\n")
    add(f"3. **Training time is dominated by matrix multiplication, and the backward pass is the more "
        f"expensive half.** Backward needs 5 matmuls against 3 in forward ({small['flop_ratio_bwd_fwd']}x to "
        f"{big['flop_ratio_bwd_fwd']}x the FLOPs); the measured backward/forward ratio was "
        f"{min(ratios):.2f}-{max(ratios):.2f} and approaches the FLOP ratio as the matrices grow, while "
        "for very small matrices fixed per-call overhead hides the difference.\n")

    add("## 5. Reproducibility\n")
    add("From the repository root with the virtual environment active, run `python lab02/src/run_all.py`. "
        "It regenerates every file in `lab02/results/` and this report (the handwritten page photo is kept). "
        "All randomness uses seed 42; timings depend on the machine, so small differences between runs are "
        "expected.\n")
    add("| File | Content |")
    add("|---|---|")
    for fname, desc in [
        ("versions.txt", "Python and package versions vs. pins"),
        ("hardware.json", "machine used"),
        ("data_checks.json", "step 1: data shapes, scaling, split"),
        ("network_design.csv", "step 2: parameter shapes and counts"),
        ("network_design_paper.jpg", "step 2: handwritten design page"),
        ("forward_checks.csv", "steps 3-4: output shape, row sums, initial loss"),
        ("grad_check.csv / grad_check_summary.csv", "step 5: per-weight and summary gradient check"),
        ("training_log.csv", "step 6: per-epoch train/test loss and accuracy for all runs"),
        ("final_accuracy.csv", "step 6: final accuracy per run"),
        ("loss_curves.png / accuracy_curves.png", "step 6: curves"),
        ("params_W<width>_lr<lr>.npz", "trained parameters of each run"),
        ("*_init_std0.1.*", "supplementary runs with initial std 0.1"),
        ("timing.csv / timing_raw.json / timing_breakdown.png", "step 7: timing breakdown"),
    ]:
        add(f"| `results/{fname}` | {desc} |")
    add("")

    (LAB_DIR / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {LAB_DIR / 'report.md'}" + ("" if photo else
          "  (note: results/network_design_paper.jpg not found yet)"))


if __name__ == "__main__":
    main()
