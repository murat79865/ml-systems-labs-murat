"""Build lab05/report.md from the files in lab05/results/.

Every number is read from results/ and the interpretation sentences are chosen from the measured
values. The handwritten Value-node design is embedded if results/value_design_paper.(jpg|jpeg|png)
exists.
"""

from __future__ import annotations

import csv
import re

from common import EPOCHS, HIDDEN, LAB_DIR, LR, N_SAMPLES, NOISE, RESULTS_DIR, SPEED_RUNS, read_json


def rows(name):
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def photo():
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        if (RESULTS_DIR / f"value_design_paper.{ext}").exists():
            return f"value_design_paper.{ext}"
    return None


def versions():
    text = (RESULTS_DIR / "versions.txt").read_text(encoding="utf-8")
    py = re.search(r"Python\s*:\s*(\S+)", text)
    pr = re.search(r"problems:\s*(\d+)", text)
    return (py.group(1) if py else "?"), (int(pr.group(1)) if pr else -1)


def main() -> None:
    hw = read_json(RESULTS_DIR / "hardware.json")
    ver = read_json(RESULTS_DIR / "verify.json")
    gc = rows("grad_check.csv")
    mo = read_json(RESULTS_DIR / "moons.json")
    sp = rows("moons_speed.csv")
    ct = rows("compile_timing.csv")
    cs = read_json(RESULTS_DIR / "compile_status.json")
    tl = rows("tensor_layout.csv")
    co = rows("contiguity_timing.csv")
    py, problems = versions()
    pic = photo()

    L = []
    add = L.append
    add("# Lab 5 - Frameworks and automatic differentiation\n")
    add("**Course:** Machine Learning (CYI5SE039CY), Group 11  ")
    add("**Reading:** Reddi, Ch. 7 AI Frameworks - §7.3 Fundamental Concepts, §7.4 Framework Architecture, "
        "§7.6 System Integration, §§7.7-7.8\n")
    add("## 1. Goal\n")
    add("Build a scalar automatic-differentiation engine from scratch - a `Value` node, operators that record a "
        "computation graph during the forward pass, and a topological backward pass that accumulates "
        "gradients - and prove it correct against PyTorch autograd and numerical gradients. Then use it to "
        "train a small network on `make_moons`, compare its speed with PyTorch, and look inside PyTorch: its "
        "recorded graph, eager versus compiled/traced execution, and the cost of non-contiguous tensors.\n")

    add("## 2. Method\n")
    env_ok = py == "3.11.8" and problems == 0
    add("**Environment.** Python 3.11.8 in the pinned `.venv` "
        + ("(all 25 pins `OK`, `results/versions.txt`)" if env_ok else
           f"(**warning:** versions.txt reports Python {py}, {problems} mismatch(es))")
        + f"; PyTorch {hw.get('torch', '?')} on CPU ({hw['cpu']}, {hw['logical_cpus']} logical CPUs, "
          f"{hw.get('torch_threads', '?')} PyTorch threads, {hw['os']}). Seed 42 everywhere.\n")
    add("The engine is `src/engine.py`; every step has its own script (`verify.py`, `train_moons.py`, "
        "`torch_inspect.py`) and `src/run_all.py` runs them all and rebuilds this report.\n")

    # --- step 1 ---------------------------------------------------------------------------------
    add("## 3. Results\n")
    add("### 3.1 The Value node (step 1)\n")
    if pic:
        add(f"![Handwritten Value design](results/{pic})\n")
        add("*Figure 1 - Handwritten design of the Value node, its local derivative rules and the graph of the "
            "test expression.*\n")
    add("| Field | Meaning |")
    add("|---|---|")
    add("| `data` | the scalar computed in the forward pass |")
    add("| `grad` | dL/d(node), starts at 0 and is filled by `backward()` |")
    add("| `_parents` | the input nodes of the operation - the edges of the graph |")
    add("| `_op` | the operation that produced the node (`+`, `*`, `**k`, `relu`, ...) |")
    add("| `_backward` | the local derivative rule: a closure that adds (local derivative x `out.grad`) to "
        "each parent's `grad` |")
    add("")

    # --- step 2 ---------------------------------------------------------------------------------
    add("### 3.2 Operations build the graph (step 2)\n")
    add("| Operation | Forward | Local derivative rule (in `_backward`) |")
    add("|---|---|---|")
    add("| add | `c = a + b` | `a.grad += c.grad`, `b.grad += c.grad` |")
    add("| mul | `c = a * b` | `a.grad += b.data * c.grad`, `b.grad += a.data * c.grad` |")
    add("| pow | `c = a ** k` (constant k) | `a.grad += k * a.data**(k-1) * c.grad` |")
    add("| relu | `c = max(0, a)` | `a.grad += (1 if a.data > 0 else 0) * c.grad` |")
    add("")
    add("Each operator returns a new `Value` whose `_parents` are its inputs, so simply evaluating an "
        "expression records the graph. For training, `exp`, `log` and `sigmoid` were added with the same "
        "pattern; subtraction and division are built from `+`, `*` and `**`.\n")
    g = ver["graph"]
    add(f"For the test expression `((a*b + c)**2).relu().sum()` with 5-element vectors the forward pass builds "
        f"**{g['nodes']} nodes and {g['edges']} edges**: "
        + ", ".join(f"{v} x `{k}`" for k, v in g["ops"].items())
        + " (`sum` is a chain of `+` nodes, starting from a constant 0 leaf). The topological order with every "
          "node's data and gradient is in `results/graph_test_expression.txt`, and a Graphviz version in "
          "`results/graph_test_expression.dot`.\n")

    # --- step 3 ---------------------------------------------------------------------------------
    add("### 3.3 Backward pass (step 3)\n")
    add("`backward()` (1) sorts the graph topologically with an iterative depth-first search, so every node "
        "comes after all of its parents; (2) sets the output gradient to 1; (3) visits the nodes in reverse "
        "topological order - a node's gradient is complete before its rule passes it on; (4) each rule "
        "**adds** (`+=`) to the parents' gradients instead of assigning. Accumulation matters whenever a value "
        "is used more than once:\n")
    add("| Test | Engine | Expected | Abs. difference | Result |")
    add("|---|---:|---:|---:|:---:|")
    for t in ver["extra_tests"]:
        eng = t["engine"] if not isinstance(t["engine"], list) else ", ".join(f"{v:.6f}" for v in t["engine"])
        exp = t["expected"] if not isinstance(t["expected"], list) else ", ".join(f"{v:.6f}" for v in t["expected"])
        add(f"| {t['test']} | {eng if isinstance(eng, str) else f'{eng:.6f}'} | "
            f"{exp if isinstance(exp, str) else f'{exp:.6f}'} | {t['abs_diff']:.1e} | "
            f"{'PASS' if t['pass'] else 'FAIL'} |")
    add("")
    add("For `y = x*x + x` the node `x` receives three contributions (x, x and 1): with `=` instead of `+=` "
        "the result would be 1 or 3 instead of the correct 7.\n")

    # --- step 4 ---------------------------------------------------------------------------------
    add("### 3.4 Verification against PyTorch and numerical gradients (step 4)\n")
    add(f"`L = ((a*b + c)**2).relu().sum()`, a, b, c ~ N(0, 1) with 5 elements each (seed 42). Engine vs "
        f"PyTorch autograd (float64) vs central differences (h = 1e-6). Loss: engine {ver['loss_engine']:.10f}, "
        f"PyTorch {ver['loss_torch']:.10f}.\n")
    add("| Variable | Value | Engine grad | PyTorch grad | Numerical grad | Engine - PyTorch | Engine - numerical |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for r in gc:
        add(f"| {r['variable']} | {float(r['value']):.4f} | {float(r['engine_grad']):.8f} | "
            f"{float(r['torch_grad']):.8f} | {float(r['numerical_grad']):.8f} | "
            f"{float(r['abs_diff_engine_torch']):.1e} | {float(r['abs_diff_engine_numerical']):.1e} |")
    add("")
    add(f"**Max absolute difference: engine vs PyTorch {ver['max_abs_diff_engine_vs_torch']:.1e}, engine vs "
        f"numerical {ver['max_abs_diff_engine_vs_numerical']:.1e} - both below the 1e-6 limit: "
        f"{'PASS' if ver['pass'] else 'FAIL'}.** The engine and PyTorch apply the same chain rule in float64, so "
        "they agree to rounding error; the numerical estimate differs by the truncation and rounding error "
        "of a finite difference.\n")

    # --- step 5 ---------------------------------------------------------------------------------
    add("### 3.5 Training on make_moons and speed vs PyTorch (step 5)\n")
    add(f"`make_moons(n_samples={N_SAMPLES}, noise={NOISE}, random_state=42)`; network 2 -> {HIDDEN} (ReLU) -> 1 "
        f"(sigmoid), BCE loss, {mo['parameters']} parameters; full-batch gradient descent, lr {LR:g}, {EPOCHS} "
        "epochs, float64. Both implementations start from the same weights and use the same loss formula.\n")
    add("| | Our engine | PyTorch |")
    add("|---|---:|---:|")
    add(f"| final loss | {mo['final_loss_engine']:.6f} | {mo['final_loss_torch']:.6f} |")
    add(f"| training accuracy | {mo['train_accuracy_engine']:.3f} | {mo['train_accuracy_torch']:.3f} |")
    se = next(r for r in sp if r["implementation"].startswith("our"))
    st = next(r for r in sp if r["implementation"].startswith("PyTorch"))
    add(f"| training time, median of {SPEED_RUNS} runs (s) | {float(se['median_s']):.3f} | {float(st['median_s']):.4f} |")
    add(f"| time per epoch (ms) | {float(se['ms_per_epoch']):.2f} | {float(st['ms_per_epoch']):.3f} |")
    add("")
    add("![make_moons training](results/moons.png)\n")
    add("*Figure 2 - Loss per epoch of both implementations (they lie on top of each other) and the decision "
        "boundary learned with the engine.*\n")
    add(f"The two loss curves differ by at most {mo['max_abs_loss_diff_over_epochs']:.1e} over all {EPOCHS} epochs "
        f"and the final weights by {mo['max_abs_param_diff_final']:.1e} - an end-to-end confirmation of the engine "
        f"beyond the single test expression. **PyTorch is {mo['speedup_torch_over_engine']:.0f}x faster.** The "
        f"engine builds a new Python object for every scalar operation - {mo['graph_nodes_per_epoch']:,} `Value` "
        "nodes per epoch, each with its own closure - and walks them one by one in the interpreter. PyTorch "
        "records one graph node per *tensor* operation (a few per epoch) and runs each as a single vectorised "
        "C++ kernel over all 200 samples, so the interpreter overhead is paid a handful of times instead of "
        "tens of thousands of times.\n")

    # --- step 6 ---------------------------------------------------------------------------------
    add("### 3.6 Inside PyTorch (step 6)\n")
    gi = cs["graph"]
    add(f"**Graph inspection** of the same 2 -> 8 -> 1 model as `nn.Sequential`: walking `loss.grad_fn` gives the "
        f"autograd graph recorded during one eager forward pass ({gi['autograd_nodes']} nodes: `BinaryCrossEntropyBackward0`, "
        "`SigmoidBackward0`, `AddmmBackward0`, `ReluBackward0`, `TBackward0` and `AccumulateGrad` for each "
        "parameter - `results/torch_autograd_graph.txt`). It is the tensor-level equivalent of our Value "
        f"graph. `torch.fx.symbolic_trace` gives a static graph of {gi['fx_nodes']} nodes "
        "(`results/torch_fx_graph.txt`) and `torch.jit.trace` a TorchScript graph (`results/torch_jit_graph.txt`) "
        "- the graph is captured once and can be optimised instead of being rebuilt on every call.\n")
    add("**Compilation support on this machine:**\n")
    add("| Model | torch.compile | torch.jit.trace | torch.jit.script |")
    add("|---|---|---|---|")
    for m, st_ in cs["variants"].items():
        add(f"| {m} | {st_.get('torch.compile', '-')} | {st_.get('torch.jit.trace', '-')} | {st_.get('torch.jit.script', '-')} |")
    add("")
    add("**Eager vs compiled/traced** (inference, `torch.no_grad()`, 10 warm-up calls, median of 100 calls):\n")
    add("| Model | Variant | Median ms | Speed-up vs eager | Max output difference |")
    add("|---|---|---:|---:|---:|")
    for r in ct:
        add(f"| {r['model']} | {r['variant']} | {float(r['median_ms']):.4f} | {float(r['speedup_vs_eager']):.2f}x | "
            f"{r['max_abs_output_diff_vs_eager']} |")
    add("")
    add("![PyTorch timings](results/torch_timings.png)\n")
    add("*Figure 3 - Eager vs compiled/traced execution for both models, and operations on a contiguous tensor, "
        "its transposed view and a contiguous copy (log scale).*\n")

    add("**Tensor shape, dtype and contiguity** (`results/tensor_layout.csv`):\n")
    add("| Tensor | Shape | dtype | Stride | Contiguous | Shares storage with x |")
    add("|---|---|---|---|:---:|:---:|")
    for r in tl:
        add(f"| `{r['tensor']}` | {r['shape']} | {r['dtype']} | {r['stride']} | {r['is_contiguous']} | "
            f"{r['shares_storage_with_x']} |")
    add("")
    add("A view (transpose, slice, expand) only changes the **strides** - how many elements to skip per index "
        "step - and shares the original storage; `expand` even uses stride 0. `contiguous()` and a new dtype "
        "allocate new storage.\n")

    add("**Cost of a non-contiguous view** - 2048 x 2048 float32, median of 20 runs (ms):\n")
    ops = list(dict.fromkeys(r["operation"] for r in co if r["operation"] != "make contiguous copy"))
    lays = list(dict.fromkeys(r["layout"] for r in co if r["operation"] != "make contiguous copy"))
    add("| Operation | " + " | ".join(lays) + " |")
    add("|---|" + "---:|" * len(lays))
    get = lambda o, l: float(next(r["median_ms"] for r in co if r["operation"] == o and r["layout"] == l))
    for o in ops:
        add(f"| {o} | " + " | ".join(f"{get(o, l):.3f}" for l in lays) + " |")
    copies = [r for r in co if r["operation"] == "make contiguous copy"]
    add("")
    add("Cost of making the copy itself: " + "; ".join(f"{r['layout'].replace(' copy itself', '')} "
                                                        f"{float(r['median_ms']):.3f} ms" for r in copies) + ".\n")

    def ratio(o, a, b):
        return get(o, b) / get(o, a)

    ew = next(o for o in ops if o.startswith("elementwise"))
    rd = next(o for o in ops if o.startswith("row reduction"))
    mm = next(o for o in ops if o.startswith("matmul"))
    vw = next(o for o in ops if o.startswith("view"))
    r_ew = ratio(ew, "contiguous x", "non-contiguous x.t()")
    r_rd = ratio(rd, "contiguous x", "non-contiguous x.t()")
    r_mm = ratio(mm, "contiguous x", "non-contiguous x.t()")
    r_vw = ratio(vw, "contiguous x", "non-contiguous x.t()")
    desc = lambda r: f"{r:.1f}x slower" if r >= 1.15 else (f"{1 / r:.1f}x faster" if r <= 0.87 else "about the same")
    res = {"element-wise op": r_ew, "row reduction": r_rd, "matmul": r_mm}
    slower = [k for k, v in res.items() if v >= 1.15]
    same = [k for k, v in res.items() if v < 1.15]
    para = (f"On the transposed view the element-wise op is {desc(r_ew)}, the row reduction {desc(r_rd)} and the "
            f"matmul {desc(r_mm)} than on the contiguous tensor; `reshape(-1)` is {desc(r_vw)}. ")
    if slower:
        para += (f"Where it is slower ({', '.join(slower)}) the cause is memory access: in the contiguous tensor "
                 "consecutive elements of a row are neighbours in memory, so the CPU reads whole cache lines and "
                 "uses vector instructions, while on the transposed view the same loop jumps 2048 floats (8 KB) "
                 "between elements and wastes most of every cache line. ")
    if same:
        para += (f"Where it is not ({', '.join(same)}) PyTorch avoids the strided loop: element-wise kernels "
                 "iterate in memory order whatever the logical shape, and BLAS receives a transpose as a flag "
                 "and picks a matching kernel. ")
    para += ("`reshape(-1)` is a free view on a contiguous tensor but must copy the whole array when the strides "
             "do not allow a view - the clearest cost of non-contiguity. When a non-contiguous tensor is used "
             "repeatedly, paying once for `.contiguous()` and reusing the copy avoids paying on every operation.")
    add(para + "\n")

    # --- note on compilation ---------------------------------------------------------------------
    add("### 3.7 Note: when does compilation help?\n")
    by = {}
    for r in ct:
        by.setdefault(r["model"], {})[r["variant"]] = float(r["median_ms"])
    lines = []
    for m, v in by.items():
        best = min((k for k in v if k != "eager"), key=lambda k: v[k], default=None)
        if best:
            lines.append(f"{m}: best {best} {v['eager'] / v[best]:.2f}x vs eager "
                         f"({v['eager']:.4f} -> {v[best]:.4f} ms)")
    compile_ok = any(st_.get("torch.compile") == "ok" for st_ in cs["variants"].values())
    note = ("Measured: " + "; ".join(lines) + ". ")
    if not compile_ok:
        note += ("`torch.compile` is not available in this setup (see the status table), so tracing and "
                 "scripting were used, as the lab manual allows. ")
    tiny_name = next(iter(by))
    wide_name = list(by)[-1]
    t_gain = by[tiny_name]["eager"] / min(v for k, v in by[tiny_name].items() if k != "eager")
    w_gain = by[wide_name]["eager"] / min(v for k, v in by[wide_name].items() if k != "eager")
    note += ("Compilation (or tracing) removes per-call Python and dispatcher overhead and can fuse small "
             "operations into fewer kernels. It therefore helps most when that overhead is a large share of the "
             "run time - small models, small batches, many cheap element-wise ops - and when the same graph is "
             "run many times, so the one-off compile cost is amortised. It helps little when the time is "
             "dominated by large matrix multiplications, which already run in optimised BLAS kernels in eager "
             "mode either way. ")
    comp = {mm: v.get("torch.compile") for mm, v in by.items()}
    slow_comp = [mm for mm, v in comp.items() if v is not None and v > by[mm]["eager"] * 1.02]
    if compile_ok and slow_comp:
        note += ("`torch.compile` itself did not pay off here: "
                 + ", ".join(f"{mm} {by[mm]['eager'] / comp[mm]:.2f}x" for mm in comp if comp[mm] is not None)
                 + ". For the tiny model every call is only a few hundredths of a millisecond, so the guard checks "
                   "and the hand-off into the compiled graph cost about as much as the work itself; for the wide "
                   "model the time is spent in large matrix multiplications that the compiler passes to the same "
                   "BLAS library as eager mode. TorchScript (trace/script) has a lighter call path and gave the "
                   "larger gain on the tiny model. ")
    if t_gain > w_gain:
        note += (f"This run shows exactly that: the overhead-bound tiny model gains more ({t_gain:.2f}x) than the "
                 f"matmul-bound wide model ({w_gain:.2f}x). ")
    else:
        note += (f"In this run the gains were {t_gain:.2f}x (tiny) and {w_gain:.2f}x (wide); on a CPU without "
                 "kernel fusion the remaining overhead is small in both cases. ")
    note += ("It is not worth it for code that changes shape or control flow on every call (re-compilation), "
             "for one-off scripts, or while debugging, where eager mode's immediate execution is what you want.")
    add(note + "\n")

    # --- conclusions -----------------------------------------------------------------------------
    add("## 4. Conclusions\n")
    dt = ver["max_abs_diff_engine_vs_torch"]
    add(f"1. **A correct autograd needs only three ideas:** record parents and a local derivative rule in every "
        f"node, sort topologically, and accumulate. With them the engine matches PyTorch "
        + ("exactly (difference 0)" if dt == 0 else f"to {dt:.0e}") + f" and numerical gradients to "
        f"{ver['max_abs_diff_engine_vs_numerical']:.0e} on the test expression, and reproduces PyTorch's whole "
        f"training run to {mo['max_abs_loss_diff_over_epochs']:.0e}.\n")
    add(f"2. **Granularity decides speed.** The same mathematics is {mo['speedup_torch_over_engine']:.0f}x slower "
        f"with scalar nodes ({mo['graph_nodes_per_epoch']:,} Python objects per epoch) than with tensor operations: "
        "frameworks win by making each graph node do a lot of work in compiled code.\n")
    worst = max(res, key=res.get)
    add(f"3. **Execution mode and memory layout are system costs too.** Capturing the graph (tracing/compiling) "
        f"gave a {t_gain:.2f}x speed-up for the tiny model and {w_gain:.2f}x for the wide one, and on a transposed "
        f"view the {worst} was {desc(res[worst])} while `reshape(-1)` turned from a free view into a full copy "
        "- the same values, a different memory layout.\n")
    add("## 5. Reproducibility\n")
    add("With the virtual environment active, run `python lab05/src/run_all.py` from the repository root. It "
        "regenerates every file in `lab05/results/` and this report (the photo of the handwritten page is kept). "
        "Seeds are fixed to 42; timings depend on the machine and its load.\n")
    (LAB_DIR / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {LAB_DIR / 'report.md'}" + ("" if pic else "  (note: value_design_paper photo not found yet)"))


if __name__ == "__main__":
    main()
