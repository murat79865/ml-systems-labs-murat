"""Step 6: inside PyTorch - graph inspection, eager vs compiled/traced execution, tensor layout.

(a) Graph of a small model (the 2 -> 8 -> 1 network of step 5 as nn.Sequential):
    the dynamic autograd graph built by one forward pass (walking loss.grad_fn), the torch.fx graph
    (symbolic_trace) and the TorchScript graph from torch.jit.trace.
(b) Eager vs compiled: torch.compile is tried first; if it is unsupported on this machine (e.g.
    Windows in PyTorch 2.3), the error is recorded and torch.jit.trace / torch.jit.script are used.
    Two models: the tiny 2 -> 8 -> 1 net (batch 200) and a wider MLP 256 -> 1024 -> 1024 -> 10
    (batch 64). Inference under torch.no_grad(); 10 warm-up calls (traced/compiled code is optimised
    during the first calls), then the median of 100 timed calls.
(c) Tensor shape, dtype, strides and contiguity of a tensor and several views.
(d) Cost of a non-contiguous view: the same operations on a contiguous 2048 x 2048 float32 tensor,
    on its transposed (non-contiguous) view, and on a contiguous copy of that view; 3 warm-ups,
    median of 20 runs. The cost of making the copy (.contiguous()) is measured separately.

Outputs: results/torch_autograd_graph.txt, results/torch_fx_graph.txt, results/torch_jit_graph.txt,
         results/compile_timing.csv, results/compile_status.json, results/tensor_layout.csv,
         results/contiguity_timing.csv, results/hardware.json, results/torch_timings.png
"""

from __future__ import annotations

import csv
import statistics

import torch
from torch import nn

from common import RESULTS_DIR, SEED, hardware_info, set_seeds, time_call, write_json

WARMUP, RUNS = 10, 100
C_WARMUP, C_RUNS = 3, 20


def tiny() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())


def wide() -> nn.Module:
    return nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 10))


# ---------------------------------------------------------------------------------------------
def autograd_graph_text(loss: torch.Tensor) -> str:
    lines, seen = [], set()
    keep = []  # hold references: grad_fn wrappers are created on access, and a freed object's id can be reused

    def walk(fn, depth):
        if fn is None:
            return
        keep.append(fn)
        key = id(fn)
        name = type(fn).__name__
        extra = ""
        if hasattr(fn, "variable"):
            v = fn.variable
            extra = f"  <- leaf parameter {tuple(v.shape)} {str(v.dtype).replace('torch.', '')}"
        lines.append("  " * depth + name + extra + ("  (already shown)" if key in seen else ""))
        if key in seen:
            return
        seen.add(key)
        for nxt, _ in fn.next_functions:
            walk(nxt, depth + 1)

    walk(loss.grad_fn, 0)
    return "\n".join(lines)


def inspect_graph(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    loss = nn.functional.binary_cross_entropy(model(x).squeeze(1), y)
    text = autograd_graph_text(loss)
    (RESULTS_DIR / "torch_autograd_graph.txt").write_text(
        "Autograd graph recorded during ONE eager forward pass (root = loss, children = inputs of each op)\n\n"
        + text + "\n", encoding="utf-8")
    fx = torch.fx.symbolic_trace(model)
    (RESULTS_DIR / "torch_fx_graph.txt").write_text(
        "torch.fx.symbolic_trace(model).graph\n\n" + str(fx.graph) + "\n\nGenerated code:\n" + fx.code,
        encoding="utf-8")
    with torch.no_grad():
        traced = torch.jit.trace(model.eval(), x)
    (RESULTS_DIR / "torch_jit_graph.txt").write_text(
        "torch.jit.trace(model, example_input).graph\n\n" + str(traced.graph), encoding="utf-8")
    model.train()
    return {"autograd_nodes": len(text.splitlines()), "fx_nodes": len(list(fx.graph.nodes)),
            "jit_graph_lines": len(str(traced.graph).splitlines())}


# ---------------------------------------------------------------------------------------------
def compile_variants(model: nn.Module, x: torch.Tensor, status: dict, tag: str):
    variants = {"eager": model}
    try:
        compiled = torch.compile(model)
        with torch.no_grad():
            compiled(x)  # triggers compilation
        variants["torch.compile"] = compiled
        status[tag]["torch.compile"] = "ok"
    except Exception as e:  # unsupported platform / missing compiler
        status[tag]["torch.compile"] = f"unsupported: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
    with torch.no_grad():
        try:
            variants["torch.jit.trace"] = torch.jit.trace(model, x)
            status[tag]["torch.jit.trace"] = "ok"
        except Exception as e:
            status[tag]["torch.jit.trace"] = f"failed: {type(e).__name__}: {e}"
    try:
        variants["torch.jit.script"] = torch.jit.script(model)
        status[tag]["torch.jit.script"] = "ok"
    except Exception as e:
        status[tag]["torch.jit.script"] = f"failed: {type(e).__name__}: {e}"
    return variants


def time_variants(name: str, model: nn.Module, x: torch.Tensor, status: dict, rows: list) -> None:
    model.eval()
    status[name] = {}
    variants = compile_variants(model, x, status, name)
    with torch.no_grad():
        ref = model(x)
        eager_ms = None
        for vname, m in variants.items():
            out = m(x)
            max_diff = float((out - ref).abs().max())
            t = time_call(lambda: m(x), WARMUP, RUNS)
            med = statistics.median(t) * 1000
            eager_ms = med if vname == "eager" else eager_ms
            rows.append({"model": name, "input_shape": str(tuple(x.shape)), "variant": vname,
                         "median_ms": f"{med:.5f}", "p95_ms": f"{sorted(t)[int(0.95 * len(t)) - 1] * 1000:.5f}",
                         "speedup_vs_eager": f"{eager_ms / med:.3f}", "runs": RUNS, "warmup": WARMUP,
                         "max_abs_output_diff_vs_eager": f"{max_diff:.2e}"})
            print(f"{name:<22} {vname:<17} median {med:9.4f} ms  ({eager_ms / med:5.2f}x eager)")


# ---------------------------------------------------------------------------------------------
def tensor_layout() -> list[dict]:
    x = torch.arange(4 * 6, dtype=torch.float32).reshape(4, 6)
    views = {
        "x = arange(24).reshape(4, 6)": x,
        "x.t()  (transpose view)": x.t(),
        "x[:, ::2]  (strided slice)": x[:, ::2],
        "x[1:3]  (row slice)": x[1:3],
        "x.unsqueeze(0).expand(3, 4, 6)": x.unsqueeze(0).expand(3, 4, 6),
        "x.t().contiguous()  (copy)": x.t().contiguous(),
        "x.half()  (new dtype)": x.half(),
    }
    rows = []
    for name, t in views.items():
        rows.append({"tensor": name, "shape": str(tuple(t.shape)), "dtype": str(t.dtype).replace("torch.", ""),
                     "stride": str(t.stride()), "is_contiguous": t.is_contiguous(),
                     "shares_storage_with_x": t.untyped_storage().data_ptr() == x.untyped_storage().data_ptr(),
                     "element_size_bytes": t.element_size()})
    return rows


def contiguity_timing() -> list[dict]:
    n = 2048
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(n, n, generator=g)
    w = torch.randn(n, n, generator=g)
    xt = x.t()                    # non-contiguous view
    xtc = xt.contiguous()         # same values, contiguous
    xs = x[:, ::2]                # strided slice
    xsc = xs.contiguous()
    cases = [
        ("elementwise  y = v * 2 + 1", lambda v: v * 2 + 1),
        ("row reduction  v.sum(dim=1)", lambda v: v.sum(dim=1)),
        ("matmul  v @ w", lambda v: v @ w[: v.shape[1]]),
        ("view to 1-D  v.reshape(-1)", lambda v: v.reshape(-1)),
    ]
    rows = []
    for op_name, op in cases:
        for layout, v in (("contiguous x", x), ("non-contiguous x.t()", xt), ("x.t().contiguous()", xtc),
                          ("strided x[:, ::2]", xs), ("x[:, ::2].contiguous()", xsc)):
            t = time_call(lambda: op(v), C_WARMUP, C_RUNS)
            rows.append({"operation": op_name, "layout": layout, "shape": str(tuple(v.shape)),
                         "is_contiguous": v.is_contiguous(), "median_ms": f"{statistics.median(t) * 1000:.4f}"})
    for layout, v in (("x.t().contiguous() copy itself", xt), ("x[:, ::2].contiguous() copy itself", xs)):
        t = time_call(lambda: v.contiguous(), C_WARMUP, C_RUNS)
        rows.append({"operation": "make contiguous copy", "layout": layout, "shape": str(tuple(v.shape)),
                     "is_contiguous": False, "median_ms": f"{statistics.median(t) * 1000:.4f}"})
    for r in rows:
        print(f"{r['operation']:<30} {r['layout']:<34} {float(r['median_ms']):9.3f} ms")
    return rows


def plot(compile_rows, contig_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, AXIS, SURFACE = "#1f1f1d", "#6b6b66", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    COL = {"eager": "#2a78d6", "torch.compile": "#1baf7a", "torch.jit.trace": "#eb6834", "torch.jit.script": "#7d5fd1"}
    models = list(dict.fromkeys(r["model"] for r in compile_rows))
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(15, 4.6), facecolor=SURFACE)
    for ax, m in zip(axes, models):
        sub = [r for r in compile_rows if r["model"] == m]
        vals = [float(r["median_ms"]) for r in sub]
        bars = ax.bar([r["variant"].replace("torch.", "") for r in sub], vals,
                      color=[COL.get(r["variant"], MUTED) for r in sub], edgecolor=SURFACE, linewidth=2, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.3f}", ha="center", fontsize=8, color=INK)
        ax.set_title(f"{m}: median ms per call", fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, max(vals) * 1.18)
    ax = axes[-1]
    ops = list(dict.fromkeys(r["operation"] for r in contig_rows if r["operation"] != "make contiguous copy"))
    layouts = ["contiguous x", "non-contiguous x.t()", "x.t().contiguous()"]
    lcol = {"contiguous x": "#2a78d6", "non-contiguous x.t()": "#eb6834", "x.t().contiguous()": "#1baf7a"}
    width = 0.26
    for k, lay in enumerate(layouts):
        vals = [float(next(r["median_ms"] for r in contig_rows if r["operation"] == o and r["layout"] == lay))
                for o in ops]
        ax.bar([i + (k - 1) * width for i in range(len(ops))], vals, width, color=lcol[lay], label=lay,
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xticks(range(len(ops)), [o.split("  ")[0] for o in ops])
    ax.set_yscale("log")
    ax.set_title("2048x2048 float32: contiguous vs view (log ms)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7)
    for a in axes:
        a.set_facecolor(SURFACE)
        a.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        a.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(AXIS)
    fig.suptitle("Lab 5 - PyTorch execution modes and memory layout (CPU)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS_DIR / "torch_timings.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    set_seeds()
    write_json(RESULTS_DIR / "hardware.json", hardware_info())
    from sklearn.datasets import make_moons

    from common import N_SAMPLES, NOISE

    X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    xm, ym = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    torch.manual_seed(SEED)
    graph_info = inspect_graph(tiny(), xm, ym)
    print(f"graph: autograd {graph_info['autograd_nodes']} lines, fx {graph_info['fx_nodes']} nodes")

    status, crow = {}, []
    torch.manual_seed(SEED)
    time_variants("tiny 2-8-1 (batch 200)", tiny(), xm, status, crow)
    torch.manual_seed(SEED)
    time_variants("wide 256-1024-1024-10 (batch 64)", wide(), torch.randn(64, 256), status, crow)
    with open(RESULTS_DIR / "compile_timing.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(crow[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(crow)
    write_json(RESULTS_DIR / "compile_status.json", {"torch_version": torch.__version__, "graph": graph_info,
                                                      "variants": status})

    lay = tensor_layout()
    with open(RESULTS_DIR / "tensor_layout.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lay[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(lay)
    cont = contiguity_timing()
    with open(RESULTS_DIR / "contiguity_timing.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cont[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(cont)
    plot(crow, cont)


if __name__ == "__main__":
    main()
