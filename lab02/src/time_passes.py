"""Step 7: time the forward and backward passes separately (median of 100 runs, 1 warm-up).

For each width (16, 64) and two batch sizes - 32 (the training mini-batch) and the full training
set (1257 rows) - we time:
  forward   : nn.forward (3 matmuls + bias adds + 2 ReLU + softmax)
  loss      : nn.cross_entropy on the cached logits
  backward  : nn.backward (5 matmuls + ReLU masks + bias sums)
and, to link the timing to matrix multiplication, the matmuls alone with exactly the same shapes:
  forward matmuls : X@W1, A1@W2, A2@W3
  backward matmuls: A2^T@dZ3, dZ3@W3^T, A1^T@dZ2, dZ2@W2^T, X^T@dZ1
The analytic FLOP count of each pass (2*m*k*n per matmul) is recorded next to the times.

Outputs: results/timing.csv, results/timing_raw.json, results/hardware.json,
         results/timing_breakdown.png
"""

from __future__ import annotations

import csv

import numpy as np

import nn
from common import (INIT_STD, RESULTS_DIR, SEED, TIMING_REPEATS, TIMING_WARMUP, WIDTHS,
                    hardware_info, load_data, set_seeds, summarize_us, time_call, write_json)

INK, MUTED, GRID, AXIS, SURFACE = "#1f1f1d", "#6b6b66", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_FWD, C_BWD = "#2a78d6", "#eb6834"


def main() -> None:
    set_seeds()
    X_train, _, Y_train, _, _, _ = load_data()
    write_json(RESULTS_DIR / "hardware.json", hardware_info())

    batches = {"32": (X_train[:32], Y_train[:32]), f"{len(X_train)} (full train)": (X_train, Y_train)}
    rows, raw = [], {}
    for width in WIDTHS:
        params = nn.init_params(width, np.random.default_rng(SEED), INIT_STD)
        for label, (X, Y) in batches.items():
            _, cache = nn.forward(params, X)
            grads = nn.backward(params, cache, Y)
            dZ3 = (cache["P"] - Y) / Y.shape[0]
            dZ2 = (dZ3 @ params["W3"].T) * (cache["Z2"] > 0)
            dZ1 = (dZ2 @ params["W2"].T) * (cache["Z1"] > 0)
            W1, W2, W3, A1, A2 = params["W1"], params["W2"], params["W3"], cache["A1"], cache["A2"]

            def fwd_mm():
                X @ W1
                A1 @ W2
                A2 @ W3

            def bwd_mm():
                A2.T @ dZ3
                dZ3 @ W3.T
                A1.T @ dZ2
                dZ2 @ W2.T
                X.T @ dZ1

            t = {
                "forward": time_call(lambda: nn.forward(params, X)),
                "loss": time_call(lambda: nn.cross_entropy(cache, Y)),
                "backward": time_call(lambda: nn.backward(params, cache, Y)),
                "forward_matmuls": time_call(fwd_mm),
                "backward_matmuls": time_call(bwd_mm),
            }
            s = {k: summarize_us(v) for k, v in t.items()}
            flops = nn.matmul_flops(width, X.shape[0])
            key = f"W{width}_batch{X.shape[0]}"
            raw[key] = {k: [x * 1e6 for x in v] for k, v in t.items()}
            f_med, b_med = s["forward"]["median_us"], s["backward"]["median_us"]
            rows.append({
                "width": width, "batch": label,
                "forward_median_us": f"{f_med:.2f}", "forward_p95_us": f"{s['forward']['p95_us']:.2f}",
                "loss_median_us": f"{s['loss']['median_us']:.2f}",
                "backward_median_us": f"{b_med:.2f}", "backward_p95_us": f"{s['backward']['p95_us']:.2f}",
                "backward_to_forward_ratio": f"{b_med / f_med:.2f}",
                "forward_matmul_median_us": f"{s['forward_matmuls']['median_us']:.2f}",
                "backward_matmul_median_us": f"{s['backward_matmuls']['median_us']:.2f}",
                "forward_matmul_share": f"{s['forward_matmuls']['median_us'] / f_med:.2f}",
                "backward_matmul_share": f"{s['backward_matmuls']['median_us'] / b_med:.2f}",
                "forward_matmuls": 3, "backward_matmuls": 5,
                "forward_mflop": f"{flops['forward'] / 1e6:.3f}",
                "backward_mflop": f"{flops['backward'] / 1e6:.3f}",
                "flop_ratio_bwd_fwd": f"{flops['backward'] / flops['forward']:.2f}",
                "runs": s["forward"]["n"], "warmup": TIMING_WARMUP,
            })
            print(f"W={width:<3} batch {label:<17} forward {f_med:9.2f} us | backward {b_med:9.2f} us | "
                  f"ratio {b_med / f_med:4.2f} | matmul share fwd {rows[-1]['forward_matmul_share']} "
                  f"bwd {rows[-1]['backward_matmul_share']} | FLOP ratio {rows[-1]['flop_ratio_bwd_fwd']}")
            assert len(t["forward"]) == TIMING_REPEATS

    write_json(RESULTS_DIR / "timing_raw.json", {"unit": "microseconds", "runs": raw})
    with open(RESULTS_DIR / "timing.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot(rows)


def plot(rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    for ax, batch_prefix in zip(axes, ("32", "1257")):
        sub = [r for r in rows if r["batch"].startswith(batch_prefix)]
        labels = [f"W = {r['width']}" for r in sub]
        x = np.arange(len(sub))
        fwd = [float(r["forward_median_us"]) for r in sub]
        bwd = [float(r["backward_median_us"]) for r in sub]
        fmm = [float(r["forward_matmul_median_us"]) for r in sub]
        bmm = [float(r["backward_matmul_median_us"]) for r in sub]
        wdt = 0.36
        ax.bar(x - wdt / 2, fwd, wdt, color=C_FWD, alpha=0.35, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.bar(x - wdt / 2, fmm, wdt, color=C_FWD, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.bar(x + wdt / 2, bwd, wdt, color=C_BWD, alpha=0.35, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.bar(x + wdt / 2, bmm, wdt, color=C_BWD, edgecolor=SURFACE, linewidth=2, zorder=4)
        top = max(bwd + fwd)
        for xi, f_, b_ in zip(x, fwd, bwd):
            ax.text(xi - wdt / 2, f_ + top * 0.02, f"{f_:.0f} us", ha="center", fontsize=8, color=INK)
            ax.text(xi + wdt / 2, b_ + top * 0.02, f"{b_:.0f} us", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, top * 1.15)
        ax.set_facecolor(SURFACE)
        ax.set_title(f"Batch {sub[0]['batch']}", fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.set_ylabel("median time (us)", fontsize=8, color=MUTED)
    handles = [Patch(color=C_FWD, label="forward: matmul part"),
               Patch(color=C_FWD, alpha=0.35, label="forward: total"),
               Patch(color=C_BWD, label="backward: matmul part"),
               Patch(color=C_BWD, alpha=0.35, label="backward: total")]
    fig.legend(handles=handles, loc="upper right", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Lab 2 - forward vs backward time, median of 100 runs", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RESULTS_DIR / "timing_breakdown.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
