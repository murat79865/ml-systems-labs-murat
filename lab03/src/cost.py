"""Step 3 (parameters and FLOPs): torchinfo.summary, hand-calculated FLOPs and thop.profile.

For one 1x28x28 image:
  * parameters      - torchinfo.summary (+ the hand count from hand_flops.py and sum(p.numel()))
  * hand FLOPs      - Ch. 4 formulas, per layer and per model (hand_flops.py)
  * thop.profile    - returns multiply-accumulate operations (MACs); FLOPs = 2 x MACs.
                      For the attention model thop is run three ways to explain the gap:
                        a) default (eval mode -> PyTorch's fused encoder fast path),
                        b) regular per-module path (fast path disabled),
                        c) b) + a custom thop rule for nn.MultiheadAttention (thop has none built in).
  * torchinfo       - its "mult-adds" estimate is recorded as well.

Outputs: results/params.csv, results/torchinfo_<model>.txt, results/flops_one_layer.csv,
         results/flops_models.csv, results/thop_layers.json
"""

from __future__ import annotations

import copy
import csv
import io
import contextlib

import torch
from torch import nn

import hand_flops
import models
from common import MODEL_NAMES, RESULTS_DIR, set_seeds, write_json


def count_mha(m: nn.MultiheadAttention, x, y) -> None:
    """thop rule for nn.MultiheadAttention (MACs): Q/K/V projections, QK^T, AV, output projection."""
    q, k = x[0], x[1]
    if getattr(m, "batch_first", False):
        b, L, E = q.shape
        S = k.shape[1]
    else:
        L, b, E = q.shape
        S = k.shape[0]
    macs = b * (L * E * E + 2 * S * E * E + L * S * E + L * S * E + L * E * E)
    m.total_ops += torch.DoubleTensor([float(macs)])


def flatten_layers(d: dict, prefix: str = "") -> dict[str, float]:
    out = {}
    for name, val in d.items():
        ops, _params, sub = val
        path = f"{prefix}{name}"
        out[path] = float(ops)
        if sub:
            out.update(flatten_layers(sub, path + "."))
    return out


def run_thop(model: nn.Module, x: torch.Tensor, custom_ops=None):
    """Return (total MACs, per-layer MACs dict). thop adds buffers, so a copy is profiled."""
    from thop import profile

    m = copy.deepcopy(model)
    try:
        res = profile(m, inputs=(x,), custom_ops=custom_ops, verbose=False, ret_layer_info=True)
    except TypeError:  # very old thop without ret_layer_info
        res = profile(m, inputs=(x,), custom_ops=custom_ops, verbose=False)
    macs = float(res[0])
    layers = flatten_layers(res[2]) if len(res) > 2 else {}
    return macs, layers


def main() -> None:
    from torchinfo import summary

    set_seeds()
    x1 = torch.zeros(1, 1, 28, 28)  # shape is all that matters for counting
    breakdown = hand_flops.model_breakdown()
    hand_tot = hand_flops.totals()

    params_rows, model_rows, layer_dump = [], [], {}
    thop_layers = {}
    for name in MODEL_NAMES:
        model = models.build(name).eval()

        # ---- parameters with torchinfo ---------------------------------------------------
        with contextlib.redirect_stdout(io.StringIO()):
            stats = summary(model, input_size=(1, 1, 28, 28), depth=4, verbose=0, device="cpu",
                            col_names=("input_size", "output_size", "num_params", "mult_adds"))
        (RESULTS_DIR / f"torchinfo_{name}.txt").write_text(str(stats), encoding="utf-8")
        numel = sum(p.numel() for p in model.parameters())
        params_rows.append({
            "model": name, "torchinfo_total_params": stats.total_params,
            "torchinfo_trainable_params": stats.trainable_params,
            "hand_params": hand_tot[name]["params"], "sum_numel": numel,
            "hand_matches_torchinfo": hand_tot[name]["params"] == stats.total_params,
            "in_target_100k_150k": 100_000 <= stats.total_params <= 150_000,
        })

        # ---- FLOPs: hand vs thop ---------------------------------------------------------
        macs, layers = run_thop(model, x1)
        row = {"model": name, "hand_flops": hand_tot[name]["flops"],
               "thop_macs": int(macs), "thop_flops_2x_macs": int(2 * macs),
               "thop_over_hand": round(2 * macs / hand_tot[name]["flops"], 4),
               "torchinfo_mult_adds": int(stats.total_mult_adds),
               "torchinfo_flops_2x": int(2 * stats.total_mult_adds)}
        thop_layers[name] = {"default": layers}
        if name == "attention":
            slow = models.disable_encoder_fastpath(copy.deepcopy(model))
            macs_slow, layers_slow = run_thop(slow, x1)
            macs_rule, layers_rule = run_thop(slow, x1, custom_ops={nn.MultiheadAttention: count_mha})
            row.update({"thop_flops_no_fastpath": int(2 * macs_slow),
                        "thop_flops_no_fastpath_plus_mha_rule": int(2 * macs_rule),
                        "no_fastpath_over_hand": round(2 * macs_slow / hand_tot[name]["flops"], 4),
                        "plus_mha_rule_over_hand": round(2 * macs_rule / hand_tot[name]["flops"], 4)})
            thop_layers[name]["no_fastpath"] = layers_slow
            thop_layers[name]["no_fastpath_plus_mha_rule"] = layers_rule
        model_rows.append(row)
        print(f"{name:<10} params torchinfo {stats.total_params:>8,} (hand {hand_tot[name]['params']:,}) | "
              f"FLOPs hand {hand_tot[name]['flops']:>10,}  thop 2xMACs {int(2 * macs):>10,}")
        if name == "attention":
            print(f"{'':<10} thop without fast path {row['thop_flops_no_fastpath']:,}; "
                  f"+ MHA rule {row['thop_flops_no_fastpath_plus_mha_rule']:,}")
        layer_dump[name] = breakdown[name]

    # ---- one layer of each type ----------------------------------------------------------
    # thop module paths of the hand-calculated layers
    paths = {"Linear": ("mlp", "1"), "Conv2d": ("cnn", "3"), "Attention": ("attention", "encoder.layers.0")}
    one_rows = []
    for typ, info in hand_flops.ONE_LAYER.items():
        model_name, path = paths[typ]
        r = {"type": typ, "model": model_name, "layer": info["layer"], "hand_formula": info["formula"],
             "hand_flops": info["flops"]}
        variants = thop_layers[model_name]
        for key in ("default", "no_fastpath", "no_fastpath_plus_mha_rule"):
            if key in variants:
                ops = variants[key].get(path)
                r[f"thop_flops_{key}"] = "" if ops is None else int(2 * ops)
        r["thop_default_minus_hand"] = (r["thop_flops_default"] - r["hand_flops"]
                                        if r.get("thop_flops_default") not in ("", None) else "")
        one_rows.append(r)
        print(f"one layer {typ:<9} hand {info['flops']:>10,}  thop(default) {r.get('thop_flops_default')}")

    def write(fname, rows):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(RESULTS_DIR / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write("params.csv", params_rows)
    write("flops_models.csv", model_rows)
    write("flops_one_layer.csv", one_rows)
    write("hand_layers.csv", [{"model": m, **r} for m, rows in layer_dump.items() for r in rows])
    write_json(RESULTS_DIR / "thop_layers.json", thop_layers)


if __name__ == "__main__":
    main()
