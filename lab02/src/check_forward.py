"""Steps 1-4: data checks, network design table, forward-pass checks and the loss sanity check.

  Step 1: dataset shapes, input range after /16, one-hot encoding, stratified split.
  Step 2: parameter shapes and counts for W = 16 and W = 64 (the paper design, as a table).
  Step 3: forward pass output shape must be (batch, 10) and every row must sum to 1 within 1e-6.
  Step 4: the untrained network's cross-entropy should be about ln(10) = 2.3026.

Outputs: results/data_checks.json, results/network_design.csv, results/forward_checks.csv
"""

from __future__ import annotations

import csv
import math

import numpy as np

import nn
from common import INIT_STD, RESULTS_DIR, SEED, WIDTHS, load_data, set_seeds, write_json

ROW_SUM_TOL = 1e-6
LN10 = math.log(10)


def main() -> None:
    set_seeds()
    X_train, X_test, Y_train, Y_test, y_train, y_test = load_data()

    # ---- step 1 -------------------------------------------------------------------------
    data = {
        "dataset": "sklearn.datasets.load_digits (8x8 images, 10 classes)",
        "n_samples": int(len(y_train) + len(y_test)),
        "n_features": int(X_train.shape[1]),
        "scaling": "X / 16",
        "x_min": float(min(X_train.min(), X_test.min())),
        "x_max": float(max(X_train.max(), X_test.max())),
        "one_hot_shape_train": list(Y_train.shape),
        "one_hot_rows_sum_to_1": bool(np.all(Y_train.sum(axis=1) == 1) and np.all(Y_test.sum(axis=1) == 1)),
        "split": "70/30 stratified, random_state=42",
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_class_counts": np.bincount(y_train, minlength=10).tolist(),
        "test_class_counts": np.bincount(y_test, minlength=10).tolist(),
    }
    write_json(RESULTS_DIR / "data_checks.json", data)
    print(f"data: {data['n_samples']} samples, {data['n_features']} features, x in "
          f"[{data['x_min']}, {data['x_max']}], train {data['train_size']} / test {data['test_size']}")

    # ---- step 2 -------------------------------------------------------------------------
    design_rows = []
    for width in WIDTHS:
        shapes = nn.param_shapes(width)
        layers = [("hidden 1", "W1", "b1", f"64 -> {width}", "ReLU"),
                  ("hidden 2", "W2", "b2", f"{width} -> {width}", "ReLU"),
                  ("output", "W3", "b3", f"{width} -> 10", "softmax")]
        for layer, wn, bn, io, act in layers:
            n = int(np.prod(shapes[wn]) + np.prod(shapes[bn]))
            design_rows.append({"width": width, "layer": layer, "in_out": io, "activation": act,
                                "weight_shape": str(shapes[wn]), "bias_shape": str(shapes[bn]),
                                "parameters": n})
        design_rows.append({"width": width, "layer": "total", "in_out": "", "activation": "",
                            "weight_shape": "", "bias_shape": "", "parameters": nn.n_params(width)})
    with open(RESULTS_DIR / "network_design.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(design_rows[0].keys()))
        w.writeheader()
        w.writerows(design_rows)

    # ---- steps 3 and 4 --------------------------------------------------------------------
    rows = []
    for width in WIDTHS:
        params = nn.init_params(width, np.random.default_rng(SEED), INIT_STD)
        # the shapes created must equal the designed shapes
        assert all(params[k].shape == s for k, s in nn.param_shapes(width).items())
        for split, X, Y in (("batch of 32 (train)", X_train[:32], Y_train[:32]),
                            ("full train set", X_train, Y_train),
                            ("full test set", X_test, Y_test)):
            P, cache = nn.forward(params, X)
            max_dev = float(np.abs(P.sum(axis=1) - 1.0).max())
            L = nn.cross_entropy(cache, Y)
            rows.append({
                "width": width, "input": split, "input_shape": str(X.shape),
                "output_shape": str(P.shape),
                "shape_ok": P.shape == (X.shape[0], 10),
                "max_abs_row_sum_minus_1": f"{max_dev:.3e}",
                "rows_sum_to_1": max_dev < ROW_SUM_TOL,
                "initial_loss": f"{L:.4f}",
                "ln10": f"{LN10:.4f}",
                "abs_diff_from_ln10": f"{abs(L - LN10):.2e}",
            })
            print(f"W={width:<3} {split:<20} out {P.shape}  max|rowsum-1| {max_dev:.1e}  "
                  f"loss {L:.4f} (ln10 = {LN10:.4f})")
    with open(RESULTS_DIR / "forward_checks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    assert all(r["shape_ok"] and r["rows_sum_to_1"] for r in rows), "forward-pass check failed"


if __name__ == "__main__":
    main()
