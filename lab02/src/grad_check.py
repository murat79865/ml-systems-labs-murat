"""Step 5: numerical gradient checking of the hand-derived backward pass.

Procedure (lab manual): pick 10 random weights, nudge each by +/-1e-5, compute the central
difference slope (L(w+e) - L(w-e)) / 2e and compare it with the analytic gradient from
nn.backward(). Relative error = |analytic - numerical| / max(|analytic|, |numerical|)
(defined as 0 when both are exactly 0). Pass: relative error < 1e-6 for at least 9 of 10 weights.

Where the check is run (both widths, a fixed mini-batch of the first 32 training samples):
  * "trained"   - the network after 50 epochs at lr = 0.1. This is the main check: gradients
                   have realistic magnitudes, so the comparison measures the backward pass and
                   not floating-point noise.
  * "init"      - the untrained N(0, 0.01) network, reported for transparency. Its gradients are
                   tiny (often 1e-7 to 1e-9), so the difference L(w+e) - L(w-e) of two numbers close
                   to 2.3026 loses most significant digits in float64; the numerical slope then
                   carries an error of roughly 1e-11 regardless of the analytic code, and some
                   relative errors can exceed 1e-6 even though the absolute errors are ~1e-11.
In addition every parameter (weights and biases) is checked at both points and the vector relative
error ||analytic - numerical|| / max(||analytic||, ||numerical||) is reported.

The 10 weights are sampled uniformly (seed 42) from all weight entries of W1, W2 and W3 whose analytic
gradient on the check batch is non-zero. Weights with an exactly-zero gradient (connected to a pixel
that is 0 in every check image, e.g. the image border, or to an inactive ReLU unit) agree trivially
(0 = 0) and would test nothing; how many there are is recorded in the summary.

Outputs: results/grad_check.csv (one row per sampled weight), results/grad_check_summary.csv
"""

from __future__ import annotations

import csv

import numpy as np

import nn
from common import INIT_STD, RESULTS_DIR, SEED, WIDTHS, load_data, run_name, set_seeds

EPS = 1e-5
REL_TOL = 1e-6
N_WEIGHTS = 10
MIN_PASS = 9
CHECK_BATCH = 32
TRAINED_LR = 0.1


def rel_error(a: float, n: float) -> float:
    denom = max(abs(a), abs(n))
    return 0.0 if denom == 0.0 else abs(a - n) / denom


def numerical_slope(params, name, idx, X, Y, detect_kink: bool = False):
    """Central difference (L(w+e) - L(w-e)) / 2e. With detect_kink=True also report whether any
    ReLU unit switched on/off between the two evaluations (the loss is not differentiable there)."""
    old = params[name][idx]
    params[name][idx] = old + EPS
    _, cp = nn.forward(params, X)
    lp = nn.cross_entropy(cp, Y)
    params[name][idx] = old - EPS
    _, cm = nn.forward(params, X)
    lm = nn.cross_entropy(cm, Y)
    params[name][idx] = old
    slope = (lp - lm) / (2 * EPS)
    if not detect_kink:
        return slope
    kink = bool(np.any((cp["Z1"] > 0) != (cm["Z1"] > 0)) or np.any((cp["Z2"] > 0) != (cm["Z2"] > 0)))
    return slope, kink


def candidate_weights(params, grads):
    """All weight entries (W1, W2, W3) with a non-zero analytic gradient, and the zero count."""
    cands, zeros = [], 0
    for n in ("W1", "W2", "W3"):
        for idx in np.ndindex(params[n].shape):
            if grads[n][idx] != 0.0:
                cands.append((n, idx))
            else:
                zeros += 1
    return cands, zeros


def sample_weights(params, grads, rng):
    """Uniformly sample N_WEIGHTS distinct weights among those with a non-zero gradient."""
    cands, zeros = candidate_weights(params, grads)
    pick = rng.choice(len(cands), size=N_WEIGHTS, replace=False)
    return [cands[k] for k in sorted(pick)], len(cands), zeros


def full_check(params, grads, X, Y) -> float:
    a_all, n_all = [], []
    for name in nn.PARAM_NAMES:
        for idx in np.ndindex(params[name].shape):
            a_all.append(grads[name][idx])
            n_all.append(numerical_slope(params, name, idx, X, Y))
    a, n = np.array(a_all), np.array(n_all)
    return float(np.linalg.norm(a - n) / max(np.linalg.norm(a), np.linalg.norm(n)))


def main() -> None:
    set_seeds()
    X_train, _, Y_train, _, _, _ = load_data()
    X, Y = X_train[:CHECK_BATCH], Y_train[:CHECK_BATCH]

    rows, summary = [], []
    for point in ("trained", "init"):
        for width in WIDTHS:
            if point == "init":
                params = nn.init_params(width, np.random.default_rng(SEED), INIT_STD)
            else:
                path = RESULTS_DIR / f"params_{run_name(width, TRAINED_LR)}.npz"
                with np.load(path) as z:
                    params = {k: z[k].astype(np.float64).copy() for k in nn.PARAM_NAMES}

            _, cache = nn.forward(params, X)
            L0 = nn.cross_entropy(cache, Y)
            grads = nn.backward(params, cache, Y)
            rng = np.random.default_rng(SEED)
            n_pass = 0
            picks, n_cand, n_zero = sample_weights(params, grads, rng)
            for i, (name, idx) in enumerate(picks, start=1):
                a = float(grads[name][idx])
                n, kink = numerical_slope(params, name, idx, X, Y, detect_kink=True)
                r = rel_error(a, n)
                ok = r < REL_TOL
                if ok:
                    reason = ""
                elif kink:
                    reason = "ReLU kink crossed by the +/-1e-5 nudge"
                else:
                    reason = "gradient near float64 noise floor of the finite difference"
                n_pass += ok
                rows.append({"point": point, "width": width, "i": i, "param": name,
                             "index": str(tuple(int(v) for v in idx)),
                             "weight": f"{params[name][idx]:.6e}",
                             "analytic": f"{a:.10e}", "numerical": f"{n:.10e}",
                             "abs_error": f"{abs(a - n):.3e}", "rel_error": f"{r:.3e}",
                             "pass": ok, "relu_kink_crossed": kink, "failure_reason": reason})
            vec = full_check(params, grads, X, Y)
            verdict = "met" if n_pass >= MIN_PASS else "not met"
            summary.append({"point": point, "width": width, "batch": CHECK_BATCH, "loss": f"{L0:.4f}",
                            "weights_checked": N_WEIGHTS, "passed": n_pass, "required": MIN_PASS,
                            "criterion_9_of_10": verdict,
                            "weights_total": n_cand + n_zero,
                            "weights_nonzero_grad": n_cand,
                            "weights_zero_grad_excluded": n_zero,
                            "median_abs_grad_sampled": f"{np.median([abs(float(r['analytic'])) for r in rows[-N_WEIGHTS:]]):.2e}",
                            "max_rel_error": f"{max(float(r['rel_error']) for r in rows[-N_WEIGHTS:]):.2e}",
                            "max_abs_error": f"{max(float(r['abs_error']) for r in rows[-N_WEIGHTS:]):.2e}",
                            "fails_relu_kink": sum(r["failure_reason"].startswith("ReLU") for r in rows[-N_WEIGHTS:]),
                            "fails_noise_floor": sum(r["failure_reason"].startswith("gradient") for r in rows[-N_WEIGHTS:]),
                            "all_params_checked": nn.n_params(width),
                            "all_params_vector_rel_error": f"{vec:.2e}"})
            print(f"{point:<8} W={width:<3} {n_pass}/{N_WEIGHTS} below {REL_TOL:g} -> {verdict}  "
                  f"(max rel {summary[-1]['max_rel_error']}, all-params vector rel error {vec:.1e})")

    with open(RESULTS_DIR / "grad_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(RESULTS_DIR / "grad_check_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
