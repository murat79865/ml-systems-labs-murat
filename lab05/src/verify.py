"""Step 4: verify the engine against PyTorch autograd and numerical gradients.

Main test (lab manual): L = ((a*b + c)**2).relu().sum() with vectors a, b, c (5 elements each,
N(0,1), seed 42). dL/da, dL/db, dL/dc from
    engine     - our Value graph + backward()
    PyTorch    - torch.float64 tensors with requires_grad, L.backward()
    numerical  - central difference (L(x+h) - L(x-h)) / 2h, h = 1e-6, in float64
Pass: max absolute difference < 1e-6 for engine vs PyTorch and engine vs numerical.

Extra tests: gradient accumulation (a node used several times), the ReLU zero branch, and an
expression that uses every operation of the engine, each compared with PyTorch.

Outputs: results/grad_check.csv, results/verify.json, results/graph_test_expression.dot,
         results/graph_test_expression.txt
"""

from __future__ import annotations

import csv
import math

import numpy as np

from common import RESULTS_DIR, SEED, set_seeds, write_json
from engine import Value, graph_summary, to_dot, total

N, H, TOL = 5, 1e-6, 1e-6


def f_engine(a, b, c):
    A = [Value(x, label=f"a{i}") for i, x in enumerate(a)]
    B = [Value(x, label=f"b{i}") for i, x in enumerate(b)]
    C = [Value(x, label=f"c{i}") for i, x in enumerate(c)]
    L = total([((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(A, B, C)])
    L.label = "L"
    return L, A, B, C


def f_float(a, b, c) -> float:
    return sum(max((ai * bi + ci) ** 2, 0.0) for ai, bi, ci in zip(a, b, c))


def numerical(a, b, c):
    grads = []
    for which in range(3):
        g = []
        for i in range(N):
            vecs_p = [list(a), list(b), list(c)]
            vecs_m = [list(a), list(b), list(c)]
            vecs_p[which][i] += H
            vecs_m[which][i] -= H
            g.append((f_float(*vecs_p) - f_float(*vecs_m)) / (2 * H))
        grads.append(g)
    return grads


def torch_grads(a, b, c):
    import torch

    ta, tb, tc = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (a, b, c))
    L = ((ta * tb + tc) ** 2).relu().sum()
    L.backward()
    return float(L.item()), [t.grad.tolist() for t in (ta, tb, tc)]


def extra_tests() -> list[dict]:
    import torch

    out = []
    # 1) accumulation: x is used three times -> dy/dx = 2x + 1
    x = Value(3.0)
    y = x * x + x
    y.backward()
    out.append({"test": "accumulation: y = x*x + x at x=3", "engine": x.grad, "expected": 2 * 3.0 + 1,
                "abs_diff": abs(x.grad - 7.0)})
    # 2) ReLU zero branch: relu(-2*z) has zero gradient for z > 0
    z = Value(1.5)
    r = (z * -2.0).relu() + z * 0.0
    r.backward()
    out.append({"test": "relu negative branch: d relu(-2z)/dz at z=1.5", "engine": z.grad, "expected": 0.0,
                "abs_diff": abs(z.grad)})
    # 3) every operation, compared with torch
    vals = [0.7, -1.3, 2.1]
    p, q, s = (Value(v) for v in vals)
    e = ((p * q + s) ** 3 + (p - q).relu() * s.exp() + (s ** 0.5) / (q * q + 1.0) - (p.sigmoid() + 2.0).log())
    e.backward()
    tp, tq, ts = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in vals)
    te = ((tp * tq + ts) ** 3 + (tp - tq).relu() * ts.exp() + (ts ** 0.5) / (tq * tq + 1.0)
          - (torch.sigmoid(tp) + 2.0).log())
    te.backward()
    diffs = [abs(v.grad - t.grad.item()) for v, t in zip((p, q, s), (tp, tq, ts))]
    out.append({"test": "all operations (+, *, **, relu, exp, log, sigmoid, -, /) vs PyTorch",
                "engine": [p.grad, q.grad, s.grad], "expected": [tp.grad.item(), tq.grad.item(), ts.grad.item()],
                "abs_diff": max(diffs)})
    for t in out:
        t["pass"] = t["abs_diff"] < TOL
    return out


def main() -> None:
    set_seeds()
    rng = np.random.default_rng(SEED)
    a, b, c = (rng.normal(size=N).tolist() for _ in range(3))

    L, A, B, C = f_engine(a, b, c)
    order = L.backward()
    eng = [[v.grad for v in A], [v.grad for v in B], [v.grad for v in C]]
    num = numerical(a, b, c)
    t_loss, tg = torch_grads(a, b, c)

    rows = []
    for k, name in enumerate("abc"):
        for i in range(N):
            rows.append({"variable": f"{name}[{i}]", "value": [a, b, c][k][i], "engine_grad": eng[k][i],
                         "torch_grad": tg[k][i], "numerical_grad": num[k][i],
                         "abs_diff_engine_torch": abs(eng[k][i] - tg[k][i]),
                         "abs_diff_engine_numerical": abs(eng[k][i] - num[k][i])})
    with open(RESULTS_DIR / "grad_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.12e}" if isinstance(v, float) else v) for k, v in r.items()})

    max_t = max(r["abs_diff_engine_torch"] for r in rows)
    max_n = max(r["abs_diff_engine_numerical"] for r in rows)
    extras = extra_tests()
    summary = {
        "expression": "((a*b + c)**2).relu().sum()", "n_elements": N,
        "loss_engine": L.data, "loss_torch": t_loss, "loss_abs_diff": abs(L.data - t_loss),
        "max_abs_diff_engine_vs_torch": max_t, "max_abs_diff_engine_vs_numerical": max_n,
        "tolerance": TOL, "pass": max_t < TOL and max_n < TOL,
        "graph": graph_summary(L), "extra_tests": extras,
    }
    write_json(RESULTS_DIR / "verify.json", summary)
    (RESULTS_DIR / "graph_test_expression.dot").write_text(to_dot(L), encoding="utf-8")
    lines = ["Topological order of the graph for L = ((a*b + c)**2).relu().sum()  (backward visits it in reverse)",
             f"{'#':>3}  {'node':<8} {'op':<6} {'parents':>7} {'data':>14} {'grad':>14}"]
    for i, n in enumerate(order):
        lines.append(f"{i:>3}  {(n.label or '-'):<8} {(n._op or 'leaf'):<6} {len(n._parents):>7} "
                     f"{n.data:>14.6f} {n.grad:>14.6f}")
    (RESULTS_DIR / "graph_test_expression.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"graph: {summary['graph']['nodes']} nodes, {summary['graph']['edges']} edges, ops {summary['graph']['ops']}")
    print(f"max |engine - torch| = {max_t:.2e}   max |engine - numerical| = {max_n:.2e}   "
          f"(< {TOL:g}: {'PASS' if summary['pass'] else 'FAIL'})")
    for t in extras:
        print(f"  {t['test']}: abs diff {t['abs_diff']:.2e} -> {'PASS' if t['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
