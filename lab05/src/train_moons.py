"""Step 5: train a 2 -> 8 -> 1 network (ReLU hidden layer, sigmoid output, BCE loss) on make_moons
with our engine and with PyTorch, and compare their speed.

Both implementations start from the SAME initial weights (numpy, seed 42), use float64, full-batch
gradient descent with the same learning rate and number of epochs, and the same loss formula
    L = -(1/N) sum_n [ y_n log p_n + (1 - y_n) log(1 - p_n) ],  p = sigmoid(W2 . relu(W1 x + b1) + b2)
so their loss curves must coincide - a second, end-to-end check of the engine.
Speed: wall-clock time of the whole training run, median of 5 runs for each implementation.

Outputs: results/moons_training.csv, results/moons_speed.csv, results/moons.json, results/moons.png
"""

from __future__ import annotations

import csv
import statistics
import time

import numpy as np
from sklearn.datasets import make_moons

from common import EPOCHS, HIDDEN, LR, N_SAMPLES, NOISE, RESULTS_DIR, SEED, SPEED_RUNS, set_seeds, write_json
from engine import Value, total


def data():
    X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    return X.astype(np.float64), y.astype(np.float64)


def init_params():
    rng = np.random.default_rng(SEED)
    return {"W1": rng.normal(0, 1 / np.sqrt(2), size=(2, HIDDEN)), "b1": np.zeros(HIDDEN),
            "W2": rng.normal(0, 1 / np.sqrt(HIDDEN), size=(HIDDEN,)), "b2": np.zeros(1)}


# ---------------------------------------------------------------------------------------------
def train_engine(X, y, p0):
    W1 = [[Value(p0["W1"][i, j]) for j in range(HIDDEN)] for i in range(2)]
    b1 = [Value(v) for v in p0["b1"]]
    W2 = [Value(v) for v in p0["W2"]]
    b2 = Value(p0["b2"][0])
    params = [w for row in W1 for w in row] + b1 + W2 + [b2]
    rows = [(list(map(float, x)), float(t)) for x, t in zip(X, y)]
    losses = []
    for _ in range(EPOCHS):
        terms = []
        for (x0, x1), t in rows:
            h = [(W1[0][j] * x0 + W1[1][j] * x1 + b1[j]).relu() for j in range(HIDDEN)]
            p = (total(W2[j] * h[j] for j in range(HIDDEN)) + b2).sigmoid()
            terms.append(p.log() * t + (1.0 - p).log() * (1.0 - t))
        loss = total(terms) * (-1.0 / len(rows))
        for q in params:
            q.grad = 0.0
        loss.backward()
        for q in params:
            q.data -= LR * q.grad
        losses.append(loss.data)
    final = {"W1": np.array([[w.data for w in row] for row in W1]), "b1": np.array([v.data for v in b1]),
             "W2": np.array([v.data for v in W2]), "b2": np.array([b2.data])}
    return losses, final


def train_torch(X, y, p0):
    import torch

    Xt, yt = torch.tensor(X), torch.tensor(y)
    W1, b1, W2, b2 = (torch.tensor(p0[k], dtype=torch.float64, requires_grad=True) for k in ("W1", "b1", "W2", "b2"))
    losses = []
    for _ in range(EPOCHS):
        h = torch.relu(Xt @ W1 + b1)
        p = torch.sigmoid(h @ W2 + b2)
        loss = -(yt * torch.log(p) + (1 - yt) * torch.log(1 - p)).mean()
        for q in (W1, b1, W2, b2):
            q.grad = None
        loss.backward()
        with torch.no_grad():
            for q in (W1, b1, W2, b2):
                q -= LR * q.grad
        losses.append(float(loss.item()))
    return losses, {k: v.detach().numpy().copy() for k, v in zip(("W1", "b1", "W2", "b2"), (W1, b1, W2, b2))}


def predict(params, X):
    h = np.maximum(X @ params["W1"] + params["b1"], 0)
    return 1 / (1 + np.exp(-(h @ params["W2"] + params["b2"][0])))


def timed(fn, *args):
    times, result = [], None
    for _ in range(SPEED_RUNS):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - t0)
    return times, result


def plot(X, y, losses_e, losses_t, params):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, AXIS, SURFACE = "#1f1f1d", "#6b6b66", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    ax = axes[0]
    ep = np.arange(1, len(losses_e) + 1)
    ax.plot(ep, losses_e, color=BLUE, linewidth=4, label="our engine")
    if losses_t:
        ax.plot(ep, losses_t, color=ORANGE, linewidth=2, linestyle=(0, (4, 2)), label="PyTorch")
    ax.set_title("BCE loss per epoch (curves coincide)", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("epoch", fontsize=8, color=MUTED)
    ax.set_ylabel("loss", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 300),
                         np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 300))
    zz = predict(params, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[0, .5, 1], colors=["#dbe8f8", "#fbe1d5"], alpha=0.9)
    ax.contour(xx, yy, zz, levels=[.5], colors=[MUTED], linewidths=1)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=18, color=BLUE, edgecolor=SURFACE, linewidth=0.8, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=18, color=ORANGE, edgecolor=SURFACE, linewidth=0.8, label="class 1")
    ax.set_title("Decision boundary of the engine-trained model", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    for a in axes:
        a.set_facecolor(SURFACE)
        a.grid(color=GRID, linewidth=0.6)
        a.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(AXIS)
    fig.suptitle(f"Lab 5 - make_moons, 2 -> {HIDDEN} -> 1, full-batch GD lr {LR}, {EPOCHS} epochs",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "moons.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    set_seeds()
    X, y = data()
    p0 = init_params()

    t_eng, (losses_e, params_e) = timed(train_engine, X, y, p0)
    print(f"engine : median {statistics.median(t_eng):.3f} s over {SPEED_RUNS} runs, final loss {losses_e[-1]:.6f}")
    try:
        import torch  # noqa: F401
        t_tor, (losses_t, params_t) = timed(train_torch, X, y, p0)
        print(f"PyTorch: median {statistics.median(t_tor):.4f} s over {SPEED_RUNS} runs, final loss {losses_t[-1]:.6f}")
    except ImportError:
        t_tor, losses_t, params_t = [], [], None

    acc_e = float(((predict(params_e, X) > 0.5) == y).mean())
    acc_t = float(((predict(params_t, X) > 0.5) == y).mean()) if params_t else None
    max_loss_diff = max(abs(a - b) for a, b in zip(losses_e, losses_t)) if losses_t else None
    max_param_diff = (max(float(np.max(np.abs(params_e[k] - params_t[k]))) for k in params_e)
                      if params_t else None)

    with open(RESULTS_DIR / "moons_training.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["epoch", "loss_engine", "loss_torch", "abs_diff"])
        for i, le in enumerate(losses_e):
            lt = losses_t[i] if losses_t else ""
            w.writerow([i + 1, f"{le:.12f}", f"{lt:.12f}" if lt != "" else "", f"{abs(le - lt):.3e}" if lt != "" else ""])
    speed = [{"implementation": "our engine (scalar Values)", "runs": SPEED_RUNS,
              "median_s": statistics.median(t_eng), "min_s": min(t_eng), "max_s": max(t_eng),
              "ms_per_epoch": statistics.median(t_eng) / EPOCHS * 1000}]
    if t_tor:
        speed.append({"implementation": "PyTorch (float64 tensors, eager)", "runs": SPEED_RUNS,
                      "median_s": statistics.median(t_tor), "min_s": min(t_tor), "max_s": max(t_tor),
                      "ms_per_epoch": statistics.median(t_tor) / EPOCHS * 1000})
    with open(RESULTS_DIR / "moons_speed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(speed[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in speed:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    write_json(RESULTS_DIR / "moons.json", {
        "dataset": f"make_moons(n_samples={N_SAMPLES}, noise={NOISE}, random_state={SEED})",
        "architecture": f"2 -> {HIDDEN} (ReLU) -> 1 (sigmoid), BCE loss",
        "optimizer": f"full-batch gradient descent, lr {LR}, {EPOCHS} epochs, float64",
        "parameters": 2 * HIDDEN + HIDDEN + HIDDEN + 1,
        "final_loss_engine": losses_e[-1], "final_loss_torch": losses_t[-1] if losses_t else None,
        "train_accuracy_engine": acc_e, "train_accuracy_torch": acc_t,
        "max_abs_loss_diff_over_epochs": max_loss_diff, "max_abs_param_diff_final": max_param_diff,
        "median_s_engine": statistics.median(t_eng),
        "median_s_torch": statistics.median(t_tor) if t_tor else None,
        "speedup_torch_over_engine": (statistics.median(t_eng) / statistics.median(t_tor)) if t_tor else None,
        "graph_nodes_per_epoch": epoch_graph_size(X, y, p0),
    })
    plot(X, y, losses_e, losses_t, params_e)
    if t_tor:
        print(f"PyTorch is {statistics.median(t_eng) / statistics.median(t_tor):.0f}x faster; max loss difference "
              f"{max_loss_diff:.2e}; train accuracy engine {acc_e:.3f} / torch {acc_t:.3f}")


def epoch_graph_size(X, y, p0) -> int:
    """Number of Value nodes the engine builds for ONE epoch's loss."""
    W1 = [[Value(p0["W1"][i, j]) for j in range(HIDDEN)] for i in range(2)]
    b1 = [Value(v) for v in p0["b1"]]
    W2 = [Value(v) for v in p0["W2"]]
    b2 = Value(p0["b2"][0])
    terms = []
    for (x0, x1), t in zip(X, y):
        h = [(W1[0][j] * float(x0) + W1[1][j] * float(x1) + b1[j]).relu() for j in range(HIDDEN)]
        p = (total(W2[j] * h[j] for j in range(HIDDEN)) + b2).sigmoid()
        terms.append(p.log() * float(t) + (1.0 - p).log() * (1.0 - float(t)))
    loss = total(terms) * (-1.0 / len(X))
    return len(loss.topological_order())


if __name__ == "__main__":
    main()
