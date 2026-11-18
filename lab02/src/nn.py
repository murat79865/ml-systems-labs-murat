"""A two-hidden-layer neural network written from scratch in NumPy (float64).

Architecture (lab manual, step 2):
    input 64 -> Linear(64, W) -> ReLU -> Linear(W, W) -> ReLU -> Linear(W, 10) -> softmax

Notation (N = batch size):
    Z1 = X  W1 + b1      A1 = ReLU(Z1)          X  : (N, 64)
    Z2 = A1 W2 + b2      A2 = ReLU(Z2)          W1 : (64, W)   b1 : (W,)
    Z3 = A2 W3 + b3      P  = softmax(Z3)       W2 : (W, W)    b2 : (W,)
    L  = -(1/N) sum_n sum_k Y[n,k] log P[n,k]   W3 : (W, 10)   b3 : (10,)

Backward pass (chain rule, derived in the report / on the handwritten design page):
    dZ3 = (P - Y) / N
    dW3 = A2^T dZ3            db3 = sum_n dZ3        dA2 = dZ3 W3^T
    dZ2 = dA2 * 1[Z2 > 0]
    dW2 = A1^T dZ2            db2 = sum_n dZ2        dA1 = dZ2 W2^T
    dZ1 = dA1 * 1[Z1 > 0]
    dW1 = X^T  dZ1            db1 = sum_n dZ1
"""

from __future__ import annotations

import numpy as np

PARAM_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")
N_IN, N_OUT = 64, 10


def param_shapes(width: int) -> dict[str, tuple[int, ...]]:
    return {
        "W1": (N_IN, width), "b1": (width,),
        "W2": (width, width), "b2": (width,),
        "W3": (width, N_OUT), "b3": (N_OUT,),
    }


def init_params(width: int, rng: np.random.Generator, std: float) -> dict[str, np.ndarray]:
    """Weights ~ N(0, std) (std = standard deviation), biases = 0."""
    params = {}
    for name, shape in param_shapes(width).items():
        if name.startswith("W"):
            params[name] = rng.normal(0.0, std, size=shape).astype(np.float64)
        else:
            params[name] = np.zeros(shape, dtype=np.float64)
    return params


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax; subtracting the row max keeps exp() from overflowing."""
    shifted = z - z.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def forward(params: dict, X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Layer-by-layer forward pass. Returns probabilities P (N, 10) and the cache for backward."""
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = relu(Z2)
    Z3 = A2 @ params["W3"] + params["b3"]
    P = softmax(Z3)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "P": P}
    return P, cache


def cross_entropy(cache: dict, Y: np.ndarray) -> float:
    """Mean cross-entropy over the batch, computed from the logits with log-softmax (stable)."""
    return float(-(Y * log_softmax(cache["Z3"])).sum() / Y.shape[0])


def loss(params: dict, X: np.ndarray, Y: np.ndarray) -> float:
    _, cache = forward(params, X)
    return cross_entropy(cache, Y)


def backward(params: dict, cache: dict, Y: np.ndarray) -> dict[str, np.ndarray]:
    """Analytic gradients of the mean cross-entropy w.r.t. every parameter."""
    N = Y.shape[0]
    dZ3 = (cache["P"] - Y) / N
    dW3 = cache["A2"].T @ dZ3
    db3 = dZ3.sum(axis=0)
    dA2 = dZ3 @ params["W3"].T
    dZ2 = dA2 * (cache["Z2"] > 0)
    dW2 = cache["A1"].T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (cache["Z1"] > 0)
    dW1 = cache["X"].T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def sgd_step(params: dict, grads: dict, lr: float) -> None:
    for name in PARAM_NAMES:
        params[name] -= lr * grads[name]


def predict(params: dict, X: np.ndarray) -> np.ndarray:
    P, _ = forward(params, X)
    return P.argmax(axis=1)


def accuracy(params: dict, X: np.ndarray, y: np.ndarray) -> float:
    return float((predict(params, X) == y).mean())


def n_params(width: int) -> int:
    return int(sum(np.prod(s) for s in param_shapes(width).values()))


def matmul_flops(width: int, n: int) -> dict[str, int]:
    """Multiply-add FLOPs (2 * m * k * n per matmul) of the forward and backward passes."""
    fwd = 2 * n * (N_IN * width + width * width + width * N_OUT)
    # backward: dW3, dA2, dW2, dA1, dW1 (no gradient w.r.t. the input X is needed)
    bwd = 2 * n * (width * N_OUT + N_OUT * width + width * width + width * width + N_IN * width)
    return {"forward": fwd, "backward": bwd}
