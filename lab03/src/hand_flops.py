"""Hand calculation of parameters and FLOPs with the Ch. 4 formulas (per ONE input image).

  Linear  : FLOPs = 2 x in x out                                   (x tokens, if applied per token)
  Conv2d  : FLOPs = 2 x K x K x C_in x C_out x H_out x W_out
  Attention (seq tokens, width d):
            QKV projections    2 x seq x d x 3d
            scores Q K^T       2 x seq^2 x d
            weighted sum A V   2 x seq^2 x d
            output projection  2 x seq x d x d
  Only multiply-adds of the matrix products are counted (1 multiply-add = 2 FLOPs); biases, ReLU,
  max-pool, softmax, LayerNorm and the residual additions are left out, as in the Ch. 4 formulas.
  Parameters: Linear in x out + out; Conv2d K x K x C_in x C_out + C_out; LayerNorm 2 x d.
"""

from __future__ import annotations


def linear_flops(n_in: int, n_out: int, tokens: int = 1) -> int:
    return 2 * n_in * n_out * tokens


def conv_flops(k: int, c_in: int, c_out: int, h_out: int, w_out: int) -> int:
    return 2 * k * k * c_in * c_out * h_out * w_out


def attention_parts(seq: int, d: int, d_ff: int) -> dict[str, int]:
    return {
        "QKV projections (2*seq*d*3d)": 2 * seq * d * 3 * d,
        "scores QK^T (2*seq^2*d)": 2 * seq * seq * d,
        "weighted sum AV (2*seq^2*d)": 2 * seq * seq * d,
        "output projection (2*seq*d*d)": 2 * seq * d * d,
        "feed-forward d->d_ff (2*seq*d*d_ff)": 2 * seq * d * d_ff,
        "feed-forward d_ff->d (2*seq*d_ff*d)": 2 * seq * d_ff * d,
    }


def model_breakdown() -> dict[str, list[dict]]:
    """Every layer of every model: hand parameter count and hand FLOPs for one image."""
    mlp = [
        {"layer": "Linear 784->256", "type": "Linear", "params": 784 * 256 + 256, "flops": linear_flops(784, 256)},
        {"layer": "Linear 256->128", "type": "Linear", "params": 256 * 128 + 128, "flops": linear_flops(256, 128)},
        {"layer": "Linear 128->10", "type": "Linear", "params": 128 * 10 + 10, "flops": linear_flops(128, 10)},
    ]
    cnn = [
        {"layer": "Conv2d 1->16, 3x3, out 28x28", "type": "Conv2d", "params": 3 * 3 * 1 * 16 + 16,
         "flops": conv_flops(3, 1, 16, 28, 28)},
        {"layer": "Conv2d 16->32, 3x3, out 14x14", "type": "Conv2d", "params": 3 * 3 * 16 * 32 + 32,
         "flops": conv_flops(3, 16, 32, 14, 14)},
        {"layer": "Linear 1568->10", "type": "Linear", "params": 1568 * 10 + 10, "flops": linear_flops(1568, 10)},
    ]
    seq, d, d_ff = 16, 64, 128
    parts = attention_parts(seq, d, d_ff)
    enc_params = (3 * d * d + 3 * d) + (d * d + d) + (d * d_ff + d_ff) + (d_ff * d + d) + 2 * (2 * d)
    att = [{"layer": "Patch projection Linear 49->64 (x16 tokens)", "type": "Linear",
            "params": 49 * 64 + 64, "flops": linear_flops(49, 64, tokens=seq)}]
    for i in (1, 2):
        att.append({"layer": f"Transformer encoder layer {i}", "type": "Attention",
                    "params": enc_params, "flops": sum(parts.values())})
    att.append({"layer": "Linear 64->10 (after mean pool)", "type": "Linear",
                "params": 64 * 10 + 10, "flops": linear_flops(64, 10)})
    return {"mlp": mlp, "cnn": cnn, "attention": att}


def totals() -> dict[str, dict[str, int]]:
    return {m: {"params": sum(r["params"] for r in rows), "flops": sum(r["flops"] for r in rows)}
            for m, rows in model_breakdown().items()}


# The single layers worked out by hand in the report (one of each type).
ONE_LAYER = {
    "Linear": {"model": "mlp", "layer": "Linear 784->256", "formula": "2 x 784 x 256",
               "flops": linear_flops(784, 256)},
    "Conv2d": {"model": "cnn", "layer": "Conv2d 16->32, 3x3, out 14x14",
               "formula": "2 x 3 x 3 x 16 x 32 x 14 x 14", "flops": conv_flops(3, 16, 32, 14, 14)},
    "Attention": {"model": "attention", "layer": "Transformer encoder layer 1 (seq 16, d 64, d_ff 128)",
                  "formula": "QKV + QK^T + AV + out-proj + FFN", "flops": sum(attention_parts(16, 64, 128).values())},
}


if __name__ == "__main__":
    for m, t in totals().items():
        print(f"{m:<10} params {t['params']:>8,}  FLOPs/image {t['flops']:>10,}")
    for k, v in attention_parts(16, 64, 128).items():
        print(f"  {k:<40} {v:>9,}")
