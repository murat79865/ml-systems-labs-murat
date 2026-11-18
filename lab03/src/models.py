"""Step 2: the three architectures, exactly as specified in the lab manual.

MLP       : Flatten 784 -> Linear 256 -> ReLU -> Linear 128 -> ReLU -> Linear 10
CNN       : Conv2d(1,16,3,p=1) -> ReLU -> MaxPool2 -> Conv2d(16,32,3,p=1) -> ReLU -> MaxPool2
            -> Flatten -> Linear(32*7*7, 10)
Attention : 28x28 image -> 16 patches of 7x7 (49 pixels) -> Linear 49->64 -> 2 x TransformerEncoderLayer
            (d_model=64, nhead=4, dim_feedforward=128) -> mean pool over the 16 tokens -> Linear 64->10
            No positional encoding is added (none is specified); see the report for its effect.
"""

from __future__ import annotations

import torch
from torch import nn

PATCH = 7
SEQ = (28 // PATCH) ** 2  # 16 tokens
D_MODEL, N_HEAD, D_FF, N_LAYERS = 64, 4, 128, 2


def make_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )


def make_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 10),
    )


class PatchAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(PATCH * PATCH, D_MODEL)
        layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_FF,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS, enable_nested_tensor=False)
        self.head = nn.Linear(D_MODEL, 10)

    @staticmethod
    def patchify(x: torch.Tensor) -> torch.Tensor:
        """(B, 1, 28, 28) -> (B, 16, 49): non-overlapping 7x7 patches in row-major order."""
        b = x.shape[0]
        x = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)  # (B, 1, 4, 4, 7, 7)
        return x.reshape(b, SEQ, PATCH * PATCH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(self.patchify(x))  # (B, 16, 64)
        tokens = self.encoder(tokens)          # (B, 16, 64)
        return self.head(tokens.mean(dim=1))   # mean pool -> (B, 10)


BUILDERS = {"mlp": make_mlp, "cnn": make_cnn, "attention": PatchAttention}


def build(name: str) -> nn.Module:
    return BUILDERS[name]()


def disable_encoder_fastpath(model: nn.Module) -> nn.Module:
    """Force the regular (module-by-module) path of TransformerEncoderLayer in eval mode.

    In eval/no-grad mode PyTorch runs each encoder layer through one fused kernel
    (torch._transformer_encoder_layer_fwd), so per-module profilers never see its matmuls. The fast
    path is skipped when this flag is False; the computation itself is unchanged.
    """
    for m in model.modules():
        if isinstance(m, nn.TransformerEncoderLayer):
            m.activation_relu_or_gelu = False
    return model
