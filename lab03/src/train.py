"""Step 4: train MLP, CNN and Attention for 5 epochs with Adam (lr 1e-3), batch 128, on CPU.

Each model starts from seed 42. After every epoch the training loss/accuracy (over the epoch) and the
validation loss/accuracy (5k held-out images) are logged; after epoch 5 the official 10k test set is
evaluated once.

Outputs: results/training_log.csv, results/test_accuracy.csv, results/model_<name>.pt
"""

from __future__ import annotations

import csv
import time

import torch
from torch import nn

import models
from common import EPOCHS, LR, MODEL_NAMES, RESULTS_DIR, get_loaders, set_seeds


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        out = model(x)
        loss_sum += loss_fn(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main() -> None:
    log_rows, test_rows = [], []
    for name in MODEL_NAMES:
        set_seeds()
        train_loader, val_loader, test_loader = get_loaders()
        model = models.build(name)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()
        total_time = 0.0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            t0 = time.perf_counter()
            seen, correct, loss_sum = 0, 0, 0.0
            for x, y in train_loader:
                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
                loss_sum += loss.item() * y.size(0)
                correct += (out.argmax(1) == y).sum().item()
                seen += y.size(0)
            secs = time.perf_counter() - t0
            total_time += secs
            val_loss, val_acc = evaluate(model, val_loader, loss_fn)
            log_rows.append({"model": name, "epoch": epoch, "train_loss": f"{loss_sum / seen:.4f}",
                             "train_acc": f"{correct / seen:.4f}", "val_loss": f"{val_loss:.4f}",
                             "val_acc": f"{val_acc:.4f}", "epoch_seconds": f"{secs:.2f}"})
            print(f"{name:<10} epoch {epoch}/{EPOCHS}  train loss {loss_sum / seen:.4f} acc {correct / seen:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f} | {secs:.1f}s", flush=True)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn)
        torch.save(model.state_dict(), RESULTS_DIR / f"model_{name}.pt")
        test_rows.append({"model": name, "epochs": EPOCHS, "optimizer": "Adam", "lr": LR, "batch_size": 128,
                          "final_val_acc": log_rows[-1]["val_acc"], "test_loss": f"{test_loss:.4f}",
                          "test_acc": f"{test_acc:.4f}", "train_seconds": f"{total_time:.1f}"})
        print(f"{name:<10} TEST accuracy {test_acc:.4f}\n", flush=True)

    for fname, rows in (("training_log.csv", log_rows), ("test_accuracy.csv", test_rows)):
        with open(RESULTS_DIR / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
