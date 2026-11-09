"""Step 2: train the two baselines on Breast Cancer Wisconsin and record test accuracy.

Output: results/baseline_accuracy.csv (accuracy to 4 decimals) and results/split_info.json.
"""

from __future__ import annotations

import csv

import numpy as np
from sklearn.metrics import accuracy_score

from common import MODEL_NAMES, RESULTS_DIR, load_split, make_model, set_seeds, write_json


def main() -> None:
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()

    write_json(RESULTS_DIR / "split_info.json", {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "n_samples": int(len(y_train) + len(y_test)),
        "n_features": int(X_train.shape[1]),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_class_counts": {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "test_class_counts": {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        "split": "70/30 stratified, random_state=42",
    })

    rows = []
    for name in MODEL_NAMES:
        model = make_model(name).fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({"model": name, "test_accuracy": f"{acc:.4f}"})
        print(f"{name:<22} test accuracy = {acc:.4f}")

    with open(RESULTS_DIR / "baseline_accuracy.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "test_accuracy"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
