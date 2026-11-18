"""Step 5: do the derived features help? LogisticRegression baseline with and without them.

Both variants use the same split, the same de-duplicated training rows and the same train-only
preprocessing; the only difference is FeatureAdder(add_derived). Reported: test accuracy and test F1
(positive class >50K), plus 5-fold stratified cross-validation F1 on the training split (preprocessing
re-fitted inside every fold, so there is no leakage across folds either).

Outputs: results/feature_eval.csv, results/feature_eval.json
"""

from __future__ import annotations

import csv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from clean_transform import split_and_dedupe
from common import CATEGORICAL, CFG, FEATURES, NULL, NUMERIC, RESULTS_DIR, SEED, TARGET, set_seeds, to_binary, write_json
from pipeline_steps import DERIVED, make_preprocessor


def model(add_derived: bool) -> Pipeline:
    pre = make_preprocessor(NUMERIC, CATEGORICAL, NULL, CFG["cleaning"]["clip_quantiles"], add_derived)
    lr = LogisticRegression(max_iter=CFG["model"]["max_iter"], random_state=SEED)
    return Pipeline(pre.steps + [("model", lr)])


def main() -> None:
    set_seeds()
    _, _, train, test = split_and_dedupe()
    Xtr, ytr = train[FEATURES], to_binary(train[TARGET])
    Xte, yte = test[FEATURES], to_binary(test[TARGET])
    cv = StratifiedKFold(n_splits=CFG["model"]["cv_folds"], shuffle=True, random_state=SEED)
    rows, detail = [], {}
    for add in (False, True):
        name = "with derived features" if add else "without derived features"
        m = model(add).fit(Xtr, ytr)
        pred = m.predict(Xte)
        acc, f1 = accuracy_score(yte, pred), f1_score(yte, pred)
        cv_f1 = cross_val_score(model(add), Xtr, ytr, cv=cv, scoring="f1")
        n_iter = int(np.max(m.named_steps["model"].n_iter_))
        rows.append({"variant": name, "features_in": len(NUMERIC) + len(CATEGORICAL) + (2 if add else 0),
                     "encoded_columns": int(m.named_steps["model"].coef_.shape[1]),
                     "test_accuracy": f"{acc:.4f}", "test_f1": f"{f1:.4f}",
                     "cv_f1_mean": f"{cv_f1.mean():.4f}", "cv_f1_std": f"{cv_f1.std():.4f}",
                     "lr_iterations": n_iter, "converged": n_iter < CFG["model"]["max_iter"]})
        if add:
            names = list(m.named_steps["encode"].get_feature_names_out())
            coef = m.named_steps["model"].coef_[0]
            detail["derived_coefficients_standardised"] = {d: float(coef[names.index(f"num__{d}")]) for d in DERIVED}
        detail[name] = {"cv_f1_folds": [float(v) for v in cv_f1]}
        print(f"{name:<26} test acc {acc:.4f}  test F1 {f1:.4f}  CV F1 {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
    a, b = rows
    detail["delta_test_accuracy"] = float(b["test_accuracy"]) - float(a["test_accuracy"])
    detail["delta_test_f1"] = float(b["test_f1"]) - float(a["test_f1"])
    detail["delta_cv_f1"] = float(b["cv_f1_mean"]) - float(a["cv_f1_mean"])
    with open(RESULTS_DIR / "feature_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    write_json(RESULTS_DIR / "feature_eval.json", detail)


if __name__ == "__main__":
    main()
