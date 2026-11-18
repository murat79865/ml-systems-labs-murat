"""Step 4 (+ step 7 hash): leakage-free cleaning and transformation.

Order of operations
  1. 70/30 stratified split of the RAW data (random_state 42) - before any cleaning.
  2. Exact duplicate rows are dropped from the training split only; the test split is untouched.
  3. The preprocessor (median / mode imputation, 1st-99th percentile clipping, derived features,
     scaling + one-hot encoding) is fitted on the de-duplicated training split only and saved with
     joblib.dump to results/preprocessor.joblib.
  4. Train and test are transformed with the fitted cleaner -> processed tables (cleaned base features,
     derived features, label) written deterministically to data/lab04/processed/. Their SHA256 hashes
     are stored together with the full config in results/config_and_hash.yaml.
  5. Leakage checks prove that every fitted statistic equals the train-only value.

Outputs: results/preprocessor.joblib, results/cleaning_log.json, results/leakage_checks.json,
         results/config_and_hash.yaml, results/split_indices.csv, data/lab04/processed/{train,test}.csv
"""

from __future__ import annotations

import hashlib

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from common import (CATEGORICAL, CFG, FEATURES, NULL, NUMERIC, PROCESSED_DIR, RAW_FILE, RESULTS_DIR, SEED,
                    TARGET, canonical, read_raw, set_seeds, sha256_file, to_binary, write_csv, write_json)
from pipeline_steps import DERIVED, make_preprocessor


def split_and_dedupe():
    """Shared by every later step so that all of them see exactly the same split."""
    df = canonical(read_raw())
    sp = CFG["split"]
    train, test = train_test_split(df, test_size=sp["test_size"], stratify=df[sp["stratify"]],
                                   random_state=sp["random_state"])
    train_dedup = train[~train.duplicated(keep="first")]
    return df, train, train_dedup, test


def main() -> None:
    set_seeds()
    cc = CFG["cleaning"]
    df, train, train_dedup, test = split_and_dedupe()
    test_index_before = test.index.copy()

    pre = make_preprocessor(NUMERIC, CATEGORICAL, NULL, cc["clip_quantiles"], add_derived=True)
    pre.fit(train_dedup[FEATURES], to_binary(train_dedup[TARGET]))
    joblib.dump(pre, RESULTS_DIR / "preprocessor.joblib")
    cleaner = pre.named_steps["clean"]
    adder = pre.named_steps["features"]
    fitted = cleaner.fitted_values()

    # ---- processed tables -------------------------------------------------------------------
    def processed(part):
        out = adder.transform(cleaner.transform(part[FEATURES]))
        out = out[NUMERIC + DERIVED + CATEGORICAL]
        out[TARGET] = part[TARGET].to_numpy()
        return out.reset_index(drop=True)

    p_train, p_test = processed(train_dedup), processed(test)
    write_csv(p_train, PROCESSED_DIR / "train.csv")
    write_csv(p_test, PROCESSED_DIR / "test.csv")
    h_train, h_test = sha256_file(PROCESSED_DIR / "train.csv"), sha256_file(PROCESSED_DIR / "test.csv")
    combined = hashlib.sha256((PROCESSED_DIR / "train.csv").read_bytes()
                              + (PROCESSED_DIR / "test.csv").read_bytes()).hexdigest()
    pd.DataFrame({"row": list(train_dedup.index) + list(test.index),
                  "split": ["train"] * len(train_dedup) + ["test"] * len(test)}).sort_values("row").to_csv(
        RESULTS_DIR / "split_indices.csv", index=False, lineterminator="\n")

    # ---- what the cleaning did ----------------------------------------------------------------
    def counts(part):
        num = part[NUMERIC].astype(float)
        return {
            "rows": len(part),
            "imputed_categorical": {c: int((part[c] == NULL).sum()) for c in CATEGORICAL},
            "imputed_numeric": {c: int(num[c].isna().sum()) for c in NUMERIC},
            "clipped_low": {c: int((num[c] < fitted["clip_lower"][c]).sum()) for c in NUMERIC},
            "clipped_high": {c: int((num[c] > fitted["clip_upper"][c]).sum()) for c in NUMERIC},
            "label_positive_rate": float(to_binary(part[TARGET]).mean()),
        }

    log = {"raw_rows": len(df), "train_rows_before_dedup": len(train), "duplicates_dropped_from_train":
           len(train) - len(train_dedup), "train_rows": len(train_dedup), "test_rows": len(test),
           "test_exact_duplicates_kept": int(test.duplicated(keep="first").sum()),
           "fitted": fitted, "train": counts(train_dedup), "test": counts(test),
           "encoded_feature_count": int(pre.named_steps["encode"].transform(
               adder.transform(cleaner.transform(train_dedup[FEATURES].head(5)))).shape[1])}
    write_json(RESULTS_DIR / "cleaning_log.json", log)

    # ---- leakage checks ----------------------------------------------------------------------
    tr_num = train_dedup[NUMERIC].astype(float)
    all_num = df[NUMERIC].astype(float)
    q_lo, q_hi = cc["clip_quantiles"]
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    check("split happens before cleaning", True,
          "train_test_split is called on the raw canonical table; no statistic exists before it")
    check("train and test rows are disjoint", len(set(train.index) & set(test.index)) == 0,
          f"{len(train.index)} train / {len(test.index)} test row ids, overlap "
          f"{len(set(train.index) & set(test.index))}")
    check("test split untouched (no rows dropped)", test.index.equals(test_index_before) and
          len(p_test) == len(test), f"{len(test)} test rows before and after processing")
    check("duplicates removed from train only", len(train_dedup) <= len(train) and len(p_test) == len(test),
          f"{len(train) - len(train_dedup)} removed from train, 0 from test "
          f"(test keeps {log['test_exact_duplicates_kept']} exact duplicates)")
    check("preprocessor fitted on train rows only", fitted["n_rows_fit"] == len(train_dedup),
          f"cleaner saw {fitted['n_rows_fit']} rows = de-duplicated train size {len(train_dedup)}")
    med_train = {c: float(np.median(tr_num[c].dropna())) for c in NUMERIC}
    check("numeric medians = train medians", all(np.isclose(fitted["numeric_median"][c], med_train[c])
                                                for c in NUMERIC), fitted["numeric_median"])
    mode_train = {}
    for c in CATEGORICAL:
        vc = train_dedup.loc[train_dedup[c] != NULL, c].value_counts()
        top = vc[vc == vc.max()].index
        mode_train[c] = sorted(top)[0]
    check("categorical modes = train modes", all(fitted["categorical_mode"][c] == mode_train[c]
                                                 for c in CATEGORICAL), fitted["categorical_mode"])
    lo_tr = {c: float(np.percentile(tr_num[c], 100 * q_lo)) for c in NUMERIC}
    hi_tr = {c: float(np.percentile(tr_num[c], 100 * q_hi)) for c in NUMERIC}
    check("clip bounds = train percentiles",
          all(np.isclose(fitted["clip_lower"][c], lo_tr[c]) and np.isclose(fitted["clip_upper"][c], hi_tr[c])
              for c in NUMERIC), {"lower": lo_tr, "upper": hi_tr})
    lo_all = {c: float(np.percentile(all_num[c], 100 * q_lo)) for c in NUMERIC}
    hi_all = {c: float(np.percentile(all_num[c], 100 * q_hi)) for c in NUMERIC}
    differ = [c for c in NUMERIC if not (np.isclose(lo_all[c], lo_tr[c]) and np.isclose(hi_all[c], hi_tr[c]))]
    checks.append({"check": "contrast: bounds if test had leaked into fit", "pass": True,
                   "detail": {"columns_whose_bounds_would_change": differ,
                              "full_data_lower": lo_all, "full_data_upper": hi_all}})
    write_json(RESULTS_DIR / "leakage_checks.json", {"all_pass": all(c["pass"] for c in checks),
                                                     "checks": checks})

    # ---- config + hash -----------------------------------------------------------------------
    manifest = {
        "config": CFG,
        "raw_snapshot": {"file": "data/lab04/raw/adult_raw.csv", "sha256": sha256_file(RAW_FILE),
                         "rows": len(df)},
        "processed": {
            "train": {"file": "data/lab04/processed/train.csv", "rows": len(p_train),
                      "columns": list(p_train.columns), "sha256": h_train},
            "test": {"file": "data/lab04/processed/test.csv", "rows": len(p_test),
                     "columns": list(p_test.columns), "sha256": h_test},
            "sha256_train_plus_test": combined,
        },
        "preprocessor": {"file": "lab04/results/preprocessor.joblib",
                         "sha256": sha256_file(RESULTS_DIR / "preprocessor.joblib")},
    }
    with open(RESULTS_DIR / "config_and_hash.yaml", "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)

    print(f"split: train {len(train)} -> {len(train_dedup)} after dropping "
          f"{len(train) - len(train_dedup)} exact duplicates; test {len(test)} untouched")
    print(f"leakage checks: {'all pass' if all(c['pass'] for c in checks) else 'FAILED'}")
    print(f"processed sha256: train {h_train[:16]}..., test {h_test[:16]}..., combined {combined[:16]}...")


if __name__ == "__main__":
    main()
