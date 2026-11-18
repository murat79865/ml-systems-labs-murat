"""Step 6: label-quality checks for Adult.

(a) 50 random rows (seed 42) are drawn and re-checked one by one against the data dictionary:
    allowed category values, schema ranges, education vs education-num, relationship vs sex, missing
    fields ('?'). Every row gets a per-rule result and a verdict (results/label_audit_sample.csv/.md).
(b) Duplicate-conflict audit: rows with identical 14 feature values but different income labels.
    Reported for the full dataset and for the training split (results/label_conflicts.csv).
Adult's labels come from census records (objective), so the subjective-label relabelling /
Cohen's kappa variant of this step does not apply.

Outputs: results/label_audit_sample.csv, results/label_audit_sample.md, results/label_conflicts.csv,
         results/label_audit.json
"""

from __future__ import annotations

import pandas as pd

from clean_transform import split_and_dedupe
from common import CATEGORICAL, CFG, DICTIONARY, EDUCATION_NUM, FEATURES, NULL, RESULTS_DIR, SEED, TARGET, write_json

RANGES = CFG["schema"]["ranges"]


def row_checks(r: pd.Series) -> dict:
    out = {}
    bad_cat = [c for c in CATEGORICAL + [TARGET] if r[c] != NULL and r[c] not in DICTIONARY[c]]
    out["categories_in_dictionary"] = "ok" if not bad_cat else "FAIL: " + ", ".join(bad_cat)
    bad_rng = [c for c, (lo, hi) in RANGES.items() if not (lo <= float(r[c]) <= hi)]
    out["values_in_range"] = "ok" if not bad_rng else "FAIL: " + ", ".join(bad_rng)
    exp = EDUCATION_NUM.get(r["education"])
    out["education_vs_num"] = "ok" if exp is None or exp == int(r["education-num"]) else \
        f"FAIL: {r['education']} should be {exp}"
    rel, sex = r["relationship"], r["sex"]
    out["relationship_vs_sex"] = "FAIL" if (rel == "Husband" and sex != "Male") or \
        (rel == "Wife" and sex != "Female") else "ok"
    out["label_valid"] = "ok" if r[TARGET] in DICTIONARY[TARGET] else "FAIL"
    miss = [c for c in CATEGORICAL if r[c] == NULL]
    out["missing_fields"] = ", ".join(miss) if miss else "-"
    fails = [k for k, v in out.items() if str(v).startswith("FAIL")]
    out["verdict"] = "impossible value" if fails else ("valid (has '?')" if miss else "valid")
    return out


def conflicts(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    g = df.groupby(FEATURES, dropna=False)[TARGET].nunique()
    bad_keys = g[g > 1].index
    if len(bad_keys) == 0:
        return df.iloc[0:0], 0, 0
    key = df[FEATURES].apply(tuple, axis=1)
    rows = df[key.isin(set(bad_keys))].copy()
    rows["group"] = key[rows.index].map({k: i for i, k in enumerate(bad_keys)})
    return rows.sort_values(["group", TARGET]), len(bad_keys), len(rows)


def main() -> None:
    df, _, train_dedup, _ = split_and_dedupe()
    n = CFG["label_audit"]["sample_size"]
    sample = df.sample(n=n, random_state=SEED)
    checks = pd.DataFrame([row_checks(r) for _, r in sample.iterrows()], index=sample.index)
    audit = pd.concat([sample, checks], axis=1)
    audit.index.name = "row"
    audit.to_csv(RESULTS_DIR / "label_audit_sample.csv", lineterminator="\n")

    cols = ["age", "education", "education-num", "relationship", "sex", "hours-per-week", "capital-gain",
            "capital-loss", TARGET, "missing_fields", "verdict"]
    md = ["# 50-row re-check against the data dictionary (step 6)\n",
          "Rules applied to every row: categories in the UCI dictionary, schema ranges, education vs "
          "education-num, Husband/Male and Wife/Female, valid label. Full columns in "
          "`label_audit_sample.csv`.\n",
          "| row | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in audit.iterrows():
        md.append(f"| {i} | " + " | ".join(str(r[c]) for c in cols) + " |")
    (RESULTS_DIR / "label_audit_sample.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    conf_all, g_all, r_all = conflicts(df)
    conf_tr, g_tr, r_tr = conflicts(train_dedup)
    conf_all.to_csv(RESULTS_DIR / "label_conflicts.csv", lineterminator="\n")
    dup_feature_groups = int((df.groupby(FEATURES, dropna=False).size() > 1).sum())
    summary = {
        "sample_size": n, "sample_rows": [int(i) for i in sample.index],
        "sample_verdicts": checks["verdict"].value_counts().to_dict(),
        "sample_rows_with_impossible_values": int((checks["verdict"] == "impossible value").sum()),
        "sample_rows_with_missing": int((checks["missing_fields"] != "-").sum()),
        "duplicate_feature_groups": dup_feature_groups,
        "conflict_groups_full_data": g_all, "conflict_rows_full_data": r_all,
        "conflict_groups_train": g_tr, "conflict_rows_train": r_tr,
        "full_data_rule_failures": {
            "education_vs_num": int((df["education"].map(EDUCATION_NUM) != df["education-num"]).sum()),
            "relationship_vs_sex": int((((df["relationship"] == "Husband") & (df["sex"] != "Male")) |
                                        ((df["relationship"] == "Wife") & (df["sex"] != "Female"))).sum()),
        },
    }
    write_json(RESULTS_DIR / "label_audit.json", summary)
    print(f"50-row re-check: {summary['sample_verdicts']}")
    print(f"duplicate-conflict audit: {g_all} groups / {r_all} rows with identical features but different "
          f"income (train split: {g_tr} groups / {r_tr} rows)")


if __name__ == "__main__":
    main()
