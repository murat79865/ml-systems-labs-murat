"""Step 3: validation. Run every schema check on the raw data and report violations.

Checks (severity "error" = breaks the schema, "warning" = suspicious but allowed, "info" = reported):
  columns   - exactly the 15 schema columns are present
  types     - int columns are numeric with whole-number values; category columns hold text labels
  ranges    - age 17-90, hours-per-week 1-99, capital-gain 0-99999, capital-loss 0-4356
  nulls     - '?' (or an empty field) means missing; allowed only in category columns
  domain    - every category value is in the UCI data dictionary (or '?'); income is >50K / <=50K
  duplicates- exact duplicate rows (all 15 columns) and duplicate feature rows
  consistency - education vs education-num, relationship Husband/Wife vs sex
No row is dropped or changed: the row count before and after validation is logged.

Outputs: results/validation_report.json, results/validation_report.md, results/validation_violations.csv
"""

from __future__ import annotations

import pandas as pd

from common import (CATEGORICAL, CFG, DICTIONARY, EDUCATION_NUM, FEATURES, NULL, NUMERIC, RESULTS_DIR,
                    TARGET, TYPES, canonical, read_raw, write_json)


def main() -> None:
    raw = read_raw()
    n_before = len(raw)
    df = canonical(raw)
    checks, violations = [], []

    def record(category, name, column, severity, bad_index, rule, note=""):
        bad_index = list(bad_index)
        checks.append({"category": category, "check": name, "column": column, "severity": severity,
                       "rule": rule, "violations": len(bad_index),
                       "status": "PASS" if not bad_index else ("FAIL" if severity == "error" else severity.upper()),
                       "example_rows": bad_index[:5], "note": note})
        for i in bad_index:
            violations.append({"row": int(i), "check": name, "column": column, "severity": severity,
                               "value": df.at[i, column] if column in df.columns else ""})

    # ---- columns ---------------------------------------------------------------------------
    expected = list(TYPES)
    missing_cols = [c for c in expected if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected]
    checks.append({"category": "columns", "check": "schema columns present", "column": "all",
                   "severity": "error", "rule": f"{len(expected)} columns: {', '.join(expected)}",
                   "violations": len(missing_cols) + len(extra_cols),
                   "status": "PASS" if not (missing_cols or extra_cols) else "FAIL",
                   "example_rows": [], "note": f"missing {missing_cols}, unexpected {extra_cols}"})

    # ---- types ------------------------------------------------------------------------------
    for c in NUMERIC:
        s = raw[c]
        is_num = pd.api.types.is_numeric_dtype(s)
        vals = pd.to_numeric(s, errors="coerce")
        bad = df.index[(vals.notna() & (vals % 1 != 0)) | (s.notna() & vals.isna())]
        record("types", f"{c} is int", c, "error", bad, "numeric, whole numbers",
               f"raw dtype {s.dtype}" + ("" if is_num else " (not numeric)"))
    for c in CATEGORICAL + [TARGET]:
        vals = df[c]
        numeric_like = pd.to_numeric(vals.where(vals != NULL), errors="coerce").notna()
        record("types", f"{c} is category", c, "error", df.index[numeric_like], "text labels",
               f"raw dtype {raw[CFG['dataset']['target_raw'] if c == TARGET else c].dtype}")

    # ---- ranges -----------------------------------------------------------------------------
    for c, (lo, hi) in CFG["schema"]["ranges"].items():
        vals = pd.to_numeric(df[c], errors="coerce")
        bad = df.index[vals.notna() & ((vals < lo) | (vals > hi))]
        record("ranges", f"{c} in [{lo}, {hi}]", c, "error", bad, f"{lo} <= {c} <= {hi}",
               f"observed min {vals.min():g}, max {vals.max():g}")

    # ---- null policy ------------------------------------------------------------------------
    null_counts = {}
    for c in FEATURES + [TARGET]:
        if c in NUMERIC:
            miss = df.index[df[c].isna()]
            null_counts[c] = len(miss)
            record("nulls", f"{c} has no missing", c, "error", miss, "numeric columns may not be missing")
        else:
            miss = df.index[df[c] == NULL]
            null_counts[c] = len(miss)
            sev = "error" if c == TARGET else "info"
            checks.append({"category": "nulls", "check": f"{c} missing ('{NULL}')", "column": c,
                           "severity": sev, "rule": f"'{NULL}' allowed in category columns" if c != TARGET
                           else "label may not be missing", "violations": len(miss) if c == TARGET else 0,
                           "status": ("PASS" if c != TARGET or not len(miss) else "FAIL"),
                           "example_rows": list(miss[:5]),
                           "note": f"{len(miss)} rows ({len(miss) / len(df):.2%}) are '{NULL}'"})

    # ---- domain -----------------------------------------------------------------------------
    for c in CATEGORICAL + [TARGET]:
        allowed = set(DICTIONARY[c]) | ({NULL} if c != TARGET else set())
        bad = df.index[~df[c].isin(allowed)]
        unknown = sorted(set(df.loc[bad, c]))[:10]
        record("domain", f"{c} values in dictionary", c, "error", bad, "UCI adult.names categories"
               + (" + '?'" if c != TARGET else ""), f"unknown values: {unknown}" if unknown else
               f"{df[c].nunique()} distinct values")

    # ---- duplicates -------------------------------------------------------------------------
    dup_all = df.index[df.duplicated(keep="first")]
    record("duplicates", "exact duplicate rows (all 15 columns)", "all", "warning", dup_all,
           "rows identical to an earlier row", "kept here; removed from the training split only (step 4)")
    dup_feat = df.index[df.duplicated(subset=FEATURES, keep="first")]
    record("duplicates", "duplicate feature rows (14 features)", "features", "info", dup_feat,
           "same 14 feature values as an earlier row", "label conflicts are audited in step 6")

    # ---- consistency (data dictionary) -------------------------------------------------------
    exp_num = df["education"].map(EDUCATION_NUM)
    bad = df.index[exp_num.notna() & (exp_num != pd.to_numeric(df["education-num"], errors="coerce"))]
    record("consistency", "education matches education-num", "education-num", "warning", bad,
           "Preschool=1 ... Doctorate=16")
    bad = df.index[((df["relationship"] == "Husband") & (df["sex"] != "Male")) |
                   ((df["relationship"] == "Wife") & (df["sex"] != "Female"))]
    record("consistency", "Husband is Male / Wife is Female", "relationship", "warning", bad,
           "relationship and sex agree")

    n_after = len(df)
    assert n_after == n_before, "validation must not drop rows"
    errors = sum(c["violations"] for c in checks if c["severity"] == "error")
    warnings_ = sum(c["violations"] for c in checks if c["severity"] == "warning")
    report = {"rows_before": n_before, "rows_after": n_after, "rows_dropped": 0,
              "columns": len(df.columns), "n_checks": len(checks),
              "checks_passed": sum(c["status"] == "PASS" for c in checks),
              "error_violations": errors, "warning_violations": warnings_,
              "missing_per_column": null_counts,
              "rows_with_any_missing": int((df[CATEGORICAL] == NULL).any(axis=1).sum()),
              "label_counts": df[TARGET].value_counts().to_dict(), "checks": checks}
    write_json(RESULTS_DIR / "validation_report.json", report)
    pd.DataFrame(violations, columns=["row", "check", "column", "severity", "value"]).to_csv(
        RESULTS_DIR / "validation_violations.csv", index=False, lineterminator="\n")

    md = ["# Validation report (step 3)\n",
          f"Rows checked: **{n_before:,}** - rows dropped: **0** (violations are reported, never silently "
          f"removed). Checks: {len(checks)}, passed: {report['checks_passed']}. Schema errors: "
          f"**{errors}**, warnings: **{warnings_}**. Every flagged row is listed in "
          "`validation_violations.csv`.\n",
          "| Category | Check | Severity | Rule | Violations | Status | Note |",
          "|---|---|---|---|---:|---|---|"]
    for c in checks:
        md.append(f"| {c['category']} | {c['check']} | {c['severity']} | {c['rule']} | {c['violations']:,} | "
                  f"{c['status']} | {c['note']} |")
    (RESULTS_DIR / "validation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Validation: {len(checks)} checks, {report['checks_passed']} passed, {errors} schema errors, "
          f"{warnings_} warnings; rows {n_before} -> {n_after} (none dropped)")
    for c in checks:
        if c["status"] != "PASS":
            print(f"  [{c['status']}] {c['check']}: {c['violations']} ({c['note']})")


if __name__ == "__main__":
    main()
