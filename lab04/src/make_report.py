"""Build lab04/report.md and lab04/data_card.md from the files in results/ (and the raw snapshot).

Every number is read from the run's outputs. A photo of the handwritten problem definition is embedded
if results/problem_definition_paper.(jpg|jpeg|png) exists.
"""

from __future__ import annotations

import csv
import re

import yaml

from common import (CATEGORICAL, CFG, FEATURES, LAB_DIR, NULL, NUMERIC, RESULTS_DIR, TARGET, canonical,
                    read_json, read_raw, to_binary)

DESCRIPTIONS = {
    "age": "age in years", "workclass": "employer type", "fnlwgt": "census sampling weight",
    "education": "highest education level", "education-num": "education level as a number (1-16)",
    "marital-status": "marital status", "occupation": "occupation group",
    "relationship": "role in the household", "race": "race (census categories)", "sex": "sex",
    "capital-gain": "capital gains (USD)", "capital-loss": "capital losses (USD)",
    "hours-per-week": "usual working hours per week", "native-country": "country of origin",
    "income": "label: annual income >50K or <=50K (USD)",
}


def rows(name):
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def photo():
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        if (RESULTS_DIR / f"problem_definition_paper.{ext}").exists():
            return f"problem_definition_paper.{ext}"
    return None


def versions():
    text = (RESULTS_DIR / "versions.txt").read_text(encoding="utf-8")
    py = re.search(r"Python\s*:\s*(\S+)", text)
    pr = re.search(r"problems:\s*(\d+)", text)
    return (py.group(1) if py else "?"), (int(pr.group(1)) if pr else -1)


def pct(x):
    return f"{100 * x:.1f}%"


def main() -> None:
    ing = read_json(RESULTS_DIR / "ingestion_log.json")
    val = read_json(RESULTS_DIR / "validation_report.json")
    cl = read_json(RESULTS_DIR / "cleaning_log.json")
    leak = read_json(RESULTS_DIR / "leakage_checks.json")
    fe = rows("feature_eval.csv")
    fej = read_json(RESULTS_DIR / "feature_eval.json")
    la = read_json(RESULTS_DIR / "label_audit.json")
    with open(RESULTS_DIR / "config_and_hash.yaml", encoding="utf-8") as f:
        man = yaml.safe_load(f)
    df = canonical(read_raw())
    py, problems = versions()
    pic = photo()
    fitted = cl["fitted"]
    pos = CFG["dataset"]["positive_label"]

    L = []
    add = L.append
    add("# Lab 4 - Data pipeline and quality\n")
    add("**Course:** Machine Learning (CYI5SE039CY), Group 11  ")
    add("**Reading:** Reddi, Ch. 5 AI Workflow §§5.2-5.5 and Ch. 6 Data Engineering §§6.2-6.8\n")
    add("**Goal.** Build a reproducible, leakage-free pipeline for the UCI Adult census data (mixed numeric "
        "and categorical features): ingest without modification, validate against an explicit schema, clean "
        "with statistics learned on the training split only, engineer and test two derived features, audit "
        "label quality, and fingerprint the processed data with SHA256 next to its configuration. "
        "The whole pipeline runs with one command: `python lab04/src/run_all.py`.\n")

    # --- 1 -------------------------------------------------------------------------------------
    add("## 1. Problem definition (step 1)\n")
    if pic:
        add(f"![Handwritten problem definition](results/{pic})\n")
        add("*Figure 1 - Handwritten problem definition.*\n")
    pos_rate = float(to_binary(df[TARGET]).mean())
    add(f"- **Task:** binary classification - predict whether a person's annual income is **>50K USD** from "
        f"census attributes. Positive class: `{pos}` ({pct(pos_rate)} of all rows).")
    add("- **Inputs:** the 14 features age, workclass, fnlwgt, education, education-num, marital-status, "
        "occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country "
        f"({len(NUMERIC)} numeric, {len(CATEGORICAL)} categorical).")
    add("- **Output:** `income` in {>50K, <=50K}.")
    add(f"- **Success metric:** F1 of the >50K class (the classes are imbalanced, so accuracy alone would "
        f"reward always predicting <=50K, which already scores {pct(1 - pos_rate)}) together with accuracy, "
        "both on a held-out 30% test split.")
    add("- **Constraints:** no leakage - the test split never influences any fitted statistic (imputation "
        "values, clip bounds, scaler, encoder, duplicate removal); a reproducible pipeline - fixed seed 42, "
        "all parameters in `config.yaml`, raw data never modified, processed data identified by SHA256, "
        "one command to rebuild everything.\n")

    # --- 2 -------------------------------------------------------------------------------------
    add("## 2. Pipeline\n")
    env_ok = py == "3.11.8" and problems == 0
    add("Environment: Python 3.11.8 in the pinned `.venv` "
        + ("(all 25 pins `OK`, `results/versions.txt`)." if env_ok else
           f"(**warning:** versions.txt reports Python {py}, {problems} mismatch(es)).") + "\n")
    add("| Stage | Script | What it does | Evidence |")
    add("|---|---|---|---|")
    add("| Ingest | `ingest.py` | `fetch_openml(name='adult', version=2, as_frame=True)`; writes the frame "
        "unchanged to `data/lab04/raw/adult_raw.csv` | `ingestion_log.json` |")
    add("| Validate | `validate.py` | schema types, ranges, null policy, dictionary, duplicates, consistency; "
        "reports, never drops | `validation_report.md/.json`, `validation_violations.csv` |")
    add("| Split | `clean_transform.py` | 70/30 stratified on the raw rows, before any cleaning | "
        "`split_indices.csv` |")
    add("| Clean | `clean_transform.py` | drop exact duplicates from train only; fit median/mode imputer and "
        "1st/99th percentile clipping on train; derived features; scaling + one-hot | "
        "`preprocessor.joblib`, `cleaning_log.json`, `leakage_checks.json` |")
    add("| Features | `features_eval.py` | LogisticRegression with vs. without derived features | "
        "`feature_eval.csv/.json` |")
    add("| Label quality | `label_audit.py` | 50-row re-check + duplicate-conflict audit | "
        "`label_audit_sample.md/.csv`, `label_conflicts.csv`, `label_audit.json` |")
    add("| Fingerprint | `clean_transform.py` | SHA256 of the processed train/test tables, stored with the "
        "config | `config_and_hash.yaml` |")
    add("| Document | `make_report.py` | this report and the data card | `report.md`, `data_card.md` |")
    add("")

    # --- 3 -------------------------------------------------------------------------------------
    add("## 3. Results\n")
    add("### 3.1 Ingestion (step 2)\n")
    add(f"Logged shape: **{ing['rows']:,} rows x {ing['columns']} columns** (expected "
        f"{ing['expected_rows']:,} x {ing['expected_columns']}: "
        f"{'match' if ing['shape_matches_expected'] else 'MISMATCH'}). OpenML dataset id "
        f"{ing.get('openml_id')}. The frame is written as returned - no renaming, recoding or filtering; raw "
        f"snapshot SHA256 `{ing['raw_snapshot_sha256']}`. Missing values in the raw frame: "
        + ", ".join(f"{c} {n:,}" for c, n in ing["missing_values_per_column"].items() if n) + ".\n")

    add("### 3.2 Validation report (step 3)\n")
    add(f"{val['n_checks']} checks on all {val['rows_before']:,} rows; **{val['checks_passed']} passed**, "
        f"schema errors: **{val['error_violations']}**, warnings: **{val['warning_violations']}**. Rows "
        f"before / after validation: {val['rows_before']:,} / {val['rows_after']:,} - **no row dropped**. "
        "Full table: `results/validation_report.md`; every flagged row: `results/validation_violations.csv`.\n")
    add("| Category | Check | Severity | Violations | Status | Note |")
    add("|---|---|---|---:|---|---|")
    hidden = 0
    for c in val["checks"]:
        shown = (c["status"] != "PASS" or c["category"] in ("columns", "ranges")
                 or (c["category"] == "nulls" and c["note"] and not c["note"].startswith("0 rows")))
        if not shown:
            hidden += 1
            continue
        add(f"| {c['category']} | {c['check']} | {c['severity']} | {c['violations']:,} | {c['status']} | {c['note']} |")
    n_types = sum(1 for c in val["checks"] if c["category"] == "types")
    n_types_ok = sum(1 for c in val["checks"] if c["category"] == "types" and c["status"] == "PASS")
    add("")
    add(f"Not repeated above ({hidden} checks, all PASS): type checks ({n_types_ok} of {n_types} columns match "
        "the declared int / category type), every category value is in the UCI dictionary, the label is never "
        "missing, numeric columns and the remaining category columns have no missing values. "
        f"Missing values ('{NULL}'): {val['rows_with_any_missing']:,} rows "
        f"({pct(val['rows_with_any_missing'] / val['rows_before'])}) have at least one; they are allowed by the "
        "null policy, reported here and imputed in step 4.\n")

    add("### 3.3 Leakage-free cleaning and transformation (step 4)\n")
    add(f"| Item | Train | Test |")
    add("|---|---:|---:|")
    add(f"| rows after 70/30 stratified split | {cl['train_rows_before_dedup']:,} | {cl['test_rows']:,} |")
    add(f"| exact duplicates dropped | {cl['duplicates_dropped_from_train']:,} | 0 (kept: "
        f"{cl['test_exact_duplicates_kept']}) |")
    add(f"| rows used | {cl['train_rows']:,} | {cl['test_rows']:,} |")
    add(f"| >50K rate | {pct(cl['train']['label_positive_rate'])} | {pct(cl['test']['label_positive_rate'])} |")
    for c in CATEGORICAL:
        if cl["train"]["imputed_categorical"][c] or cl["test"]["imputed_categorical"][c]:
            add(f"| '{NULL}' imputed in {c} (mode `{fitted['categorical_mode'][c]}`) | "
                f"{cl['train']['imputed_categorical'][c]:,} | {cl['test']['imputed_categorical'][c]:,} |")
    add("")
    add("Numeric features - median (imputation value) and clip bounds, all learned from the training split:\n")
    add("| Feature | Median | 1st pct (lower) | 99th pct (upper) | Clipped train (low / high) | Clipped test (low / high) |")
    add("|---|---:|---:|---:|---:|---:|")
    for c in NUMERIC:
        add(f"| {c} | {fitted['numeric_median'][c]:g} | {fitted['clip_lower'][c]:g} | {fitted['clip_upper'][c]:g} | "
            f"{cl['train']['clipped_low'][c]:,} / {cl['train']['clipped_high'][c]:,} | "
            f"{cl['test']['clipped_low'][c]:,} / {cl['test']['clipped_high'][c]:,} |")
    add("")
    add(f"The fitted preprocessor (imputer -> clipper -> derived features -> StandardScaler + OneHotEncoder, "
        f"{cl['encoded_feature_count']} model inputs) is saved with `joblib.dump` to "
        "`results/preprocessor.joblib`.\n")
    add("**Leakage checks** (`results/leakage_checks.json`):\n")
    add("| Check | Result |")
    add("|---|:---:|")
    for c in leak["checks"]:
        if c["check"].startswith("contrast"):
            continue
        add(f"| {c['check']} | {'PASS' if c['pass'] else 'FAIL'} |")
    contrast = next((c for c in leak["checks"] if c["check"].startswith("contrast")), None)
    add("")
    if contrast and contrast["detail"]["columns_whose_bounds_would_change"]:
        add("As a contrast, fitting the clip bounds on train + test would change them for "
            + ", ".join(contrast["detail"]["columns_whose_bounds_would_change"])
            + " - exactly the kind of silent test-set influence the train-only fit prevents.\n")

    add("### 3.4 Derived features (step 5)\n")
    add("`capital_net = capital-gain - capital-loss`; `hours_per_age = hours-per-week / (age - 16)` "
        "(computed after imputation and clipping, so age - 16 >= 1).\n")
    add("| Variant | Model inputs | Test accuracy | **Test F1** | 5-fold CV F1 (train) | LR iterations |")
    add("|---|---:|---:|---:|---:|---:|")
    for r in fe:
        add(f"| {r['variant']} | {r['encoded_columns']} | {r['test_accuracy']} | **{r['test_f1']}** | "
            f"{r['cv_f1_mean']} +/- {r['cv_f1_std']} | {r['lr_iterations']} |")
    add("")
    d_acc, d_f1, d_cv = fej["delta_test_accuracy"], fej["delta_test_f1"], fej["delta_cv_f1"]
    std = max(float(fe[0]["cv_f1_std"]), float(fe[1]["cv_f1_std"]))
    coefs = fej.get("derived_coefficients_standardised", {})
    effect = ("improve" if d_f1 > 0 and d_cv > 0 else "hurt" if d_f1 < 0 and d_cv < 0 else "do not consistently change")
    hpa = coefs.get("hours_per_age", 0.0)
    cnet = coefs.get("capital_net", 0.0)
    para = (f"Adding the two features changes test accuracy by {d_acc:+.4f} and test F1 by {d_f1:+.4f}; the "
            f"cross-validated F1 changes by {d_cv:+.4f} (fold standard deviation about {std:.4f}). "
            f"The derived features **{effect}** the baseline"
            + (", but the change is within the fold-to-fold noise. " if abs(d_cv) < std else ". "))
    para += ("The two features behave differently in a linear model. `capital_net` is an exact linear "
             "combination of two inputs the model already has (clipped capital-gain minus clipped "
             "capital-loss), so it adds no new information - its weight only redistributes the existing "
             f"capital weights (standardised coefficient {cnet:+.3f}). `hours_per_age` is a ratio, i.e. a "
             "non-linear interaction between hours and age that a linear model cannot build from the separate "
             f"columns; it receives the largest weight of the two ({hpa:+.3f})")
    para += (", so any gain comes from it." if d_f1 > 0 else ".")
    add(para + "\n")

    add("### 3.5 Label quality (step 6)\n")
    add(f"**Duplicate-conflict audit.** {la['duplicate_feature_groups']:,} groups of rows share identical "
        f"values for all 14 features. Of these, **{la['conflict_groups_full_data']} groups "
        f"({la['conflict_rows_full_data']} rows)** carry *different* income labels in the full dataset "
        f"({la['conflict_groups_train']} groups / {la['conflict_rows_train']} rows inside the training split). "
        "Such rows cannot all be predicted correctly by any model; they set a small floor on the error and are "
        "listed in `results/label_conflicts.csv`. They were kept, because the census does not tell which "
        "label is wrong (two different people can have identical recorded attributes).\n")
    sv = la["sample_verdicts"]
    add(f"**50-row re-check.** 50 rows drawn at random (seed 42) were checked one by one against the data "
        "dictionary - allowed categories, schema ranges, education vs education-num, Husband/Male and "
        "Wife/Female, valid label (`results/label_audit_sample.md`). Verdicts: "
        + ", ".join(f"{k}: {v}" for k, v in sv.items())
        + f". Across the full dataset the same rules find {la['full_data_rule_failures']['education_vs_num']} "
          f"education/education-num mismatches and {la['full_data_rule_failures']['relationship_vs_sex']} "
          "relationship/sex contradictions. Adult's labels come from census records, so the subjective-label "
          "variant (blind relabelling + Cohen's kappa) does not apply.\n")

    add("### 3.6 Processed-data hash and configuration (step 7)\n")
    p = man["processed"]
    add("| Artefact | Rows | SHA256 |")
    add("|---|---:|---|")
    add(f"| raw snapshot `data/lab04/raw/adult_raw.csv` | {man['raw_snapshot']['rows']:,} | `{man['raw_snapshot']['sha256']}` |")
    add(f"| processed train `data/lab04/processed/train.csv` | {p['train']['rows']:,} | `{p['train']['sha256']}` |")
    add(f"| processed test `data/lab04/processed/test.csv` | {p['test']['rows']:,} | `{p['test']['sha256']}` |")
    add(f"| processed train + test (combined) | {p['train']['rows'] + p['test']['rows']:,} | `{p['sha256_train_plus_test']}` |")
    add(f"| `results/preprocessor.joblib` | - | `{man['preprocessor']['sha256']}` |")
    add("")
    add("The hashes are stored together with the complete configuration in `results/config_and_hash.yaml`; "
        "the input configuration is `lab04/config.yaml`. The processed tables live in the git-ignored `data/` "
        "folder and are rebuilt by `python lab04/src/run_all.py`; re-running on the same environment "
        "reproduces the same hashes. The data card is `lab04/data_card.md`.\n")

    # --- 4 -------------------------------------------------------------------------------------
    add("## 4. Conclusions\n")
    add((f"1. **Validation found no schema errors but real quality issues.** All values respect the declared "
         f"types and ranges, yet " if val["error_violations"] == 0 else
         f"1. **Validation found {val['error_violations']} schema violations and further quality issues.** "
         "They are listed row by row in `validation_violations.csv`; in addition ")
        + f"{val['rows_with_any_missing']:,} rows contain '?', the data holds exact "
        f"duplicates and {la['conflict_groups_full_data']} conflicting-label groups. Reporting instead of dropping "
        "keeps these decisions visible and lets each later step handle them explicitly.\n")
    add(f"2. **Leakage is prevented by the order of operations.** Splitting first and fitting every statistic "
        f"on the de-duplicated training split ({cl['train_rows']:,} rows) is verified by "
        f"{sum(1 for c in leak['checks'] if not c['check'].startswith('contrast'))} automatic checks; the test "
        f"split ({cl['test_rows']:,} rows) is only ever transformed.\n")
    add(f"3. **Derived features must be tested, not assumed.** capital_net and hours_per_age changed test F1 by "
        f"{d_f1:+.4f} and CV F1 by {d_cv:+.4f}" + (" - within noise" if abs(d_cv) < std else "")
        + " for a linear baseline. Only the ratio hours_per_age carries information a linear model cannot "
          "form itself; capital_net is a linear combination of existing inputs. Together with the SHA256 + "
          "config manifest, the one-command pipeline makes such comparisons repeatable.\n")
    add("## 5. Reproducibility\n")
    add("With the virtual environment active, run `python lab04/src/run_all.py` from the repository root. It "
        "downloads Adult into `data/openml/` on first use (git-ignored), regenerates every file in "
        "`lab04/results/`, `report.md` and `data_card.md`; the photo of the handwritten page is kept.\n")
    (LAB_DIR / "report.md").write_text("\n".join(L), encoding="utf-8")

    # ============================== data card ==================================================
    D = []
    put = D.append
    sex = df["sex"].value_counts(normalize=True)
    race = df["race"].value_counts(normalize=True)
    us = float((df["native-country"] == "United-States").mean())
    y = to_binary(df[TARGET])
    rate = lambda col, v: float(y[(df[col] == v).to_numpy()].mean())
    put("# Data card - UCI Adult (Census Income), as used in Lab 4\n")
    put("## Source")
    put("- **Dataset:** Adult (\"Census Income\"), donated to the UCI Machine Learning Repository in 1996 by "
        "Barry Becker and Ronny Kohavi; extracted by Barry Becker from the **1994 US Census** database.")
    put(f"- **Accessed via:** OpenML, `fetch_openml(name='adult', version=2, as_frame=True)` (OpenML id "
        f"{ing.get('openml_id')}).")
    put("- **Citation:** Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5XW20\n")
    put("## License")
    put("- **Creative Commons Attribution 4.0 International (CC BY 4.0)** - reuse is allowed with attribution "
        "(cite as above).\n")
    put("## Size and splits")
    put(f"- Raw: **{ing['rows']:,} rows x {ing['columns']} columns** (14 features + label); raw snapshot SHA256 "
        f"`{ing['raw_snapshot_sha256']}`.")
    put(f"- Pipeline: 70/30 stratified split (seed 42) -> train {cl['train_rows_before_dedup']:,} rows, "
        f"{cl['duplicates_dropped_from_train']} exact duplicates removed -> **{cl['train_rows']:,} train**, "
        f"**{cl['test_rows']:,} test** (untouched).")
    put(f"- Processed tables (train + test) SHA256 `{p['sha256_train_plus_test']}`; config and per-file hashes "
        "in `results/config_and_hash.yaml`.\n")
    put("## Fields")
    put("| Column | Type | Description | Missing ('?') |")
    put("|---|---|---|---:|")
    for c in FEATURES + [TARGET]:
        miss = val["missing_per_column"].get(c, 0)
        put(f"| {c} | {CFG['schema']['types'][c]} | {DESCRIPTIONS[c]} | {miss:,} ({pct(miss / len(df))}) |")
    put("")
    put(f"Label balance: >50K {pct(y.mean())}, <=50K {pct(1 - y.mean())}.\n")
    put("## Known biases")
    put(f"- **Sampling and time:** a 1994 US census extract; incomes are not adjusted for inflation and the "
        "50K threshold reflects 1994 dollars, so the data does not describe today's population.")
    put(f"- **Group imbalance:** Male {pct(sex.get('Male', 0))} vs Female {pct(sex.get('Female', 0))}; "
        f"White {pct(race.get('White', 0))} of rows (Black {pct(race.get('Black', 0))}, Asian-Pac-Islander "
        f"{pct(race.get('Asian-Pac-Islander', 0))}, Amer-Indian-Eskimo {pct(race.get('Amer-Indian-Eskimo', 0))}, "
        f"Other {pct(race.get('Other', 0))}); United-States {pct(us)} of native-country.")
    put(f"- **Label disparity:** >50K rate is {pct(rate('sex', 'Male'))} for men vs {pct(rate('sex', 'Female'))} "
        f"for women and {pct(rate('race', 'White'))} for White vs {pct(rate('race', 'Black'))} for Black "
        "respondents. A model trained on this data will learn these historical gaps; group-wise metrics are "
        "needed before any use.")
    put(f"- **Missingness is not random:** '?' is concentrated in workclass, occupation and native-country "
        f"({val['rows_with_any_missing']:,} rows); mode imputation pushes these people into the majority category.")
    put("- **Proxy features:** relationship (Husband/Wife) and marital-status encode sex; fnlwgt is a sampling "
        "weight, not a personal attribute.")
    put(f"- **Label noise:** {la['conflict_groups_full_data']} groups of identical feature rows carry different "
        "labels (step 6).\n")
    put("## Privacy concerns")
    put("- Records are anonymised (no names or IDs), but **quasi-identifiers** - age, sex, race, native-country, "
        "occupation, marital status and education - can be combined to single out people, especially in small "
        "groups (rare countries or occupations).")
    put("- Several columns are **sensitive attributes** (race, sex, marital status, national origin); using them "
        "for decisions about individuals (credit, hiring, insurance) can be discriminatory and is regulated in "
        "many jurisdictions.")
    put("- Use for teaching and benchmarking; do not join with other data to re-identify individuals.\n")
    put("## Intended use and limitations")
    put("- Intended: teaching and benchmarking of tabular ML pipelines and fairness analysis.")
    put("- Not intended: real decisions about people's income, credit or employment.\n")
    put("## Processing applied in this lab")
    put("- '?' treated as missing; categorical values imputed with the training mode, numeric with the training "
        "median; numeric features clipped to the training 1st/99th percentiles; exact duplicates removed from "
        "the training split only; derived features capital_net and hours_per_age added. Details: `report.md`.")
    (LAB_DIR / "data_card.md").write_text("\n".join(D) + "\n", encoding="utf-8")
    print(f"Wrote {LAB_DIR / 'report.md'} and {LAB_DIR / 'data_card.md'}"
          + ("" if pic else "  (note: problem_definition_paper photo not found yet)"))


if __name__ == "__main__":
    main()
