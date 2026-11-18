# Lab 4 - Data pipeline and quality

**Course:** Machine Learning (CYI5SE039CY), Group 11  
**Reading:** Reddi, Ch. 5 AI Workflow §§5.2-5.5 and Ch. 6 Data Engineering §§6.2-6.8

**Goal.** Build a reproducible, leakage-free pipeline for the UCI Adult census data (mixed numeric and categorical features): ingest without modification, validate against an explicit schema, clean with statistics learned on the training split only, engineer and test two derived features, audit label quality, and fingerprint the processed data with SHA256 next to its configuration. The whole pipeline runs with one command: `python lab04/src/run_all.py`.

## 1. Problem definition (step 1)

![Handwritten problem definition](results/problem_definition_paper.jpg)

*Figure 1 - Handwritten problem definition.*

- **Task:** binary classification - predict whether a person's annual income is **>50K USD** from census attributes. Positive class: `>50K` (23.9% of all rows).
- **Inputs:** the 14 features age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country (6 numeric, 8 categorical).
- **Output:** `income` in {>50K, <=50K}.
- **Success metric:** F1 of the >50K class (the classes are imbalanced, so accuracy alone would reward always predicting <=50K, which already scores 76.1%) together with accuracy, both on a held-out 30% test split.
- **Constraints:** no leakage - the test split never influences any fitted statistic (imputation values, clip bounds, scaler, encoder, duplicate removal); a reproducible pipeline - fixed seed 42, all parameters in `config.yaml`, raw data never modified, processed data identified by SHA256, one command to rebuild everything.

## 2. Pipeline

Environment: Python 3.11.8 in the pinned `.venv` (all 25 pins `OK`, `results/versions.txt`).

| Stage | Script | What it does | Evidence |
|---|---|---|---|
| Ingest | `ingest.py` | `fetch_openml(name='adult', version=2, as_frame=True)`; writes the frame unchanged to `data/lab04/raw/adult_raw.csv` | `ingestion_log.json` |
| Validate | `validate.py` | schema types, ranges, null policy, dictionary, duplicates, consistency; reports, never drops | `validation_report.md/.json`, `validation_violations.csv` |
| Split | `clean_transform.py` | 70/30 stratified on the raw rows, before any cleaning | `split_indices.csv` |
| Clean | `clean_transform.py` | drop exact duplicates from train only; fit median/mode imputer and 1st/99th percentile clipping on train; derived features; scaling + one-hot | `preprocessor.joblib`, `cleaning_log.json`, `leakage_checks.json` |
| Features | `features_eval.py` | LogisticRegression with vs. without derived features | `feature_eval.csv/.json` |
| Label quality | `label_audit.py` | 50-row re-check + duplicate-conflict audit | `label_audit_sample.md/.csv`, `label_conflicts.csv`, `label_audit.json` |
| Fingerprint | `clean_transform.py` | SHA256 of the processed train/test tables, stored with the config | `config_and_hash.yaml` |
| Document | `make_report.py` | this report and the data card | `report.md`, `data_card.md` |

## 3. Results

### 3.1 Ingestion (step 2)

Logged shape: **48,842 rows x 15 columns** (expected 48,842 x 15: match). OpenML dataset id 1590. The frame is written as returned - no renaming, recoding or filtering; raw snapshot SHA256 `8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc`. Missing values in the raw frame: workclass 2,799, occupation 2,809, native-country 857.

### 3.2 Validation report (step 3)

48 checks on all 48,842 rows; **45 passed**, schema errors: **0**, warnings: **56**. Rows before / after validation: 48,842 / 48,842 - **no row dropped**. Full table: `results/validation_report.md`; every flagged row: `results/validation_violations.csv`.

| Category | Check | Severity | Violations | Status | Note |
|---|---|---|---:|---|---|
| columns | schema columns present | error | 0 | PASS | missing [], unexpected [] |
| ranges | age in [17, 90] | error | 0 | PASS | observed min 17, max 90 |
| ranges | hours-per-week in [1, 99] | error | 0 | PASS | observed min 1, max 99 |
| ranges | capital-gain in [0, 99999] | error | 0 | PASS | observed min 0, max 99999 |
| ranges | capital-loss in [0, 4356] | error | 0 | PASS | observed min 0, max 4356 |
| nulls | workclass missing ('?') | info | 0 | PASS | 2799 rows (5.73%) are '?' |
| nulls | occupation missing ('?') | info | 0 | PASS | 2809 rows (5.75%) are '?' |
| nulls | native-country missing ('?') | info | 0 | PASS | 857 rows (1.75%) are '?' |
| duplicates | exact duplicate rows (all 15 columns) | warning | 52 | WARNING | kept here; removed from the training split only (step 4) |
| duplicates | duplicate feature rows (14 features) | info | 57 | INFO | label conflicts are audited in step 6 |
| consistency | Husband is Male / Wife is Female | warning | 4 | WARNING |  |

Not repeated above (37 checks, all PASS): type checks (15 of 15 columns match the declared int / category type), every category value is in the UCI dictionary, the label is never missing, numeric columns and the remaining category columns have no missing values. Missing values ('?'): 3,620 rows (7.4%) have at least one; they are allowed by the null policy, reported here and imputed in step 4.

### 3.3 Leakage-free cleaning and transformation (step 4)

| Item | Train | Test |
|---|---:|---:|
| rows after 70/30 stratified split | 34,189 | 14,653 |
| exact duplicates dropped | 24 | 0 (kept: 5) |
| rows used | 34,165 | 14,653 |
| >50K rate | 23.9% | 23.9% |
| '?' imputed in workclass (mode `Private`) | 1,923 | 873 |
| '?' imputed in occupation (mode `Prof-specialty`) | 1,933 | 873 |
| '?' imputed in native-country (mode `United-States`) | 590 | 267 |

Numeric features - median (imputation value) and clip bounds, all learned from the training split:

| Feature | Median | 1st pct (lower) | 99th pct (upper) | Clipped train (low / high) | Clipped test (low / high) |
|---|---:|---:|---:|---:|---:|
| age | 37 | 17 | 74 | 0 / 302 | 0 / 143 |
| fnlwgt | 178272 | 27035.2 | 509629 | 342 / 341 | 127 / 145 |
| education-num | 10 | 3 | 16 | 240 / 0 | 88 / 0 |
| capital-gain | 0 | 0 | 15024 | 0 / 255 | 0 / 136 |
| capital-loss | 0 | 0 | 1980 | 0 / 340 | 0 / 151 |
| hours-per-week | 40 | 8 | 80 | 311 / 226 | 144 / 92 |

The fitted preprocessor (imputer -> clipper -> derived features -> StandardScaler + OneHotEncoder, 107 model inputs) is saved with `joblib.dump` to `results/preprocessor.joblib`.

**Leakage checks** (`results/leakage_checks.json`):

| Check | Result |
|---|:---:|
| split happens before cleaning | PASS |
| train and test rows are disjoint | PASS |
| test split untouched (no rows dropped) | PASS |
| duplicates removed from train only | PASS |
| preprocessor fitted on train rows only | PASS |
| numeric medians = train medians | PASS |
| categorical modes = train modes | PASS |
| clip bounds = train percentiles | PASS |

As a contrast, fitting the clip bounds on train + test would change them for fnlwgt, capital-loss - exactly the kind of silent test-set influence the train-only fit prevents.

### 3.4 Derived features (step 5)

`capital_net = capital-gain - capital-loss`; `hours_per_age = hours-per-week / (age - 16)` (computed after imputation and clipping, so age - 16 >= 1).

| Variant | Model inputs | Test accuracy | **Test F1** | 5-fold CV F1 (train) | LR iterations |
|---|---:|---:|---:|---:|---:|
| without derived features | 105 | 0.8540 | **0.6601** | 0.6552 +/- 0.0118 | 61 |
| with derived features | 107 | 0.8578 | **0.6696** | 0.6618 +/- 0.0129 | 75 |

Adding the two features changes test accuracy by +0.0038 and test F1 by +0.0095; the cross-validated F1 changes by +0.0066 (fold standard deviation about 0.0129). The derived features **improve** the baseline, but the change is within the fold-to-fold noise. The two features behave differently in a linear model. `capital_net` is an exact linear combination of two inputs the model already has (clipped capital-gain minus clipped capital-loss), so it adds no new information - its weight only redistributes the existing capital weights (standardised coefficient +0.410). `hours_per_age` is a ratio, i.e. a non-linear interaction between hours and age that a linear model cannot build from the separate columns; it receives the largest weight of the two (-1.783), so any gain comes from it.

### 3.5 Label quality (step 6)

**Duplicate-conflict audit.** 54 groups of rows share identical values for all 14 features. Of these, **5 groups (10 rows)** carry *different* income labels in the full dataset (2 groups / 4 rows inside the training split). Such rows cannot all be predicted correctly by any model; they set a small floor on the error and are listed in `results/label_conflicts.csv`. They were kept, because the census does not tell which label is wrong (two different people can have identical recorded attributes).

**50-row re-check.** 50 rows drawn at random (seed 42) were checked one by one against the data dictionary - allowed categories, schema ranges, education vs education-num, Husband/Male and Wife/Female, valid label (`results/label_audit_sample.md`). Verdicts: valid: 47, valid (has '?'): 3. Across the full dataset the same rules find 0 education/education-num mismatches and 4 relationship/sex contradictions. Adult's labels come from census records, so the subjective-label variant (blind relabelling + Cohen's kappa) does not apply.

### 3.6 Processed-data hash and configuration (step 7)

| Artefact | Rows | SHA256 |
|---|---:|---|
| raw snapshot `data/lab04/raw/adult_raw.csv` | 48,842 | `8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc` |
| processed train `data/lab04/processed/train.csv` | 34,165 | `82b9bcb4dde83e362287ddbfca370b5631c5ba073d95a993cf838e26a6faf569` |
| processed test `data/lab04/processed/test.csv` | 14,653 | `fa008efbb6448ba167448d7003784fe4fd20f990b765fc8ab7fe2b87fdb1ab9b` |
| processed train + test (combined) | 48,818 | `72cfbc9a88489ff2d54e03ebe4bf59a601be690cfb3a8e3e73dfa32eb082bce6` |
| `results/preprocessor.joblib` | - | `168c3a36aa30806128d59b08f792a60cfcc5d9a9c72f1fffe19633ab68fbdbc7` |

The hashes are stored together with the complete configuration in `results/config_and_hash.yaml`; the input configuration is `lab04/config.yaml`. The processed tables live in the git-ignored `data/` folder and are rebuilt by `python lab04/src/run_all.py`; re-running on the same environment reproduces the same hashes. The data card is `lab04/data_card.md`.

## 4. Conclusions

1. **Validation found no schema errors but real quality issues.** All values respect the declared types and ranges, yet 3,620 rows contain '?', the data holds exact duplicates and 5 conflicting-label groups. Reporting instead of dropping keeps these decisions visible and lets each later step handle them explicitly.

2. **Leakage is prevented by the order of operations.** Splitting first and fitting every statistic on the de-duplicated training split (34,165 rows) is verified by 8 automatic checks; the test split (14,653 rows) is only ever transformed.

3. **Derived features must be tested, not assumed.** capital_net and hours_per_age changed test F1 by +0.0095 and CV F1 by +0.0066 - within noise for a linear baseline. Only the ratio hours_per_age carries information a linear model cannot form itself; capital_net is a linear combination of existing inputs. Together with the SHA256 + config manifest, the one-command pipeline makes such comparisons repeatable.

## 5. Reproducibility

With the virtual environment active, run `python lab04/src/run_all.py` from the repository root. It downloads Adult into `data/openml/` on first use (git-ignored), regenerates every file in `lab04/results/`, `report.md` and `data_card.md`; the photo of the handwritten page is kept.
