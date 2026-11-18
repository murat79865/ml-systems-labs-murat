# Validation report (step 3)

Rows checked: **48,842** - rows dropped: **0** (violations are reported, never silently removed). Checks: 48, passed: 45. Schema errors: **0**, warnings: **56**. Every flagged row is listed in `validation_violations.csv`.

| Category | Check | Severity | Rule | Violations | Status | Note |
|---|---|---|---|---:|---|---|
| columns | schema columns present | error | 15 columns: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income | 0 | PASS | missing [], unexpected [] |
| types | age is int | error | numeric, whole numbers | 0 | PASS | raw dtype int64 |
| types | fnlwgt is int | error | numeric, whole numbers | 0 | PASS | raw dtype int64 |
| types | education-num is int | error | numeric, whole numbers | 0 | PASS | raw dtype int64 |
| types | capital-gain is int | error | numeric, whole numbers | 0 | PASS | raw dtype int64 |
| types | capital-loss is int | error | numeric, whole numbers | 0 | PASS | raw dtype int64 |
| types | hours-per-week is int | error | numeric, whole numbers | 0 | PASS | raw dtype int64 |
| types | workclass is category | error | text labels | 0 | PASS | raw dtype object |
| types | education is category | error | text labels | 0 | PASS | raw dtype object |
| types | marital-status is category | error | text labels | 0 | PASS | raw dtype object |
| types | occupation is category | error | text labels | 0 | PASS | raw dtype object |
| types | relationship is category | error | text labels | 0 | PASS | raw dtype object |
| types | race is category | error | text labels | 0 | PASS | raw dtype object |
| types | sex is category | error | text labels | 0 | PASS | raw dtype object |
| types | native-country is category | error | text labels | 0 | PASS | raw dtype object |
| types | income is category | error | text labels | 0 | PASS | raw dtype object |
| ranges | age in [17, 90] | error | 17 <= age <= 90 | 0 | PASS | observed min 17, max 90 |
| ranges | hours-per-week in [1, 99] | error | 1 <= hours-per-week <= 99 | 0 | PASS | observed min 1, max 99 |
| ranges | capital-gain in [0, 99999] | error | 0 <= capital-gain <= 99999 | 0 | PASS | observed min 0, max 99999 |
| ranges | capital-loss in [0, 4356] | error | 0 <= capital-loss <= 4356 | 0 | PASS | observed min 0, max 4356 |
| nulls | age has no missing | error | numeric columns may not be missing | 0 | PASS |  |
| nulls | workclass missing ('?') | info | '?' allowed in category columns | 0 | PASS | 2799 rows (5.73%) are '?' |
| nulls | fnlwgt has no missing | error | numeric columns may not be missing | 0 | PASS |  |
| nulls | education missing ('?') | info | '?' allowed in category columns | 0 | PASS | 0 rows (0.00%) are '?' |
| nulls | education-num has no missing | error | numeric columns may not be missing | 0 | PASS |  |
| nulls | marital-status missing ('?') | info | '?' allowed in category columns | 0 | PASS | 0 rows (0.00%) are '?' |
| nulls | occupation missing ('?') | info | '?' allowed in category columns | 0 | PASS | 2809 rows (5.75%) are '?' |
| nulls | relationship missing ('?') | info | '?' allowed in category columns | 0 | PASS | 0 rows (0.00%) are '?' |
| nulls | race missing ('?') | info | '?' allowed in category columns | 0 | PASS | 0 rows (0.00%) are '?' |
| nulls | sex missing ('?') | info | '?' allowed in category columns | 0 | PASS | 0 rows (0.00%) are '?' |
| nulls | capital-gain has no missing | error | numeric columns may not be missing | 0 | PASS |  |
| nulls | capital-loss has no missing | error | numeric columns may not be missing | 0 | PASS |  |
| nulls | hours-per-week has no missing | error | numeric columns may not be missing | 0 | PASS |  |
| nulls | native-country missing ('?') | info | '?' allowed in category columns | 0 | PASS | 857 rows (1.75%) are '?' |
| nulls | income missing ('?') | error | label may not be missing | 0 | PASS | 0 rows (0.00%) are '?' |
| domain | workclass values in dictionary | error | UCI adult.names categories + '?' | 0 | PASS | 9 distinct values |
| domain | education values in dictionary | error | UCI adult.names categories + '?' | 0 | PASS | 16 distinct values |
| domain | marital-status values in dictionary | error | UCI adult.names categories + '?' | 0 | PASS | 7 distinct values |
| domain | occupation values in dictionary | error | UCI adult.names categories + '?' | 0 | PASS | 15 distinct values |
| domain | relationship values in dictionary | error | UCI adult.names categories + '?' | 0 | PASS | 6 distinct values |
| domain | race values in dictionary | error | UCI adult.names categories + '?' | 0 | PASS | 5 distinct values |
| domain | sex values in dictionary | error | UCI adult.names categories + '?' | 0 | PASS | 2 distinct values |
| domain | native-country values in dictionary | error | UCI adult.names categories + '?' | 0 | PASS | 42 distinct values |
| domain | income values in dictionary | error | UCI adult.names categories | 0 | PASS | 2 distinct values |
| duplicates | exact duplicate rows (all 15 columns) | warning | rows identical to an earlier row | 52 | WARNING | kept here; removed from the training split only (step 4) |
| duplicates | duplicate feature rows (14 features) | info | same 14 feature values as an earlier row | 57 | INFO | label conflicts are audited in step 6 |
| consistency | education matches education-num | warning | Preschool=1 ... Doctorate=16 | 0 | PASS |  |
| consistency | Husband is Male / Wife is Female | warning | relationship and sex agree | 4 | WARNING |  |
