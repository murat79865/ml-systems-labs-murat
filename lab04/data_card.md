# Data card - UCI Adult (Census Income), as used in Lab 4

## Source
- **Dataset:** Adult ("Census Income"), donated to the UCI Machine Learning Repository in 1996 by Barry Becker and Ronny Kohavi; extracted by Barry Becker from the **1994 US Census** database.
- **Accessed via:** OpenML, `fetch_openml(name='adult', version=2, as_frame=True)` (OpenML id 1590).
- **Citation:** Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

## License
- **Creative Commons Attribution 4.0 International (CC BY 4.0)** - reuse is allowed with attribution (cite as above).

## Size and splits
- Raw: **48,842 rows x 15 columns** (14 features + label); raw snapshot SHA256 `8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc`.
- Pipeline: 70/30 stratified split (seed 42) -> train 34,189 rows, 24 exact duplicates removed -> **34,165 train**, **14,653 test** (untouched).
- Processed tables (train + test) SHA256 `72cfbc9a88489ff2d54e03ebe4bf59a601be690cfb3a8e3e73dfa32eb082bce6`; config and per-file hashes in `results/config_and_hash.yaml`.

## Fields
| Column | Type | Description | Missing ('?') |
|---|---|---|---:|
| age | int | age in years | 0 (0.0%) |
| workclass | category | employer type | 2,799 (5.7%) |
| fnlwgt | int | census sampling weight | 0 (0.0%) |
| education | category | highest education level | 0 (0.0%) |
| education-num | int | education level as a number (1-16) | 0 (0.0%) |
| marital-status | category | marital status | 0 (0.0%) |
| occupation | category | occupation group | 2,809 (5.8%) |
| relationship | category | role in the household | 0 (0.0%) |
| race | category | race (census categories) | 0 (0.0%) |
| sex | category | sex | 0 (0.0%) |
| capital-gain | int | capital gains (USD) | 0 (0.0%) |
| capital-loss | int | capital losses (USD) | 0 (0.0%) |
| hours-per-week | int | usual working hours per week | 0 (0.0%) |
| native-country | category | country of origin | 857 (1.8%) |
| income | category | label: annual income >50K or <=50K (USD) | 0 (0.0%) |

Label balance: >50K 23.9%, <=50K 76.1%.

## Known biases
- **Sampling and time:** a 1994 US census extract; incomes are not adjusted for inflation and the 50K threshold reflects 1994 dollars, so the data does not describe today's population.
- **Group imbalance:** Male 66.8% vs Female 33.2%; White 85.5% of rows (Black 9.6%, Asian-Pac-Islander 3.1%, Amer-Indian-Eskimo 1.0%, Other 0.8%); United-States 89.7% of native-country.
- **Label disparity:** >50K rate is 30.4% for men vs 10.9% for women and 25.4% for White vs 12.1% for Black respondents. A model trained on this data will learn these historical gaps; group-wise metrics are needed before any use.
- **Missingness is not random:** '?' is concentrated in workclass, occupation and native-country (3,620 rows); mode imputation pushes these people into the majority category.
- **Proxy features:** relationship (Husband/Wife) and marital-status encode sex; fnlwgt is a sampling weight, not a personal attribute.
- **Label noise:** 5 groups of identical feature rows carry different labels (step 6).

## Privacy concerns
- Records are anonymised (no names or IDs), but **quasi-identifiers** - age, sex, race, native-country, occupation, marital status and education - can be combined to single out people, especially in small groups (rare countries or occupations).
- Several columns are **sensitive attributes** (race, sex, marital status, national origin); using them for decisions about individuals (credit, hiring, insurance) can be discriminatory and is regulated in many jurisdictions.
- Use for teaching and benchmarking; do not join with other data to re-identify individuals.

## Intended use and limitations
- Intended: teaching and benchmarking of tabular ML pipelines and fairness analysis.
- Not intended: real decisions about people's income, credit or employment.

## Processing applied in this lab
- '?' treated as missing; categorical values imputed with the training mode, numeric with the training median; numeric features clipped to the training 1st/99th percentiles; exact duplicates removed from the training split only; derived features capital_net and hours_per_age added. Details: `report.md`.
