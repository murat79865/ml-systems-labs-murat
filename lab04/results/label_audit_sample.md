# 50-row re-check against the data dictionary (step 6)

Rules applied to every row: categories in the UCI dictionary, schema ranges, education vs education-num, Husband/Male and Wife/Female, valid label. Full columns in `label_audit_sample.csv`.

| row | age | education | education-num | relationship | sex | hours-per-week | capital-gain | capital-loss | income | missing_fields | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7762 | 56 | HS-grad | 9 | Unmarried | Female | 40 | 0 | 0 | <=50K | - | valid |
| 23881 | 25 | HS-grad | 9 | Own-child | Male | 40 | 0 | 0 | <=50K | - | valid |
| 30507 | 43 | Bachelors | 13 | Not-in-family | Female | 40 | 14344 | 0 | >50K | - | valid |
| 28911 | 32 | HS-grad | 9 | Husband | Male | 40 | 0 | 0 | <=50K | - | valid |
| 19484 | 39 | Bachelors | 13 | Wife | Female | 30 | 0 | 0 | <=50K | - | valid |
| 43031 | 20 | HS-grad | 9 | Unmarried | Female | 40 | 0 | 0 | <=50K | - | valid |
| 28188 | 54 | HS-grad | 9 | Not-in-family | Male | 45 | 0 | 0 | <=50K | - | valid |
| 12761 | 25 | Bachelors | 13 | Wife | Female | 40 | 0 | 1887 | >50K | - | valid |
| 40834 | 30 | HS-grad | 9 | Not-in-family | Male | 50 | 0 | 0 | <=50K | - | valid |
| 27875 | 33 | Some-college | 10 | Own-child | Female | 40 | 0 | 0 | <=50K | - | valid |
| 1276 | 42 | Some-college | 10 | Husband | Male | 70 | 0 | 0 | <=50K | - | valid |
| 22608 | 24 | Bachelors | 13 | Not-in-family | Male | 30 | 0 | 0 | <=50K | - | valid |
| 36230 | 48 | Some-college | 10 | Husband | Male | 40 | 0 | 0 | <=50K | - | valid |
| 13398 | 21 | 11th | 7 | Own-child | Male | 45 | 0 | 0 | <=50K | - | valid |
| 43536 | 37 | Bachelors | 13 | Other-relative | Male | 40 | 0 | 0 | <=50K | - | valid |
| 18627 | 36 | Assoc-voc | 11 | Husband | Male | 60 | 0 | 0 | <=50K | - | valid |
| 38424 | 26 | Bachelors | 13 | Not-in-family | Female | 45 | 0 | 0 | <=50K | - | valid |
| 35505 | 25 | 5th-6th | 3 | Other-relative | Male | 40 | 0 | 0 | <=50K | - | valid |
| 2372 | 66 | HS-grad | 9 | Husband | Male | 42 | 0 | 0 | >50K | - | valid |
| 3375 | 37 | Assoc-voc | 11 | Not-in-family | Female | 4 | 0 | 0 | <=50K | - | valid |
| 28331 | 54 | Bachelors | 13 | Not-in-family | Female | 38 | 0 | 1408 | <=50K | - | valid |
| 46820 | 45 | Doctorate | 16 | Unmarried | Female | 35 | 0 | 3004 | >50K | - | valid |
| 147 | 33 | HS-grad | 9 | Husband | Male | 40 | 0 | 0 | >50K | - | valid |
| 21063 | 67 | Bachelors | 13 | Husband | Male | 60 | 9386 | 0 | >50K | workclass, occupation | valid (has '?') |
| 20139 | 31 | Bachelors | 13 | Unmarried | Female | 40 | 0 | 0 | <=50K | native-country | valid (has '?') |
| 23288 | 22 | Some-college | 10 | Own-child | Female | 39 | 0 | 0 | <=50K | - | valid |
| 18957 | 62 | Some-college | 10 | Husband | Male | 24 | 0 | 0 | <=50K | workclass, occupation | valid (has '?') |
| 31256 | 22 | Some-college | 10 | Other-relative | Male | 39 | 0 | 0 | <=50K | - | valid |
| 29642 | 24 | Bachelors | 13 | Wife | Female | 40 | 0 | 0 | <=50K | - | valid |
| 23638 | 32 | 10th | 6 | Husband | Male | 40 | 0 | 0 | <=50K | - | valid |
| 18067 | 35 | Masters | 14 | Wife | Female | 35 | 0 | 0 | >50K | - | valid |
| 879 | 45 | Bachelors | 13 | Husband | Male | 40 | 0 | 0 | >50K | - | valid |
| 7155 | 40 | HS-grad | 9 | Own-child | Male | 40 | 0 | 0 | <=50K | - | valid |
| 43769 | 27 | 5th-6th | 3 | Not-in-family | Male | 75 | 0 | 0 | <=50K | - | valid |
| 17697 | 32 | HS-grad | 9 | Not-in-family | Male | 52 | 0 | 0 | <=50K | - | valid |
| 21167 | 34 | HS-grad | 9 | Unmarried | Female | 30 | 0 | 0 | <=50K | - | valid |
| 3136 | 58 | HS-grad | 9 | Husband | Male | 84 | 0 | 0 | <=50K | - | valid |
| 34254 | 25 | HS-grad | 9 | Husband | Male | 40 | 0 | 0 | <=50K | - | valid |
| 17657 | 44 | HS-grad | 9 | Husband | Male | 55 | 0 | 0 | <=50K | - | valid |
| 37254 | 21 | HS-grad | 9 | Own-child | Male | 36 | 0 | 0 | <=50K | - | valid |
| 1422 | 61 | HS-grad | 9 | Not-in-family | Female | 40 | 0 | 0 | <=50K | - | valid |
| 32507 | 18 | 11th | 7 | Own-child | Male | 15 | 0 | 0 | <=50K | - | valid |
| 30150 | 40 | HS-grad | 9 | Husband | Male | 50 | 0 | 0 | <=50K | - | valid |
| 14172 | 31 | HS-grad | 9 | Own-child | Male | 40 | 0 | 0 | <=50K | - | valid |
| 32273 | 28 | HS-grad | 9 | Not-in-family | Male | 30 | 0 | 0 | <=50K | - | valid |
| 48262 | 26 | HS-grad | 9 | Own-child | Male | 40 | 0 | 0 | <=50K | - | valid |
| 10007 | 46 | 10th | 6 | Husband | Male | 40 | 0 | 0 | <=50K | - | valid |
| 47649 | 27 | Some-college | 10 | Other-relative | Male | 50 | 0 | 0 | >50K | - | valid |
| 43058 | 51 | Assoc-voc | 11 | Unmarried | Female | 40 | 0 | 0 | <=50K | - | valid |
| 708 | 50 | Bachelors | 13 | Husband | Male | 40 | 0 | 0 | >50K | - | valid |
