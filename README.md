# ML Systems Labs

Individual laboratory repository for **Machine Learning (CYI5SE039CY)**, Group 11.
Textbook: Vijay Janapa Reddi, *Introduction to Machine Learning Systems: Principles and Practices of
Engineering Artificially Intelligent Systems*.

## Layout

```
repo/
  README.md            this file
  requirements.txt     exact pinned stack (Python 3.11.8)
  .gitignore           ignores data/, .venv/, caches
  lab01/
    report.md          goal, method, results, conclusions (no code)
    src/               all code for the lab
    results/           every output the code produces
  lab02/ ... lab11/    added lab by lab
  project/             README.md, src/, results/, data_card.md, model_card.md
data/                  datasets (git-ignored; scripts download them if missing)
```

## Environment (Python 3.11.8)

Install Python **3.11.8** (python.org installers keep old versions available), then from the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, if `python3.11` is not found, use `py -3.11 -m venv .venv`.

Check it: `python lab01/src/print_versions.py` must show `Python : 3.11.8 ... OK` and `problems: 0`.

## Running a lab

```bash
python lab01/src/run_all.py
```

`run_all.py` runs every step of the lab, writes all outputs into `lab01/results/` and regenerates
`lab01/report.md` from those outputs. Each lab ends with a commit.

## Reproducibility rules followed in every lab

- Seeds: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`.
- Timings: one warm-up, then at least 5 repetitions, median reported; single-sample inference: 100 repetitions, median.
- Every lab saves Python and all package versions to `results/versions.txt`.
- Reports contain results and explanations only; code lives in `src/`.

## Note: mlflow vs pyarrow conflict in the pinned stack

`mlflow==2.14.1` declares `pyarrow<16`, but the course pins `pyarrow==16.1.0`, so a plain
`pip install -r requirements.txt` fails with `ResolutionImpossible`. To keep every pinned version,
all packages except mlflow were installed from the requirements file, then mlflow's other
dependencies (constrained to the pins), then `pip install --no-deps mlflow==2.14.1`.
`lab01/results/versions.txt` confirms Python 3.11.8 and all 25 pins (problems: 0).
thop reports `0.1.1-2209072238`, which is the same PEP 440 version as `0.1.1.post2209072238`.
