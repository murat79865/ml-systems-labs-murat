"""Shared helpers for Lab 4: paths, config, schema/data dictionary, canonical view, hashing."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = LAB_DIR.parent
RESULTS_DIR = LAB_DIR / "results"
DATA_DIR = REPO_DIR / "data"                      # git-ignored
OPENML_CACHE = DATA_DIR / "openml"                # sklearn's download cache
RAW_DIR = DATA_DIR / "lab04" / "raw"              # untouched raw snapshot
PROCESSED_DIR = DATA_DIR / "lab04" / "processed"  # cleaned train/test tables
RAW_FILE = RAW_DIR / "adult_raw.csv"
for d in (RESULTS_DIR, RAW_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(LAB_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()
SEED = CFG["seed"]
TARGET = CFG["dataset"]["target"]
NULL = CFG["schema"]["null_token"]
TYPES = CFG["schema"]["types"]
FEATURES = [c for c in TYPES if c != TARGET]                     # the 14 input features
NUMERIC = [c for c in FEATURES if TYPES[c] == "int"]
CATEGORICAL = [c for c in FEATURES if TYPES[c] == "category"]

# UCI "adult.names" data dictionary: allowed values of every categorical column ('?' = missing).
DICTIONARY = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov",
                  "Without-pay", "Never-worked"],
    "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc",
                  "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                   "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                   "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                       "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba",
                       "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico",
                       "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan",
                       "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
                       "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
    "income": [">50K", "<=50K"],
}
# education -> education-num in the data dictionary (Preschool = 1 ... Doctorate = 16)
EDUCATION_NUM = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6, "11th": 7,
                 "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
                 "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16}


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_raw() -> pd.DataFrame:
    """The untouched raw snapshot written by ingest.py (empty fields = missing)."""
    return pd.read_csv(RAW_FILE, keep_default_na=True)


def canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """In-memory view used by every later step (the raw file is never changed):
    the label column is renamed to the schema name, categorical values become plain strings with
    surrounding whitespace removed, and every missing categorical value is written as the null token '?'.
    Numeric columns are left exactly as read."""
    df = raw.rename(columns={CFG["dataset"]["target_raw"]: TARGET}).copy()
    for c in CATEGORICAL + [TARGET]:
        s = df[c].astype(object)
        df[c] = [NULL if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v).strip() for v in s]
        df[c] = df[c].astype(object)
    return df


def to_binary(y: pd.Series) -> np.ndarray:
    return (y.astype(str) == CFG["dataset"]["positive_label"]).astype(int).to_numpy()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Deterministic CSV: fixed column order, '\\n' line endings, fixed float format."""
    df.to_csv(path, index=False, lineterminator="\n", float_format="%.10g")


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
