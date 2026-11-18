"""Fitted transformation steps of the Lab 4 pipeline (pickled inside results/preprocessor.joblib).

AdultCleaner   - imputation (median for numeric, mode for categorical; '?' = missing) and outlier
                 clipping to the 1st/99th percentile. All statistics are learned in fit() from the data
                 it is given - the training split only.
FeatureAdder   - derived features capital_net = capital-gain - capital-loss and
                 hours_per_age = hours-per-week / (age - 16). Stateless.
make_preprocessor() - AdultCleaner -> FeatureAdder -> (StandardScaler | OneHotEncoder).

To load the joblib file elsewhere, put lab04/src on sys.path first (these classes live here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DERIVED = ["capital_net", "hours_per_age"]


class AdultCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, numeric=None, categorical=None, missing_token="?", clip_quantiles=(0.01, 0.99)):
        self.numeric = numeric
        self.categorical = categorical
        self.missing_token = missing_token
        self.clip_quantiles = clip_quantiles

    def fit(self, X: pd.DataFrame, y=None):
        num = X[self.numeric].astype(float).to_numpy()
        cat = X[self.categorical].astype(object).to_numpy()
        self.num_imputer_ = SimpleImputer(strategy="median").fit(num)
        self.cat_imputer_ = SimpleImputer(strategy="most_frequent", missing_values=self.missing_token).fit(cat)
        num_imputed = self.num_imputer_.transform(num)
        lo, hi = self.clip_quantiles
        self.clip_lower_ = np.percentile(num_imputed, 100 * lo, axis=0)
        self.clip_upper_ = np.percentile(num_imputed, 100 * hi, axis=0)
        self.n_rows_fit_ = int(X.shape[0])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        num = self.num_imputer_.transform(X[self.numeric].astype(float).to_numpy())
        num = np.clip(num, self.clip_lower_, self.clip_upper_)
        cat = self.cat_imputer_.transform(X[self.categorical].astype(object).to_numpy())
        out = pd.DataFrame(num, columns=self.numeric, index=X.index)
        for j, c in enumerate(self.categorical):
            out[c] = pd.Series(cat[:, j], index=X.index, dtype=object)
        return out

    def fitted_values(self) -> dict:
        return {
            "numeric_median": dict(zip(self.numeric, map(float, self.num_imputer_.statistics_))),
            "categorical_mode": dict(zip(self.categorical, map(str, self.cat_imputer_.statistics_))),
            "clip_lower": dict(zip(self.numeric, map(float, self.clip_lower_))),
            "clip_upper": dict(zip(self.numeric, map(float, self.clip_upper_))),
            "n_rows_fit": self.n_rows_fit_,
        }


class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_derived=True):
        self.add_derived = add_derived

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.add_derived:
            out["capital_net"] = out["capital-gain"] - out["capital-loss"]
            out["hours_per_age"] = out["hours-per-week"] / np.maximum(out["age"] - 16, 1)
        return out


def make_preprocessor(numeric, categorical, missing_token, clip_quantiles, add_derived=True) -> Pipeline:
    num_cols = list(numeric) + (DERIVED if add_derived else [])
    encode = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
    ])
    return Pipeline([
        ("clean", AdultCleaner(list(numeric), list(categorical), missing_token, tuple(clip_quantiles))),
        ("features", FeatureAdder(add_derived)),
        ("encode", encode),
    ])
