"""Leak-free feature engineering for the NFL Draft prediction project.

Every statistic (imputation medians, per-position means/stds, school target
encoding) is learned from the training data passed to ``fit`` only, then applied
unchanged to validation/test data. This keeps cross-validation estimates honest.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

COMBINE_TESTS = ["Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
                 "Broad_Jump", "Agility_3cone", "Shuttle"]
CATEGORICAL = ["Player_Type", "Position_Type", "Position"]
LOWER_IS_BETTER = ["Sprint_40yd", "Agility_3cone", "Shuttle"]
KG_TO_LB = 2.20462
CM_TO_IN = 1 / 2.54


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Turn the raw player table into a numeric model matrix.

    Parameters
    ----------
    use_age : include the Age column (median-imputed when missing).
    use_age_missing_flag : also include the Age_Missing indicator. In this
        dataset Age is almost only recorded for drafted players, so the flag
        works as a proxy for the target; see the notebook for the analysis.
    use_school : add smoothed (out-of-fold) school draft rate and school size.
    use_year : include the draft year as a numeric feature.
    smoothing : pseudo-count for the school target encoding.
    min_group : minimum rows for a Position to use its own statistics;
        rarer positions fall back to their Position_Type.
    """

    def __init__(self, use_age=True, use_age_missing_flag=True, use_school=True,
                 use_year=False, smoothing=10.0, min_group=10, random_state=42):
        self.use_age = use_age
        self.use_age_missing_flag = use_age_missing_flag
        self.use_school = use_school
        self.use_year = use_year
        self.smoothing = smoothing
        self.min_group = min_group
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        X = X.reset_index(drop=True)
        self._learn(X, None if y is None else np.asarray(y))
        self.feature_names_ = list(self._transform(X.head(5), self._dummy_school(5)).columns)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit, and return training features with out-of-fold school encoding."""
        X = X.reset_index(drop=True)
        y = None if y is None else np.asarray(y)
        self._learn(X, y)
        school = self._oof_school_te(X, y) if self.use_school else None
        out = self._transform(X, school)
        self.feature_names_ = list(out.columns)
        return out

    def _learn(self, X, y):
        self.cats_ = {c: sorted(X[c].unique()) for c in CATEGORICAL}
        counts = X["Position"].value_counts()
        self.rare_positions_ = set(counts[counts < self.min_group].index)
        key = self._group_key(X)
        cols = COMBINE_TESTS + ["Age"]
        self.global_median_ = X[cols].median()
        self.group_median_ = X.assign(_g=key).groupby("_g")[cols].median()
        self.group_mean_ = X.assign(_g=key).groupby("_g")[COMBINE_TESTS].mean()
        self.group_std_ = X.assign(_g=key).groupby("_g")[COMBINE_TESTS].std()
        self.global_mean_ = X[COMBINE_TESTS].mean()
        self.global_std_ = X[COMBINE_TESTS].std()
        if self.use_school:
            if y is None:
                raise ValueError("use_school=True requires y at fit time")
            self.prior_ = float(np.mean(y))
            self.school_stats_ = self._school_stats(X["School"], y)

    # --------------------------------------------------------------- school
    @staticmethod
    def _school_stats(school, y):
        return (pd.DataFrame({"s": np.asarray(school), "y": np.asarray(y)})
                .groupby("s")["y"].agg(["sum", "count"]))

    def _encode_school(self, school, stats):
        s = pd.Series(np.asarray(school))
        n = s.map(stats["count"]).fillna(0).to_numpy(dtype=float)
        tot = s.map(stats["sum"]).fillna(0).to_numpy(dtype=float)
        rate = (tot + self.smoothing * self.prior_) / (n + self.smoothing)
        return rate, n

    def _oof_school_te(self, X, y):
        """Out-of-fold encoding: a row's own label never feeds its own feature."""
        rate = np.zeros(len(X)); size = np.zeros(len(X))
        for tr, va in KFold(5, shuffle=True, random_state=self.random_state).split(X):
            stats = self._school_stats(X["School"].iloc[tr], y[tr])
            rate[va], size[va] = self._encode_school(X["School"].iloc[va], stats)
        return rate, size * 5 / 4  # rescale to full-training-set school size

    def _dummy_school(self, n):
        return (np.zeros(n), np.zeros(n)) if self.use_school else None

    # ------------------------------------------------------------ transform
    def transform(self, X):
        X = X.reset_index(drop=True)
        school = self._encode_school(X["School"], self.school_stats_) if self.use_school else None
        return self._transform(X, school)

    def _group_key(self, X):
        return np.where(X["Position"].isin(self.rare_positions_), X["Position_Type"], X["Position"])

    @staticmethod
    def _lookup(table, key, fallback):
        out = table.reindex(key)
        out.index = range(len(key))
        return out.fillna(fallback)

    def _transform(self, X, school=None):
        key = self._group_key(X)
        out = pd.DataFrame(index=range(len(X)))

        # ---- missingness signals ----------------------------------------
        out["Tests_Missing"] = X[COMBINE_TESTS].isna().sum(axis=1).to_numpy()
        if self.use_age and self.use_age_missing_flag:
            out["Age_Missing"] = X["Age"].isna().astype(int).to_numpy()

        # ---- imputation with position medians (training statistics) ------
        med = self._lookup(self.group_median_, key, self.global_median_)
        filled = X[COMBINE_TESTS + ["Age"]].reset_index(drop=True).fillna(med)
        for c in COMBINE_TESTS:
            out[c] = filled[c].to_numpy()
        if self.use_age:
            out["Age"] = filled["Age"].to_numpy()
        if self.use_year:
            out["Year"] = X["Year"].to_numpy()

        # ---- body measurements and derived athletic metrics --------------
        h, w = X["Height"].to_numpy(), X["Weight"].to_numpy()
        out["Height"], out["Weight"] = h, w
        out["BMI"] = w / h ** 2
        out["Speed_Score"] = (w * KG_TO_LB * 200) / out["Sprint_40yd"] ** 4  # standard NFL formula
        out["Burst"] = (out["Vertical_Jump"] + out["Broad_Jump"]) * CM_TO_IN  # inches
        out["Agility_Sum"] = out["Agility_3cone"] + out["Shuttle"]
        out["Bench_per_kg"] = out["Bench_Press_Reps"] / w

        # ---- standing relative to positional peers (training statistics) -
        mu = self._lookup(self.group_mean_, key, self.global_mean_)
        sd = self._lookup(self.group_std_, key, self.global_std_).replace(0, np.nan).fillna(1.0)
        for c in COMBINE_TESTS:
            z = (out[c] - mu[c].to_numpy()) / sd[c].to_numpy()
            out[c + "_pos_z"] = -z if c in LOWER_IS_BETTER else z  # positive = better than peers
        out["Athletic_Composite"] = out[[c + "_pos_z" for c in COMBINE_TESTS]].mean(axis=1)

        # ---- school ------------------------------------------------------
        if self.use_school and school is not None:
            out["School_DraftRate"], out["School_Size"] = school

        # ---- one-hot position information (categories fixed at fit time) -
        for c in CATEGORICAL:
            for cat in self.cats_[c]:
                out[f"{c}={cat}"] = (X[c].to_numpy() == cat).astype(int)
        return out
