import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, TransformerMixin


def constant_features(df: pd.DataFrame, features: list) -> list:
    counts = df[features].nunique(dropna=True)
    return counts[counts <= 1].index.tolist()


def _within_class_redundant(df: pd.DataFrame, features: list, label_col: str, threshold: float) -> np.ndarray:
    """Boolean matrix: True where a feature pair is redundant inside EVERY class.

    Pooled correlation is misleading on this dataset: the flood classes send uniform
    single-size packets, so e.g. Fwd Packet Length Max/Min correlate at 0.97 pooled but
    only 0.03 within BENIGN -- the class where their difference carries the signal.
    """
    redundant = np.ones((len(features), len(features)), dtype=bool)
    for _, group in df.groupby(label_col, observed=True):
        varying = (group[features].std() > 0).to_numpy()
        corr = group[features].corr().abs().to_numpy()
        both_constant = ~varying[:, None] & ~varying[None, :]
        one_constant = varying[:, None] ^ varying[None, :]
        redundant &= np.where(both_constant, True, np.where(one_constant, False, corr >= threshold))
    return redundant


def select_features(df: pd.DataFrame, features: list, label_col: str, threshold: float = 0.99):
    """Drop constant features, then collapse groups of features that are redundant both
    pooled and within every class, keeping the first of each group in column order.

    Returns (kept, constant, dropped) where dropped maps each removed feature to the
    feature kept in its place.
    """
    constant = constant_features(df, features)
    live = [f for f in features if f not in constant]

    pooled = df[live].corr().abs().to_numpy() >= threshold
    adjacency = pooled & _within_class_redundant(df, live, label_col, threshold)
    np.fill_diagonal(adjacency, False)

    _, component = connected_components(adjacency, directed=False)
    kept, dropped = [], {}
    for comp in np.unique(component):
        members = [live[i] for i in np.flatnonzero(component == comp)]
        kept.append(members[0])
        dropped.update({m: members[0] for m in members[1:]})

    kept_set = set(kept)
    return [f for f in live if f in kept_set], constant, dropped


def signed_log1p(X):
    """log1p that tolerates negatives -- a few CICFlowMeter features (e.g. Init_Win_bytes,
    min_seg_size_forward) contain negative sentinel values."""
    return np.sign(X) * np.log1p(np.abs(X))


class MaxImputer(BaseEstimator, TransformerMixin):
    """Fill NaN with each column's training maximum.

    In this dataset NaN only occurs in Flow Bytes/s and Flow Packets/s, exactly when
    Flow Duration == 0 (rate = x / 0). The honest value is "faster than anything
    observed", so the training max preserves the feature's ordering; the fill is
    learned from training data only to avoid leakage.
    """

    def fit(self, X, y=None):
        self.fill_ = np.nanmax(np.asarray(X, dtype="float64"), axis=0)
        return self

    def transform(self, X):
        X = np.array(X, dtype="float64", copy=True)
        rows, cols = np.nonzero(np.isnan(X))
        X[rows, cols] = self.fill_[cols]
        return X
