"""Global SHAP analysis for the XGBoost classifiers, with the shap package used for the plots.

Per-flow contributions already exist in contributions.py (XGBoost's own TreeSHAP). This module adds
the dataset-level view: which features matter overall, which matter for each attack class, and how
feature values push a class up or down.
"""

import numpy as np
import pandas as pd
import xgboost as xgb


def stratified_sample(y: np.ndarray, per_class: int, seed: int = 42) -> np.ndarray:
    """Row positions with up to `per_class` rows from every class, in original order."""
    rng = np.random.default_rng(seed)
    picked = [
        rng.choice(np.flatnonzero(y == c), size=min(per_class, int((y == c).sum())), replace=False)
        for c in np.unique(y)
    ]
    return np.sort(np.concatenate(picked))


def shap_values(estimator, X: pd.DataFrame) -> tuple:
    """TreeSHAP values as (rows, features, classes) plus the per-class base values (log-odds).

    Uses XGBoost's own exact TreeSHAP (pred_contribs), which is the same algorithm as
    shap.TreeExplainer and needs no compiled extension from the shap package; shap is used
    here only for plotting.
    """
    raw = estimator.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)  # (rows, classes, features + 1)
    return raw[:, :, :-1].transpose(0, 2, 1), raw[:, :, -1].mean(axis=0)


def importance_table(values: np.ndarray, features: list, classes: list) -> pd.DataFrame:
    """Mean |SHAP| per feature and class; the 'all classes' column averages the class columns."""
    per_class = np.abs(values).mean(axis=0)  # (features, classes)
    table = pd.DataFrame(per_class, index=features, columns=classes)
    table.insert(0, "all classes", per_class.mean(axis=1))
    return table.sort_values("all classes", ascending=False)


def class_profile(table: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """The strongest features for each class, for a readable summary."""
    rows = []
    for c in table.columns[1:]:
        best = table[c].sort_values(ascending=False).head(top)
        rows.append({"class": c, "top features": ", ".join(f"{f} ({v:.2f})" for f, v in best.items())})
    return pd.DataFrame(rows)
