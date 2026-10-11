import pandas as pd
import xgboost as xgb


def tree_contributions(estimator, X: pd.DataFrame, class_index: int) -> tuple:
    """Per-feature contribution of each flow toward one class, straight from the trees
    (TreeSHAP via XGBoost's pred_contribs, in log-odds). Returns (contributions, bias): the
    contributions plus the bias term give the class's raw score before the softmax."""
    raw = estimator.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    # multiclass output is (rows, classes, features + 1); the last column is the bias
    per_class = raw[:, class_index, :]
    return pd.DataFrame(per_class[:, :-1], columns=X.columns, index=X.index), per_class[:, -1]
