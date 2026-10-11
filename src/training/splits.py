import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def behavior_keys(df: pd.DataFrame, columns: list) -> np.ndarray:
    """One hash per row over `columns`; rows with identical values share a key."""
    return pd.util.hash_pandas_object(df[columns], index=False).to_numpy()


def random_split(labels, test_size: float = 0.2, seed: int = 42) -> np.ndarray:
    """Stratified random split. Returns a boolean mask, True = test row."""
    idx = np.arange(len(labels))
    _, test_idx = train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
    mask = np.zeros(len(labels), dtype=bool)
    mask[test_idx] = True
    return mask


def group_split(labels, groups, test_size: float = 0.2, seed: int = 42) -> np.ndarray:
    """Split so every row of a group lands on the same side, across ALL classes.

    Groups are assigned globally rather than per class because identical behavior
    vectors often carry different labels here (one fragmented-UDP pattern appears under
    LDAP, DNS, SNMP and MSSQL). Groups are visited in random order and moved to test only
    if that keeps every class within its test budget, so large recurring behaviors tend
    to stay in training and the test set leans toward rarer behaviors -- a conservative
    (harder) evaluation. Returns a boolean mask, True = test row.
    """
    rng = np.random.default_rng(seed)
    classes, label_idx = np.unique(np.asarray(labels), return_inverse=True)
    _, group_idx = np.unique(np.asarray(groups), return_inverse=True)

    counts = np.zeros((group_idx.max() + 1, len(classes)), dtype=np.int64)
    np.add.at(counts, (group_idx, label_idx), 1)

    budget = test_size * np.bincount(label_idx, minlength=len(classes))
    taken = np.zeros(len(classes))
    chosen = np.zeros(len(counts), dtype=bool)
    for g in rng.permutation(len(counts)):
        if np.all(taken + counts[g] <= budget):
            chosen[g] = True
            taken += counts[g]

    return chosen[group_idx]


def twin_rate(keys: np.ndarray, test_mask: np.ndarray) -> float:
    """Fraction of test rows whose key also appears in the training rows."""
    return float(pd.Series(keys[test_mask]).isin(keys[~test_mask]).mean())
