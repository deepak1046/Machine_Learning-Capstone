"""Performance metrics from confusion matrices, with a cluster bootstrap over behavior groups.

Flows that share every feature except ports are near-duplicates, so resampling single rows would make the
intervals too narrow. Instead every behavior group is assigned to one of N_BUCKETS buckets by its hash, the
confusion matrix is computed per bucket, and the bootstrap resamples whole buckets. A bucket holds complete
groups, so the dependence between look-alike flows stays inside it.
"""

import numpy as np

N_BUCKETS = 500


def confusion_by_bucket(y_true, y_pred, bucket, n_classes: int, n_buckets: int = N_BUCKETS) -> np.ndarray:
    """(buckets, classes, classes) counts, rows = true class, columns = predicted class."""
    index = (np.asarray(bucket, dtype="int64") * n_classes + np.asarray(y_true)) * n_classes + np.asarray(y_pred)
    return np.bincount(index, minlength=n_buckets * n_classes * n_classes).reshape(n_buckets, n_classes, n_classes)


def per_class_table(cm: np.ndarray) -> dict:
    """precision, recall, f1 and support per class from one confusion matrix."""
    tp = np.diag(cm).astype("float64")
    support, predicted = cm.sum(axis=1).astype("float64"), cm.sum(axis=0).astype("float64")
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def summary_metrics(cm: np.ndarray, benign: int = 0) -> dict:
    """Every headline number from one confusion matrix (classes with no true rows are left out of the macro averages)."""
    cm = cm.astype("float64")
    n = cm.sum()
    pc = per_class_table(cm)
    present = pc["support"] > 0
    rows, cols = cm.sum(axis=1), cm.sum(axis=0)

    # multiclass Matthews correlation coefficient and Cohen's kappa
    s, c = n, np.trace(cm)
    mcc_den = np.sqrt((s**2 - (cols**2).sum()) * (s**2 - (rows**2).sum()))
    p_o, p_e = c / n, (rows * cols).sum() / n**2

    # attack versus benign: any attack class counts as attack
    attack = np.ones(len(cm), bool)
    attack[benign] = False
    tp = cm[attack][:, attack].sum()
    fn = cm[attack][:, ~attack].sum()
    fp = cm[~attack][:, attack].sum()
    tn = cm[~attack][:, ~attack].sum()
    det_precision = tp / (tp + fp) if tp + fp else 0.0
    det_recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": c / n,
        "macro_f1": pc["f1"][present].mean(),
        "macro_precision": pc["precision"][present].mean(),
        "macro_recall": pc["recall"][present].mean(),
        "weighted_f1": (pc["f1"] * pc["support"]).sum() / n,
        "balanced_accuracy": pc["recall"][present].mean(),
        "mcc": (c * s - (rows * cols).sum()) / mcc_den if mcc_den else 0.0,
        "kappa": (p_o - p_e) / (1 - p_e) if p_e < 1 else 0.0,
        "attack_detection_rate": det_recall,
        "attack_precision": det_precision,
        "attack_f1": 2 * det_precision * det_recall / (det_precision + det_recall) if det_precision + det_recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
    }


def bootstrap_draws(n_buckets: int = N_BUCKETS, repeats: int = 1000, seed: int = 0) -> np.ndarray:
    """(repeats, buckets) how many times each bucket is drawn; shared by all models so differences are paired."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_buckets, np.full(n_buckets, 1 / n_buckets), size=repeats)


def bootstrap_metrics(cm_by_bucket: np.ndarray, draws: np.ndarray, benign: int = 0) -> dict:
    """metric -> array of bootstrap values, one per repeat."""
    sums = np.tensordot(draws, cm_by_bucket, axes=(1, 0))  # (repeats, classes, classes)
    values = [summary_metrics(cm, benign) for cm in sums]
    return {key: np.array([v[key] for v in values]) for key in values[0]}


def interval(samples: np.ndarray, level: float = 0.95) -> tuple:
    lo, hi = np.percentile(samples, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
