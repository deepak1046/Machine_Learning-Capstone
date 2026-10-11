import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def evaluate(y_true, y_pred, class_names: list):
    """Headline metrics plus a per-class table. Macro-F1 is the headline: with 13
    classes and heavy imbalance in the source data, plain accuracy is misleading."""
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=class_names)
    summary = {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    return summary, per_class


def plot_confusion(y_true, y_pred, class_names: list, title: str, path=None):
    """Row-normalized confusion matrix: each row shows where a true class's flows went."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
        xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 7}, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
