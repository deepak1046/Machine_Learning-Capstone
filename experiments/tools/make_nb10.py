import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s)); code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 10: Performance metrics

Every final model, scored on its untouched test split, with the full metric set and 95% confidence intervals.

- **Splits.** *Group*: test behaviors never seen in training (the honest generalization estimate). *Random*: stratified random split (comparable to published work).
- **Tasks.** 13 attack types, and 8 attack families.
- **Intervals.** Cluster bootstrap over behavior groups (`src/evaluation/report.py`): flows that look alike differ only in ports, so whole groups are resampled together, which keeps the intervals honest. 1,000 resamples, shared by all models so that differences between two models are paired.
- **Attack vs. benign.** Any attack class counts as attack: detection rate = share of attack flows flagged, false positive rate = share of benign flows flagged.
- Predictions come from `experiments/collect_predictions.py`.""")
code('''import sys
from pathlib import Path
sys.path.append(str(Path("..").resolve()))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.evaluation.report import (bootstrap_draws, bootstrap_metrics, confusion_by_bucket, interval, per_class_table, summary_metrics)
from src.preprocessing.labels import CLASSES, FAMILY_CLASSES

ROOT = Path("..").resolve()
PRED, REPORTS, FIGS = ROOT / "data/processed/predictions", ROOT / "reports", ROOT / "reports/figures"
NAMES = {"13": list(CLASSES), "family": list(FAMILY_CLASSES)}
DISPLAY = {"xgboost": "XGBoost", "xgboost__no_ports": "XGBoost (no ports)", "logistic_regression": "Logistic regression",
           "cnn_bilstm_attention_v1": "CNN-BiLSTM-Attention v1", "cnn_bilstm_attention_v2": "CNN-BiLSTM-Attention v2",
           "cnn_bilstm_attention_v2___offsets": "CNN-BiLSTM-Attention v2 + offsets", "xgboost___offsets": "XGBoost + offsets",
           "lightgbm": "LightGBM", "paper_cnn_1_epoch": "Paper CNN (1 epoch)", "paper_lstm_1_epoch": "Paper LSTM (1 epoch)",
           "paper_cnn_10_epochs": "Paper CNN (10 epochs)"}
draws = bootstrap_draws(repeats=1000, seed=0)
files = sorted(PRED.glob("*.npz"))
print(len(files), "prediction files")''')
md("## Metrics with confidence intervals")
code('''rows, per_class_rows, boot, curves = [], [], {}, {}
for f in files:
    task, split, raw = f.stem.split("_", 2)
    slug = DISPLAY.get(raw, raw)
    d = np.load(f)
    n_classes = len(NAMES[task])
    cm_b = confusion_by_bucket(d["y"], d["pred"], d["bucket"], n_classes)
    point = summary_metrics(cm_b.sum(axis=0))
    boots = bootstrap_metrics(cm_b, draws)
    key = (task, split, slug)
    boot[key] = boots
    row = {"task": task, "split": split, "model": slug, "n_test": len(d["y"])}
    for metric, value in point.items():
        lo, hi = interval(boots[metric])
        row.update({metric: value, f"{metric}_lo": lo, f"{metric}_hi": hi})
    attack = (d["y"] != 0)
    row["roc_auc_attack"] = roc_auc_score(attack, d["p_attack"]); row["pr_auc_attack"] = average_precision_score(attack, d["p_attack"])
    rows.append(row)
    curves[key] = roc_curve(attack, d["p_attack"])[:2]
    pc = per_class_table(cm_b.sum(axis=0))
    for i, c in enumerate(NAMES[task]):
        per_class_rows.append({"task": task, "split": split, "model": slug, "class": c, **{k: v[i] for k, v in pc.items()}})
metrics = pd.DataFrame(rows)
metrics.to_csv(REPORTS / "performance_metrics.csv", index=False)
pd.DataFrame(per_class_rows).to_csv(REPORTS / "performance_per_class.csv", index=False)
show = ["task", "split", "model", "accuracy", "macro_f1", "macro_precision", "macro_recall", "weighted_f1", "mcc", "kappa"]
metrics[show].round(4)''')
md("## Macro-F1 and accuracy with 95% intervals")
code('''def fmt(r, m): return f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]"
table = metrics.assign(**{"macro-F1 [95% CI]": metrics.apply(lambda r: fmt(r, "macro_f1"), axis=1),
                          "accuracy [95% CI]": metrics.apply(lambda r: fmt(r, "accuracy"), axis=1)})
table[["task", "split", "model", "macro-F1 [95% CI]", "accuracy [95% CI]"]]''')
code('''fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
for ax, (task, split) in zip(axes.ravel(), [("13", "group"), ("13", "random"), ("family", "group"), ("family", "random")]):
    part = metrics[(metrics.task == task) & (metrics.split == split)].sort_values("macro_f1")
    y = np.arange(len(part))
    ax.barh(y, part["macro_f1"], xerr=[part["macro_f1"] - part["macro_f1_lo"], part["macro_f1_hi"] - part["macro_f1"]], color="#2a78d6", capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(part["model"], fontsize=8); ax.set_xlim(0.6 if task == "13" else 0.9, 1.0 if task == "family" else 0.85)
    ax.set_title(f"{task} classes, {split} split"); ax.set_xlabel("macro-F1 (95% CI)")
fig.tight_layout(); fig.savefig(FIGS / "performance_macro_f1_ci.png", dpi=150); plt.show()''')
md("## Attack vs. benign\nDetection rate, false positive rate and ranking quality (AUC) when every attack class is treated as one.")
code('''binary = metrics[["task", "split", "model", "attack_detection_rate", "attack_precision", "attack_f1", "false_positive_rate", "false_negative_rate", "roc_auc_attack", "pr_auc_attack"]]
binary.to_csv(REPORTS / "performance_binary.csv", index=False)
binary.round(5)''')
code('''fig, ax = plt.subplots(figsize=(7, 5.5))
for (task, split, slug), (fpr, tpr) in curves.items():
    if task == "13" and split == "group":
        ax.plot(fpr, tpr, label=slug)
ax.set_xscale("log"); ax.set_xlim(1e-5, 1); ax.set_ylim(0.9, 1.001)
ax.set_xlabel("false positive rate (log scale)"); ax.set_ylabel("detection rate"); ax.set_title("Attack vs. benign ROC, 13-class models, group split")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGS / "performance_roc_attack_vs_benign.png", dpi=150); plt.show()''')
md("## Is the difference between two models real?\nPaired bootstrap of the macro-F1 difference (same resampled groups for both models). If the interval excludes zero, the gap is not noise.")
code('''def paired(task, split, a, b):
    d = boot[(task, split, a)]["macro_f1"] - boot[(task, split, b)]["macro_f1"]
    lo, hi = interval(d); return {"task": task, "split": split, "A": a, "B": b, "A - B": d.mean(), "lo": lo, "hi": hi, "A better": bool(lo > 0), "B better": bool(hi < 0)}
slugs = lambda task, split: sorted(k[2] for k in boot if k[0] == task and k[1] == split)
pairs = []
for task, split in [("13", "group"), ("13", "random"), ("family", "group"), ("family", "random")]:
    names = slugs(task, split)
    pairs += [paired(task, split, a, b) for i, a in enumerate(names) for b in names[i + 1:]]
comparison = pd.DataFrame(pairs); comparison.to_csv(REPORTS / "performance_paired_differences.csv", index=False)
comparison.round(4)''')
md("## Per-class results\nPrecision, recall and F1 per class for each model on the group split (13 classes).")
code('''pc = pd.read_csv(REPORTS / "performance_per_class.csv")
pc[(pc.task == "13") & (pc.split == "group")].pivot(index="class", columns="model", values="f1").round(3)''')
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
nbf.write(nb, "notebooks/10_performance_metrics.ipynb")
