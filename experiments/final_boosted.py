"""Served XGBoost models with a tuned decision rule.

Retrain XGBoost on the training rows minus a group-held-out validation slice (same recipe as the served models), tune per-class
log-probability offsets for macro-F1 on that validation slice only, and score the untouched group-split test set with and without
them. Saves saved_models/XGBoost[_family]_group_v2.joblib and saved_models/offsets_<task>_group.json.
usage: python experiments/final_boosted.py <13|family>
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, joblib, xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.preprocessing.labels import FAMILY_MAP, FAMILY_ENCODING
task = sys.argv[1]; THREADS = int(os.environ.get("THREADS", "6"))
feats = json.loads((ROOT / "reports/final_features.json").read_text())["features"]
df = pd.read_parquet(ROOT / "data/processed/training_sample.parquet", columns=feats + ["Label", "Label_Encoded", "behavior_key", "split_group"])
y = df["Label"].map(FAMILY_MAP).map(FAMILY_ENCODING).to_numpy() if task == "family" else df["Label_Encoded"].to_numpy()
train = df["split_group"].eq("train").to_numpy()
val = train & (pd.util.hash_array(df["behavior_key"].to_numpy().astype("uint64")) % 10 == 0)
fit, test = train & ~val, ~train
X = df[feats]
t = time.time()
model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist", n_jobs=THREADS, random_state=42)
model.fit(X[fit], y[fit], sample_weight=compute_sample_weight("balanced", y[fit]))
pv, pt = model.predict_proba(X[val]), model.predict_proba(X[test])
print(f"trained in {(time.time() - t) / 60:.0f} min", flush=True)

lp = np.log(np.clip(pv, 1e-9, 1)); b = np.zeros(pv.shape[1]); best = f1_score(y[val], lp.argmax(1), average="macro")
for _ in range(4):
    for c in range(len(b)):
        for d in (-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0):
            b2 = b.copy(); b2[c] += d; s = f1_score(y[val], (lp + b2).argmax(1), average="macro")
            if s > best + 1e-5: best, b = s, b2
raw, tuned = pt.argmax(1), (np.log(np.clip(pt, 1e-9, 1)) + b).argmax(1)
row = {"task": task, "split": "group", "val_macro_f1_tuned": round(best, 4),
       "test_macro_f1_raw": round(f1_score(y[test], raw, average="macro"), 4), "test_accuracy_raw": round(accuracy_score(y[test], raw), 4),
       "test_macro_f1_offsets": round(f1_score(y[test], tuned, average="macro"), 4), "test_accuracy_offsets": round(accuracy_score(y[test], tuned), 4)}
print(row, flush=True)
slug = "XGBoost_family_group_v2" if task == "family" else "XGBoost_group_v2"
joblib.dump(model, ROOT / f"saved_models/{slug}.joblib")
(ROOT / f"saved_models/offsets_{'xgb8' if task == 'family' else 'xgb13'}_group.json").write_text(json.dumps([float(v) for v in b]))
path = ROOT / "reports/xgb_offsets_results.csv"
old = pd.read_csv(path) if path.exists() else pd.DataFrame()
pd.concat([old[old["task"] != task] if len(old) else old, pd.DataFrame([row])]).to_csv(path, index=False)
print("done", flush=True)
