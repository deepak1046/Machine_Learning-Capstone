"""Accuracy/F1 levers for any (task, split): LightGBM, engineered features, XGBoost, ensemble, tuned class offsets.

usage: python experiments/push_f1.py <13|family> <group|random>      (THREADS env var sets CPU threads)
Validation for early stopping and offset tuning is carved out of TRAIN (whole behavior groups for the group split,
random rows for the random split); the test rows are only used for the final scores.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.preprocessing.labels import FAMILY_MAP, FAMILY_ENCODING
task, split = sys.argv[1], sys.argv[2]
THREADS = int(os.environ.get("THREADS", "6"))
feats = json.loads((ROOT / "reports/final_features.json").read_text())["features"]
df = pd.read_parquet(ROOT / "data/processed/training_sample.parquet", columns=feats + ["Label", "Label_Encoded", "behavior_key", f"split_{split}"])
y = df["Label"].map(FAMILY_MAP).map(FAMILY_ENCODING).to_numpy() if task == "family" else df["Label_Encoded"].to_numpy()
n_classes = int(y.max()) + 1
train = df[f"split_{split}"].eq("train").to_numpy()
if split == "group":
    val = train & (pd.util.hash_array(df["behavior_key"].to_numpy().astype("uint64")) % 10 == 0)
else:
    rng = np.random.default_rng(7); val = np.zeros(len(df), bool); tr = np.flatnonzero(train); val[rng.choice(tr, len(tr) // 10, replace=False)] = True
fit, test = train & ~val, ~train
print(f"[{task}/{split}] fit {fit.sum():,} val {val.sum():,} test {test.sum():,} classes {n_classes}", flush=True)

SERVICE = {53: 1, 123: 2, 137: 3, 111: 4, 161: 5, 389: 6, 1900: 7, 69: 8, 1434: 9, 19: 10, 11211: 11, 80: 12, 443: 13}
X0 = df[feats]; X1 = df[feats].copy(); e = 1.0
X1["bytes_per_fwd_pkt"] = df["Total Length of Fwd Packets"] / (df["Total Fwd Packets"] + e)
X1["bytes_per_bwd_pkt"] = df["Total Length of Bwd Packets"] / (df["Total Backward Packets"] + e)
X1["fwd_bwd_pkt_ratio"] = df["Total Fwd Packets"] / (df["Total Backward Packets"] + e)
X1["fwd_bwd_byte_ratio"] = df["Total Length of Fwd Packets"] / (df["Total Length of Bwd Packets"] + e)
X1["frag_1472"] = (df["Fwd Packet Length Max"] == 1472).astype("int8")
X1["src_service"] = df["Source Port"].map(SERVICE).fillna(0).astype("int8"); X1["dst_service"] = df["Destination Port"].map(SERVICE).fillna(0).astype("int8")
X1["src_port_low"] = (df["Source Port"] < 1024).astype("int8"); X1["dst_port_low"] = (df["Destination Port"] < 1024).astype("int8")
w = compute_sample_weight("balanced", y[fit])
out = []
def score(name, proba, extra=None):
    p = proba.argmax(1) if extra is None else (np.log(np.clip(proba, 1e-9, 1)) + extra).argmax(1)
    r = dict(task=task, split=split, model=name, macro_f1=round(f1_score(y[test], p, average="macro"), 4), accuracy=round(accuracy_score(y[test], p), 4))
    out.append(r); print(r, flush=True)
    pd.DataFrame(out).to_csv(ROOT / f"reports/push_f1_{task}_{split}.csv", index=False)

def lgbm(X):
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=800, learning_rate=0.08, num_leaves=127, min_child_samples=40, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, n_jobs=THREADS, verbose=-1, random_state=42)
    m.fit(X[fit], y[fit], sample_weight=w, eval_set=[(X[val], y[val])], callbacks=[lgb.early_stopping(40, verbose=False)])
    return m.predict_proba(X[val]), m.predict_proba(X[test])
pv_l, pt_l = lgbm(X0); score("LightGBM", pt_l)
pv_e, pt_e = lgbm(X1); score("LightGBM + engineered features", pt_e)
xm = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist", n_jobs=THREADS, random_state=42)
xm.fit(X0[fit], y[fit], sample_weight=w); pv_x, pt_x = xm.predict_proba(X0[val]), xm.predict_proba(X0[test]); score("XGBoost (train minus val)", pt_x)
pv_a, pt_a = (pv_x + pv_l + pv_e) / 3, (pt_x + pt_l + pt_e) / 3; score("Ensemble (XGB + LGB + LGB-eng)", pt_a)

def offsets(pv, yv, rounds=3):
    lp = np.log(np.clip(pv, 1e-9, 1)); b = np.zeros(pv.shape[1]); best = f1_score(yv, lp.argmax(1), average="macro")
    for _ in range(rounds):
        for c in range(len(b)):
            for d in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
                b2 = b.copy(); b2[c] += d; s = f1_score(yv, (lp + b2).argmax(1), average="macro")
                if s > best + 1e-5: best, b = s, b2
    return b
score("LightGBM + eng + tuned offsets", pt_e, offsets(pv_e, y[val]))
score("Ensemble + tuned offsets", pt_a, offsets(pv_a, y[val]))
print("done", flush=True)
