"""Does more training data help? LightGBM + engineered features, group split, fixed test set."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import importlib.util, sys
ROOT = Path(__file__).resolve().parents[1]
feats = json.loads((ROOT / "reports/final_features.json").read_text())["features"]
df = pd.read_parquet(ROOT / "data/processed/training_sample.parquet", columns=feats + ["Label_Encoded", "behavior_key", "split_group"])
y = df["Label_Encoded"].to_numpy(); train = df["split_group"].eq("train").to_numpy()
val = train & (pd.util.hash_array(df["behavior_key"].to_numpy().astype("uint64")) % 10 == 0)
fit, test = train & ~val, ~train
SERVICE = {53: 1, 123: 2, 137: 3, 111: 4, 161: 5, 389: 6, 1900: 7, 69: 8, 1434: 9, 19: 10, 11211: 11, 80: 12, 443: 13}
X = df[feats].copy(); eps = 1.0
X["bytes_per_fwd_pkt"] = df["Total Length of Fwd Packets"] / (df["Total Fwd Packets"] + eps)
X["bytes_per_bwd_pkt"] = df["Total Length of Bwd Packets"] / (df["Total Backward Packets"] + eps)
X["fwd_bwd_pkt_ratio"] = df["Total Fwd Packets"] / (df["Total Backward Packets"] + eps)
X["fwd_bwd_byte_ratio"] = df["Total Length of Fwd Packets"] / (df["Total Length of Bwd Packets"] + eps)
X["frag_1472"] = (df["Fwd Packet Length Max"] == 1472).astype("int8")
X["src_service"] = df["Source Port"].map(SERVICE).fillna(0).astype("int8")
X["dst_service"] = df["Destination Port"].map(SERVICE).fillna(0).astype("int8")
X["src_port_low"] = (df["Source Port"] < 1024).astype("int8"); X["dst_port_low"] = (df["Destination Port"] < 1024).astype("int8")
rng = np.random.default_rng(0)
fit_idx = np.flatnonzero(fit)
rows = []
for frac in (0.1, 0.25, 0.5, 1.0):
    # sample whole behavior groups so a smaller set is a smaller world, not a thinned copy of the same one
    groups = df["behavior_key"].to_numpy()[fit_idx]
    ug = np.unique(groups); keep = set(rng.choice(ug, size=int(len(ug) * frac), replace=False).tolist()) if frac < 1 else None
    sel = fit_idx if keep is None else fit_idx[np.isin(groups, list(keep))]
    # and a thinned-rows variant: same groups, fewer rows
    for mode, idx in (("groups", sel),) + ((("rows", rng.choice(fit_idx, size=int(len(fit_idx) * frac), replace=False)),) if frac < 1 else ()):
        t = time.time()
        m = lgb.LGBMClassifier(objective="multiclass", n_estimators=800, learning_rate=0.08, num_leaves=127, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=16, verbose=-1, random_state=42)
        m.fit(X.iloc[idx], y[idx], sample_weight=compute_sample_weight("balanced", y[idx]),
              eval_set=[(X[val], y[val])], callbacks=[lgb.early_stopping(40, verbose=False)])
        p = m.predict(X[test])
        r = dict(frac=frac, mode=mode, rows=len(idx), f1=round(f1_score(y[test], p, average="macro"), 4), acc=round(accuracy_score(y[test], p), 4), rounds=m.best_iteration_, sec=round(time.time() - t))
        rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(ROOT / "reports/learning_curve_group.csv", index=False)
print("done", flush=True)
