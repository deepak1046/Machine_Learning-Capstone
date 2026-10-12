"""Legitimate accuracy/F1 levers on the 13-class task, group split, same persisted splits.

1. XGBoost and LightGBM retrained on train minus a group-aware validation carve-out (test never touched)
2. probability-average ensemble
3. per-class log-probability offsets tuned on the validation carve-out to maximize macro-F1
4. engineered features (rate ratios, fragment flags, service-port codes) for LightGBM
"""
import json, time, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
feats = json.loads((ROOT / "reports/final_features.json").read_text())["features"]
classes = json.loads((ROOT / "reports/label_encoding.json").read_text())
print("encoding:", classes if len(str(classes)) < 400 else "ok", flush=True)

cols = feats + ["Label_Encoded", "behavior_key", "split_group"]
df = pd.read_parquet(ROOT / "data/processed/training_sample.parquet", columns=cols)
y = df["Label_Encoded"].to_numpy()
train = df["split_group"].eq("train").to_numpy()

# group-aware validation: whole behavior groups out of train (hash of the key, so deterministic)
key = df["behavior_key"].to_numpy()
val = train & (pd.util.hash_array(key.astype("uint64")) % 10 == 0)
fit = train & ~val
test = ~train
print(f"fit {fit.sum():,} val {val.sum():,} test {test.sum():,}", flush=True)

SERVICE = {53: 1, 123: 2, 137: 3, 111: 4, 161: 5, 389: 6, 1900: 7, 69: 8, 1434: 9, 19: 10, 11211: 11, 80: 12, 443: 13}

def engineer(d):
    o = d[feats].copy()
    eps = 1.0
    o["bytes_per_fwd_pkt"] = d["Total Length of Fwd Packets"] / (d["Total Fwd Packets"] + eps)
    o["bytes_per_bwd_pkt"] = d["Total Length of Bwd Packets"] / (d["Total Backward Packets"] + eps)
    o["fwd_bwd_pkt_ratio"] = d["Total Fwd Packets"] / (d["Total Backward Packets"] + eps)
    o["fwd_bwd_byte_ratio"] = d["Total Length of Fwd Packets"] / (d["Total Length of Bwd Packets"] + eps)
    o["frag_1472"] = (d["Fwd Packet Length Max"] == 1472).astype("int8")
    o["src_service"] = d["Source Port"].map(SERVICE).fillna(0).astype("int8")
    o["dst_service"] = d["Destination Port"].map(SERVICE).fillna(0).astype("int8")
    o["src_port_low"] = (d["Source Port"] < 1024).astype("int8")
    o["dst_port_low"] = (d["Destination Port"] < 1024).astype("int8")
    return o

def report(name, proba, mask=test):
    pred = proba.argmax(1)
    f1 = f1_score(y[mask], pred, average="macro"); acc = accuracy_score(y[mask], pred)
    print(f"{name:<38} macro-F1 {f1:.4f}  accuracy {acc:.4f}", flush=True)
    return f1, acc

results = {}
w_fit = compute_sample_weight("balanced", y[fit])

def run_lgb(name, X):
    t = time.time()
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=800, learning_rate=0.08, num_leaves=127,
                           min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           n_jobs=16, verbose=-1, random_state=42)
    m.fit(X[fit], y[fit], sample_weight=w_fit, eval_set=[(X[val], y[val])],
          callbacks=[lgb.early_stopping(40, verbose=False)])
    pv, pt = m.predict_proba(X[val]), m.predict_proba(X[test])
    print(f"  {name}: {m.best_iteration_} rounds, {time.time()-t:.0f}s", flush=True)
    return pv, pt

X0 = df[feats]
pv_l, pt_l = run_lgb("LightGBM base features", X0); results["lgb"] = report("LightGBM (base features)", pt_l)
X1 = engineer(df)
pv_le, pt_le = run_lgb("LightGBM engineered", X1); results["lgb_eng"] = report("LightGBM (+engineered features)", pt_le)

t = time.time()
xm = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist", n_jobs=16, random_state=42)
xm.fit(X0[fit], y[fit], sample_weight=w_fit)
pv_x, pt_x = xm.predict_proba(X0[val]), xm.predict_proba(X0[test])
print(f"  XGBoost: {time.time()-t:.0f}s", flush=True)
results["xgb"] = report("XGBoost (retrained, train minus val)", pt_x)

pv_ens, pt_ens = (pv_x + pv_l + pv_le) / 3, (pt_x + pt_l + pt_le) / 3
results["ens"] = report("Ensemble (XGB + LGB + LGB-eng)", pt_ens)

def tune_offsets(pv, yv, rounds=3):
    """Coordinate ascent on per-class log-prob offsets, scored on the validation carve-out only."""
    lp = np.log(np.clip(pv, 1e-9, 1)); b = np.zeros(pv.shape[1])
    best = f1_score(yv, (lp + b).argmax(1), average="macro")
    for _ in range(rounds):
        for c in range(len(b)):
            for d in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
                b2 = b.copy(); b2[c] += d
                s = f1_score(yv, (lp + b2).argmax(1), average="macro")
                if s > best + 1e-5: best, b = s, b2
    return b, best

for name, pv, pt in [("LGB-eng", pv_le, pt_le), ("Ensemble", pv_ens, pt_ens)]:
    b, vbest = tune_offsets(pv, y[val])
    pred = (np.log(np.clip(pt, 1e-9, 1)) + b).argmax(1)
    f1 = f1_score(y[test], pred, average="macro"); acc = accuracy_score(y[test], pred)
    print(f"{name + ' + tuned class offsets':<38} macro-F1 {f1:.4f}  accuracy {acc:.4f}  (val {vbest:.4f})", flush=True)
    results[name + "_offsets"] = (f1, acc)

(ROOT / "reports/push_group_f1_results.json").write_text(json.dumps(results, indent=1))
print("done", flush=True)
