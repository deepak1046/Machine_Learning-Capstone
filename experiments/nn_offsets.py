"""Class offsets for the CNN-BiLSTM-Attention v2 network.

Recomputes the exact validation slice used for early stopping (a stratified 10% of the training rows), tunes per-class
log-probability offsets for macro-F1 on it, and scores the untouched test set with and without them.
usage: python experiments/nn_offsets.py <group|random>
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.explainability.attention import load_network
split = sys.argv[1]
feats = json.loads((ROOT / "reports/final_features.json").read_text())["features"]
df = pd.read_parquet(ROOT / "data/processed/training_sample.parquet", columns=feats + ["Label_Encoded", f"split_{split}"])
train = df[f"split_{split}"].eq("train").to_numpy(); y = df["Label_Encoded"].to_numpy()
_, val_idx = train_test_split(np.flatnonzero(train), test_size=0.1, stratify=y[train], random_state=2026)  # same call as training
test_idx = np.flatnonzero(~train)

pre = joblib.load(ROOT / f"saved_models/hybrid_v2_preprocessor_{split}.joblib")
net = load_network(ROOT / f"saved_models/cnn_bilstm_attention_v2_{split}.keras")
pv = net.predict(pre.transform(df.loc[val_idx, feats]), batch_size=2048, verbose=0)
pt = net.predict(pre.transform(df.loc[test_idx, feats]), batch_size=2048, verbose=0)

lp = np.log(np.clip(pv, 1e-9, 1)); b = np.zeros(pv.shape[1]); best = f1_score(y[val_idx], lp.argmax(1), average="macro")
for _ in range(4):
    for c in range(len(b)):
        for d in (-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0):
            b2 = b.copy(); b2[c] += d; s = f1_score(y[val_idx], (lp + b2).argmax(1), average="macro")
            if s > best + 1e-5: best, b = s, b2
raw, tuned = pt.argmax(1), (np.log(np.clip(pt, 1e-9, 1)) + b).argmax(1)
row = {"split": split, "val_macro_f1_tuned": round(best, 4),
       "test_macro_f1_raw": round(f1_score(y[test_idx], raw, average="macro"), 4), "test_accuracy_raw": round(accuracy_score(y[test_idx], raw), 4),
       "test_macro_f1_offsets": round(f1_score(y[test_idx], tuned, average="macro"), 4), "test_accuracy_offsets": round(accuracy_score(y[test_idx], tuned), 4)}
print(row, flush=True)
(ROOT / f"saved_models/offsets_hybrid13_{split}.json").write_text(json.dumps([float(v) for v in b]))
path = ROOT / "reports/hybrid_v2_offsets_results.csv"
old = pd.read_csv(path) if path.exists() else pd.DataFrame()
pd.concat([old[old["split"] != split] if len(old) else old, pd.DataFrame([row])]).to_csv(path, index=False)
print("done", flush=True)
