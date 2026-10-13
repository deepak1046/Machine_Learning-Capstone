"""Run every final model once on its test split and keep what the metrics need: true class, predicted class, P(attack) and the
behavior-group bucket of each flow. Predictions are cached in data/processed/predictions/ (not committed).
usage: python experiments/collect_predictions.py
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, joblib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.evaluation.report import N_BUCKETS
from src.explainability.attention import load_network
from src.preprocessing.labels import FAMILY_MAP, FAMILY_ENCODING

feats = json.loads((ROOT / "reports/final_features.json").read_text())["features"]
df = pd.read_parquet(ROOT / "data/processed/training_sample.parquet",
                     columns=feats + ["Label", "Label_Encoded", "behavior_key", "split_group", "split_random"])
OUT = ROOT / "data/processed/predictions"; MODELS = ROOT / "saved_models"
offsets = lambda name: np.asarray(json.loads((MODELS / name).read_text()))

def with_offsets(proba, b):
    z = np.log(np.clip(proba, 1e-9, 1)) + b
    z = np.exp(z - z.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)

def tree(file, off=None, subset=None):
    def run(X):
        model = joblib.load(MODELS / file)
        cols = model.get_booster().feature_names if subset == "booster" else feats
        p = model.predict_proba(X[cols]); return with_offsets(p, offsets(off)) if off else p
    return run

def network_v2(split, off=None):
    def run(X):
        p = load_network(MODELS / f"cnn_bilstm_attention_v2_{split}.keras").predict(
            joblib.load(MODELS / f"hybrid_v2_preprocessor_{split}.joblib").transform(X), batch_size=2048, verbose=0)
        return with_offsets(p, offsets(off)) if off else p
    return run

def network_v1(X):
    Z = joblib.load(MODELS / "hybrid_preprocessor_group.joblib").transform(X)
    return load_network(MODELS / "cnn_bilstm_attention_group.keras").predict(Z.reshape(len(Z), Z.shape[1], 1), batch_size=2048, verbose=0)

def logistic(X):
    return joblib.load(MODELS / "LogisticRegression_group.joblib").predict_proba(X)

# (name, task, split, runner); the group-split offsets for xgb13/LightGBM were not helpful or not saved, so only served rules are listed
CONFIGS = [
    ("XGBoost", "13", "group", tree("XGBoost_group.joblib")),
    ("XGBoost (no ports)", "13", "group", tree("XGBoost_no_ports_group.joblib", subset="booster")),
    ("Logistic regression", "13", "group", logistic),
    ("CNN-BiLSTM-Attention v1", "13", "group", network_v1),
    ("CNN-BiLSTM-Attention v2", "13", "group", network_v2("group")),
    ("CNN-BiLSTM-Attention v2 + offsets", "13", "group", network_v2("group", "offsets_hybrid13_group.json")),
    ("XGBoost", "family", "group", tree("XGBoost_family_group.joblib")),
    ("XGBoost + offsets", "family", "group", tree("XGBoost_family_group_v2.joblib", "offsets_xgb8_group.json")),
    ("XGBoost", "13", "random", tree("XGBoost_random.joblib")),
    ("LightGBM", "13", "random", tree("LightGBM_random.joblib")),
    ("CNN-BiLSTM-Attention v2", "13", "random", network_v2("random")),
    ("XGBoost", "family", "random", tree("XGBoost_family_random.joblib")),
]
slug = lambda s: "".join(ch if ch.isalnum() else "_" for ch in s.lower()).strip("_")
for name, task, split, run in CONFIGS:
    path = OUT / f"{task}_{split}_{slug(name)}.npz"
    if path.exists(): print("cached", path.name, flush=True); continue
    t = time.time()
    test = df[df[f"split_{split}"].eq("test")]
    y = test["Label"].map(FAMILY_MAP).map(FAMILY_ENCODING).to_numpy() if task == "family" else test["Label_Encoded"].to_numpy()
    proba = run(test[feats])
    np.savez_compressed(path, y=y.astype("int8"), pred=proba.argmax(1).astype("int8"), p_attack=(1 - proba[:, 0]).astype("float32"),
                        bucket=(test["behavior_key"].to_numpy().astype("uint64") % N_BUCKETS).astype("int16"))
    print(f"{task:>6} {split:>6} {name:<36} {len(y):,} rows  accuracy {(proba.argmax(1) == y).mean():.4f}  {time.time() - t:.0f}s", flush=True)
print("done", flush=True)
