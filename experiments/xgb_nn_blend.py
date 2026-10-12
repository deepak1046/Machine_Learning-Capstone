"""Plain 50/50 probability average of the served XGBoost and the v2 network, group split. Nothing is tuned: one fixed recipe, scored once."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.metrics import f1_score, accuracy_score
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.explainability.attention import load_network
feats = json.loads((ROOT / "reports/final_features.json").read_text())["features"]
df = pd.read_parquet(ROOT / "data/processed/training_sample.parquet", columns=feats + ["Label_Encoded", "split_group"])
test = df[df.split_group.eq("test")]; y = test["Label_Encoded"].to_numpy(); X = test[feats]
p_x = joblib.load(ROOT / "saved_models/XGBoost_group.joblib").predict_proba(X)
p_n = load_network(ROOT / "saved_models/cnn_bilstm_attention_v2_group.keras").predict(
    joblib.load(ROOT / "saved_models/hybrid_v2_preprocessor_group.joblib").transform(X), batch_size=2048, verbose=0)
rows = []
for name, p in [("XGBoost", p_x), ("Network v2 (raw)", p_n), ("Average of both", (p_x + p_n) / 2)]:
    rows.append({"model": name, "macro_f1": round(f1_score(y, p.argmax(1), average="macro"), 4), "accuracy": round(accuracy_score(y, p.argmax(1)), 4)})
out = pd.DataFrame(rows); print(out.to_string(index=False)); out.to_csv(ROOT / "reports/xgb_nn_blend_group.csv", index=False)
