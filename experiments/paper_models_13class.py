"""The base paper's CNN and LSTM (Table 5 architecture and recipe) on OUR 13-class task.

Recipe as in notebooks/04b: up to 50,000 rows per class, StandardScaler after a naive fillna(0), Adam 1e-3, batch 64, no class weights,
1 epoch (the paper's setting) and, for the CNN, also 10 epochs as a properly trained reference. Same 60 features and the same persisted
splits as every other model; scored on the full untouched test split. The scaler is fitted on the training rows only.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, keras
from keras import layers
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.evaluation.report import N_BUCKETS
SEED = 2026
feats = json.loads((ROOT / "reports/final_features.json").read_text())["features"]
df = pd.read_parquet(ROOT / "data/processed/training_sample.parquet", columns=feats + ["Label_Encoded", "behavior_key", "split_group", "split_random"])
y_all = df["Label_Encoded"].to_numpy()

def cnn(): return keras.Sequential([layers.Input(shape=(len(feats), 1)), layers.Conv1D(32, 3, activation="relu"), layers.MaxPooling1D(2),
                                    layers.Flatten(), layers.Dense(128, activation="relu"), layers.Dense(13, activation="softmax")])
def lstm(): return keras.Sequential([layers.Input(shape=(len(feats), 1)), layers.LSTM(128, return_sequences=True), layers.LSTM(64),
                                     layers.Dense(128, activation="relu"), layers.Dense(13, activation="softmax")])
RUNS = [("Paper CNN 1 epoch", cnn, 1), ("Paper LSTM 1 epoch", lstm, 1), ("Paper CNN 10 epochs", cnn, 10)]

rows = []
for split in ("group", "random"):
    train = df[f"split_{split}"].eq("train").to_numpy()
    rng = np.random.default_rng(SEED)
    fit = np.sort(np.concatenate([rng.choice(np.flatnonzero(train & (y_all == c)), size=min(50_000, int((train & (y_all == c)).sum())), replace=False) for c in range(13)]))
    test = df[f"split_{split}"].eq("test").to_numpy()
    scaler = StandardScaler().fit(df.loc[fit, feats].fillna(0.0))
    prep = lambda idx: scaler.transform(df.loc[idx, feats].fillna(0.0)).astype("float32").reshape(len(idx), len(feats), 1)
    X_fit, X_test = prep(fit), prep(np.flatnonzero(test))
    y_fit, y_test = y_all[fit], y_all[test]
    for name, builder, epochs in RUNS:
        keras.utils.set_random_seed(SEED); t = time.time()
        m = builder(); m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        m.fit(X_fit, y_fit, batch_size=64, epochs=epochs, verbose=0)
        proba = m.predict(X_test, batch_size=2048, verbose=0); pred = proba.argmax(1)
        row = dict(split=split, model=name, train_rows=len(fit), macro_f1=round(f1_score(y_test, pred, average="macro"), 4),
                   accuracy=round(accuracy_score(y_test, pred), 4), minutes=round((time.time() - t) / 60, 1))
        rows.append(row); print(row, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "reports/paper_models_13class.csv", index=False)
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        np.savez_compressed(ROOT / f"data/processed/predictions/13_{split}_{slug}.npz", y=y_test.astype("int8"), pred=pred.astype("int8"),
                            p_attack=(1 - proba[:, 0]).astype("float32"),
                            bucket=(df.loc[test, "behavior_key"].to_numpy().astype("uint64") % N_BUCKETS).astype("int16"))
print("done", flush=True)
