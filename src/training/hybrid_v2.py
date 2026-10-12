"""Train and evaluate CNN-BiLSTM-Attention v2 (log-scaled inputs + port embedding) on the persisted splits.

Same data, features, splits, class weights and early stopping as notebook 05; the only changes are the
input handling in src/preprocessing/neural.py and the port-embedding branch (see
reports/nn_inputs_experiment.csv for the controlled experiment that motivated them).

Run from the project root:  python -m src.training.hybrid_v2 group random
Writes per split: saved_models/cnn_bilstm_attention_v2_<split>.keras,
saved_models/hybrid_v2_preprocessor_<split>.joblib, reports/hybrid_v2_results.csv,
reports/hybrid_v2_per_class_f1.csv, reports/hybrid_v2_history_<split>.json.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.evaluation.metrics import evaluate
from src.models.hybrid import build_cnn_bilstm_attention
from src.preprocessing.neural import NeuralPreprocessor

ROOT = Path(__file__).resolve().parents[2]
SEED, BATCH_SIZE, MAX_EPOCHS, PATIENCE = 2026, 1024, 30, 4


def train_split(split_name: str) -> dict:
    features = json.loads((ROOT / "reports/final_features.json").read_text())["features"]
    encoding = json.loads((ROOT / "reports/label_encoding.json").read_text())
    class_names = sorted(encoding, key=encoding.get)
    data = pd.read_parquet(
        ROOT / "data/processed/training_sample.parquet", columns=features + ["Label_Encoded", f"split_{split_name}"]
    )
    train = data[f"split_{split_name}"].eq("train").to_numpy()
    y = data["Label_Encoded"].to_numpy()

    keras.utils.set_random_seed(SEED)
    # fit the preprocessor on the training rows only; early stopping uses a stratified slice of them,
    # and the test rows are never touched until the final prediction
    train_idx, val_idx = train_test_split(np.flatnonzero(train), test_size=0.1, stratify=y[train], random_state=SEED)
    preprocessor = NeuralPreprocessor().fit(data.loc[train_idx, features])
    X_tr, X_val = preprocessor.transform(data.loc[train_idx, features]), preprocessor.transform(data.loc[val_idx, features])
    y_tr, y_val = y[train_idx], y[val_idx]
    test_idx = np.flatnonzero(~train)
    X_test, y_test = preprocessor.transform(data.loc[test_idx, features]), y[test_idx]
    print(f"[{split_name}] train {len(y_tr):,}  val {len(y_val):,}  test {len(y_test):,}", flush=True)

    class_weight = dict(enumerate(compute_class_weight("balanced", classes=np.arange(len(class_names)), y=y_tr)))
    model = build_cnn_bilstm_attention(len(features), len(class_names), port_embedding=True)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    started = time.time()
    history = model.fit(
        X_tr, y_tr, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=MAX_EPOCHS,
        class_weight=class_weight, verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        ],
    )
    elapsed = time.time() - started

    y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)
    summary, per_class = evaluate(y_test, y_pred, class_names)
    epochs = len(history.history["loss"])
    print(f"[{split_name}] macro-F1 {summary['macro_f1']:.4f}  accuracy {summary['accuracy']:.4f}  "
          f"{epochs} epochs  {elapsed / 60:.0f} min", flush=True)

    model.save(ROOT / f"saved_models/cnn_bilstm_attention_v2_{split_name}.keras")
    joblib.dump(preprocessor, ROOT / f"saved_models/hybrid_v2_preprocessor_{split_name}.joblib")
    (ROOT / f"reports/hybrid_v2_history_{split_name}.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()})
    )
    np.save(ROOT / f"reports/hybrid_v2_predictions_{split_name}.npy", np.stack([y_test, y_pred]).astype("int16"))

    # merge this split into the shared result tables
    results_path, per_class_path = ROOT / "reports/hybrid_v2_results.csv", ROOT / "reports/hybrid_v2_per_class_f1.csv"
    row = pd.DataFrame([{"split": split_name, "model": "CNN-BiLSTM-Attention (log + ports)", **summary,
                         "epochs": epochs, "seconds": round(elapsed, 1)}])
    if results_path.exists():
        table = pd.read_csv(results_path)
        row = pd.concat([table[table["split"] != split_name], row], ignore_index=True)
    row.to_csv(results_path, index=False)
    per_class_table = pd.read_csv(per_class_path, index_col=0) if per_class_path.exists() else pd.DataFrame(index=class_names)
    per_class_table[f"CNN-BiLSTM-Attention v2 ({split_name})"] = per_class["f1"].round(3)
    per_class_table.to_csv(per_class_path)
    return summary


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    for name in sys.argv[1:] or ["group"]:
        train_split(name)
