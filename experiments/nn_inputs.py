"""Why does the CNN-BiLSTM-Attention trail XGBoost? Two input-side hypotheses, tested one at a time.

  control   the served recipe: max-impute -> standard scale -> network
  log       signed log1p before scaling (the original recipe feeds heavy-tailed values like Flow Bytes/s ~1e9 to the net)
  ports     control + a port-embedding branch (service ports are categorical, not magnitudes)
  log+ports both

Same group split, same class weights, same early stopping; every variant trains on the SAME 25% sample of
the training rows (the learning curve showed 25% is as good as 100% for the boosted trees) and is scored on the
full, untouched group-split test set.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, keras
from keras import layers
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.models.hybrid import AttentionPooling
from src.preprocessing.features import MaxImputer, signed_log1p

SEED, BATCH, MAX_EPOCHS, PATIENCE, FRACTION = 2026, 1024, 15, 3, 0.25
features = json.loads((ROOT / "reports/final_features.json").read_text())["features"]
classes = list(json.loads((ROOT / "reports/label_encoding.json").read_text()))
data = pd.read_parquet(ROOT / "data/processed/training_sample.parquet", columns=features + ["Label_Encoded", "split_group"])
train = data["split_group"].eq("train").to_numpy(); y_all = data["Label_Encoded"].to_numpy()

rng = np.random.default_rng(SEED)
fit_rows = np.sort(rng.choice(np.flatnonzero(train), size=int(train.sum() * FRACTION), replace=False))
tr_i, va_i = train_test_split(fit_rows, test_size=0.1, stratify=y_all[fit_rows], random_state=SEED)
te_i = np.flatnonzero(~train)
print(f"train {len(tr_i):,} val {len(va_i):,} test {len(te_i):,}", flush=True)

def port_buckets(df):
    """Ports below 1024 keep their own bucket (they name the service); the rest share 64 coarse buckets."""
    out = []
    for col in ("Source Port", "Destination Port"):
        p = df[col].to_numpy(dtype="float64")
        b = np.where(p < 1024, p, 1024 + np.minimum((p - 1024) // 1024, 63))
        out.append(np.where(np.isnan(p), 1088, b).astype("int32"))
    return np.stack(out, axis=1)

def build(ports: bool):
    seq_in = layers.Input(shape=(len(features), 1), name="flow")
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(seq_in); x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = AttentionPooling()(x)
    inputs = [seq_in]
    if ports:
        port_in = layers.Input(shape=(2,), dtype="int32", name="ports")
        e = layers.Embedding(1089, 8)(port_in); e = layers.Flatten()(e)
        x = layers.Concatenate()([x, e]); inputs.append(port_in)
    x = layers.Dense(128, activation="relu")(x); x = layers.Dropout(0.3)(x)
    return keras.Model(inputs, layers.Dense(len(classes), activation="softmax")(x))

results = []
for name, use_log, use_ports in [("control", False, False), ("log", True, False), ("ports", False, True), ("log+ports", True, True)]:
    keras.utils.set_random_seed(SEED)
    t = time.time()
    imputer = MaxImputer().fit(data.loc[tr_i, features])
    def prep(idx, scaler=None):
        X = imputer.transform(data.loc[idx, features])
        if use_log: X = signed_log1p(X)
        return X
    scaler = StandardScaler().fit(prep(tr_i))
    def make(idx):
        X = scaler.transform(prep(idx)).reshape(len(idx), len(features), 1)
        return [X, port_buckets(data.loc[idx])] if use_ports else X
    Xtr, Xva, Xte = make(tr_i), make(va_i), make(te_i)
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_all[tr_i])))
    model = build(use_ports)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(Xtr, y_all[tr_i], validation_data=(Xva, y_all[va_i]), batch_size=BATCH, epochs=MAX_EPOCHS, class_weight=cw, verbose=0,
                     callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)])
    pred = model.predict(Xte, batch_size=BATCH, verbose=0).argmax(1)
    r = dict(variant=name, macro_f1=round(f1_score(y_all[te_i], pred, average="macro"), 4), accuracy=round(accuracy_score(y_all[te_i], pred), 4),
             epochs=len(hist.history["loss"]), minutes=round((time.time() - t) / 60, 1))
    results.append(r); print(r, flush=True)
    pd.DataFrame(results).to_csv(ROOT / "reports/nn_inputs_experiment.csv", index=False)
print("done", flush=True)
