"""Inference for the trained flow classifiers.

Mirrors the training pipeline exactly: raw CICFlowMeter columns -> the 60 selected
features -> (neural net only) the train-fitted preprocessor -> model -> per-class scores.
All served models were trained on the group split (behaviorally identical flows kept on one
side), so held-out flows are genuinely unseen behavior.
"""

import json
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.preprocessing.labels import CLASSES, FAMILY_CLASSES, FAMILY_MAP, LABEL_MAP

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "saved_models"
FEATURES = json.loads((ROOT / "reports" / "final_features.json").read_text())["features"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    kind: str  # "tree" or "neural"
    level: str  # "attack type" or "attack family"
    file: str
    classes: tuple
    preprocessor: str | None = None
    # per-class log-probability offsets (JSON list in class order), tuned for macro-F1 on a validation
    # slice of the training data; shifts the decision rule, not the trees
    offsets: str | None = None


MODEL_SPECS = {
    "xgb13": ModelSpec("xgb13", "XGBoost", "tree", "attack type", "XGBoost_group.joblib", tuple(CLASSES)),
    "xgb8": ModelSpec(
        "xgb8", "XGBoost", "tree", "attack family", "XGBoost_family_group_v2.joblib", tuple(FAMILY_CLASSES),
        offsets="offsets_xgb8_group.json",
    ),
    "hybrid13": ModelSpec(
        "hybrid13", "CNN-BiLSTM-Attention", "neural", "attack type", "cnn_bilstm_attention_v2_group.keras",
        tuple(CLASSES), preprocessor="hybrid_v2_preprocessor_group.joblib", offsets="offsets_hybrid13_group.json",
    ),
}


class FlowModel:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.classes = list(spec.classes)
        self.offsets = (
            np.asarray(json.loads((MODELS_DIR / spec.offsets).read_text()), dtype="float64") if spec.offsets else None
        )
        path = MODELS_DIR / spec.file
        if spec.kind == "tree":
            self.estimator = joblib.load(path)
        else:
            import keras  # heavy import, only paid when the neural model is actually requested

            from src.models.hybrid import AttentionPooling

            self.estimator = keras.saving.load_model(
                path, custom_objects={"AttentionPooling": AttentionPooling}, compile=False
            )
            self.preprocessor = joblib.load(MODELS_DIR / spec.preprocessor)
            self._lock = threading.Lock()

    def predict_proba(self, X: pd.DataFrame, batch_size: int = 2048) -> np.ndarray:
        proba = self._raw_proba(X, batch_size)
        if self.offsets is None:
            return proba
        shifted = np.log(np.clip(proba, 1e-9, 1.0)) + self.offsets
        shifted = np.exp(shifted - shifted.max(axis=1, keepdims=True))
        return shifted / shifted.sum(axis=1, keepdims=True)

    def _raw_proba(self, X: pd.DataFrame, batch_size: int = 2048) -> np.ndarray:
        if self.spec.kind == "tree":
            return self.estimator.predict_proba(X)
        Z = self.preprocessor.transform(X)
        if not isinstance(Z, dict):  # v1 network: one scaled array, read as a sequence of features
            Z = Z.reshape(len(Z), Z.shape[1], 1)
        with self._lock:  # v2 network: {"flow": ..., "ports": ...} from src.preprocessing.neural
            return self.estimator.predict(Z, batch_size=batch_size, verbose=0)


_models: dict = {}
_load_locks: defaultdict = defaultdict(threading.Lock)


def load_model(key: str) -> FlowModel:
    if key not in MODEL_SPECS:
        raise KeyError(f"unknown model '{key}', expected one of {sorted(MODEL_SPECS)}")
    with _load_locks[key]:
        if key not in _models:
            _models[key] = FlowModel(MODEL_SPECS[key])
        return _models[key]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw CICFlowMeter / CICDDoS2019 frame -> the model's 60 features, in training order."""
    df = df.rename(columns=lambda c: str(c).strip())
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:6]) + (f", ... (+{len(missing) - 6} more)" if len(missing) > 6 else "")
        raise ValueError(f"missing {len(missing)} required column(s): {shown}")
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").astype("float64")
    return X.replace([np.inf, -np.inf], np.nan)


@dataclass
class Classification:
    classes: list
    proba: np.ndarray
    predicted: np.ndarray
    confidence: np.ndarray


def classify(model: FlowModel, X: pd.DataFrame) -> Classification:
    proba = np.asarray(model.predict_proba(X), dtype="float64")
    top = proba.argmax(axis=1)
    return Classification(model.classes, proba, np.asarray(model.classes)[top], proba[np.arange(len(top)), top])


def standardize_label(raw) -> str | None:
    """Raw or already-canonical label -> canonical 13-class name; None if not one of the 13."""
    s = str(raw).strip()
    return LABEL_MAP.get(s, s if s in CLASSES else None)


def truth_labels(df: pd.DataFrame, level: str) -> pd.Series | None:
    """True labels at the requested level, or None if the frame has no Label column.
    Labels outside the 13 classes (e.g. WebDDoS) come back as None."""
    label_col = next((c for c in df.columns if str(c).strip() == "Label"), None)
    if label_col is None:
        return None
    canonical = df[label_col].map(standardize_label)
    if level == "attack family":
        canonical = canonical.map(lambda c: FAMILY_MAP.get(c) if c is not None else None)
    return canonical


def score_against_truth(truth: pd.Series, predicted: np.ndarray, classes: list) -> dict | None:
    """Accuracy, macro-F1 over the classes that actually occur in the truth, per-class
    precision/recall/F1 and a count confusion matrix. Rows with unknown labels are skipped."""
    known = truth.notna().to_numpy()
    if not known.any():
        return None
    y_true, y_pred = truth.to_numpy()[known], np.asarray(predicted)[known]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    present = support > 0
    return {
        "n_labeled": int(known.sum()),
        "n_skipped": int((~known).sum()),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1[present].mean()),
        "per_class": [
            {"class": c, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for c, p, r, f, s in zip(classes, precision, recall, f1, support)
        ],
        "confusion": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }
