"""Web API behind the DDoS flow classifier demo.

Serves the trained models (classify a held-out sample or an uploaded CSV, explain one flow)
and the project's consolidated results. In production the built Vite frontend in
app/web/dist is served from the same process.

Run from the project root:  python -m uvicorn app.api.main:app --port 8000
"""

import io
import math
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.explainability.contributions import tree_contributions
from src.models.predictor import (
    FEATURES, MODEL_SPECS, _models, classify, load_model, prepare_features, score_against_truth, truth_labels,
)
from src.preprocessing.labels import CLASSES, FAMILY_MAP

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SAMPLE_PATH = ROOT / "app" / "data" / "sample_flows.csv"
WEB_DIST = ROOT / "app" / "web" / "dist"

MAX_UPLOAD_ROWS = 100_000
MAX_FLOWS_RETURNED = 500
TOP_ALTERNATIVES = 3
TOP_FEATURES = 12

MODEL_BLURBS = {
    "xgb13": "300 gradient-boosted trees. The strongest model on the full 13-class task.",
    "xgb8": "The same trees, with DNS/LDAP/SNMP, NetBIOS/Portmap and SSDP/UDP/UDPLag merged into families.",
    "hybrid13": "The proposed network: 1-D convolutions, a bidirectional LSTM and attention pooling.",
}
SUMMARY_ROW = {"xgb13": "XGBoost", "hybrid13": "CNN-BiLSTM-Attention (un-pooled)"}


def _warm_models() -> None:
    for key in MODEL_SPECS:
        try:
            load_model(key)
        except Exception as exc:  # a missing artifact must not stop the API from starting
            print(f"[warm-up] could not load {key}: {exc}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm_models, daemon=True).start()
    yield


app = FastAPI(title="DDoS flow classifier", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _records(df: pd.DataFrame) -> list:
    """DataFrame -> JSON-safe records (NaN -> null)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _spec(key: str):
    if key not in MODEL_SPECS:
        raise HTTPException(404, f"unknown model '{key}'")
    return MODEL_SPECS[key]


@lru_cache
def _sample_frame() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


@lru_cache
def _heldout_scores() -> dict:
    """Group-split test scores and ceilings per served model, read from the saved reports."""
    thirteen = pd.read_csv(REPORTS / "final_summary_13class.csv", index_col=0)
    family = pd.read_csv(REPORTS / "final_summary_family.csv", index_col=0)
    ceilings = pd.read_csv(REPORTS / "ceiling_summary.csv", index_col=0)

    scores = {}
    for key, row in SUMMARY_ROW.items():
        ceiling = ceilings.loc["13-class (attack type)"]
        scores[key] = {
            "macro_f1": _finite(thirteen.loc[row, "macro-F1 | group"]),
            "accuracy": _finite(thirteen.loc[row, "accuracy | group"]),
            "ceiling_macro_f1": _finite(ceiling["macro_f1"]),
            "ceiling_accuracy": _finite(ceiling["accuracy"]),
        }
    ceiling = ceilings.loc["8-class (attack family)"]
    scores["xgb8"] = {
        "macro_f1": _finite(family.loc["group", "macro-F1"]),
        "accuracy": _finite(family.loc["group", "accuracy"]),
        "ceiling_macro_f1": _finite(ceiling["macro_f1"]),
        "ceiling_accuracy": _finite(ceiling["accuracy"]),
    }
    return scores


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": sorted(_models)}


@app.get("/api/families")
def families():
    """Attack types that the family scheme merges, keyed by family."""
    groups: dict = {}
    for attack, family in FAMILY_MAP.items():
        groups.setdefault(family, []).append(attack)
    return {family: sorted(members) for family, members in groups.items() if len(members) > 1}


@app.get("/api/models")
def models():
    scores = _heldout_scores()
    return [
        {
            "key": spec.key,
            "title": spec.title,
            "kind": spec.kind,
            "level": spec.level,
            "classes": list(spec.classes),
            "blurb": MODEL_BLURBS[spec.key],
            "explainable": spec.kind == "tree",
            "loaded": spec.key in _models,
            "heldout": scores[spec.key],
        }
        for spec in MODEL_SPECS.values()
    ]


def _classification_payload(model_key: str, df: pd.DataFrame, source: str, truncated: bool) -> dict:
    spec = _spec(model_key)
    started = time.perf_counter()
    try:
        X = prepare_features(df)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    result = classify(load_model(model_key), X)
    classes = list(spec.classes)

    truth = truth_labels(df, spec.level)
    scored = score_against_truth(truth, result.predicted, classes) if truth is not None else None

    counts = pd.Series(result.predicted).value_counts()
    n_out = min(len(X), MAX_FLOWS_RETURNED)
    values = X.iloc[:n_out].to_numpy(dtype="float64")
    top = np.argsort(-result.proba[:n_out], axis=1)[:, :TOP_ALTERNATIVES]

    flows = []
    for i in range(n_out):
        truth_i = truth.iloc[i] if truth is not None else None
        flows.append({
            "id": i,
            "prediction": str(result.predicted[i]),
            "confidence": float(result.confidence[i]),
            "truth": truth_i if isinstance(truth_i, str) else None,
            "top": [{"class": classes[j], "p": float(result.proba[i, j])} for j in top[i]],
            "x": [float(v) if math.isfinite(v) else None for v in values[i]],
        })

    return {
        "model": model_key,
        "level": spec.level,
        "source": source,
        "classes": classes,
        "n_flows": len(X),
        "truncated": truncated,
        "prediction_counts": [{"class": str(c), "count": int(n)} for c, n in counts.items()],
        "attack_share": float((result.predicted != "BENIGN").mean()),
        "mean_confidence": float(result.confidence.mean()),
        "truth": scored,
        "feature_names": FEATURES,
        "flows": flows,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }


class SampleRequest(BaseModel):
    model: str
    n: int = Field(500, ge=10, le=2000)
    seed: int = 0
    classes: list[str] | None = None


@app.post("/api/classify/sample")
def classify_sample(req: SampleRequest):
    _spec(req.model)
    pool = _sample_frame()
    wanted = set(req.classes) if req.classes else set(CLASSES)
    pool = pool[pool["Label"].isin(wanted)]
    if pool.empty:
        raise HTTPException(422, "no sample flows match the selected attack types")
    drawn = pool.sample(n=min(req.n, len(pool)), random_state=req.seed).reset_index(drop=True)
    return _classification_payload(req.model, drawn, "sample", truncated=False)


@app.post("/api/classify/upload")
def classify_upload(model: str = Form(...), file: UploadFile = File(...)):
    _spec(model)
    wanted = set(FEATURES) | {"Label"}
    try:
        df = pd.read_csv(
            io.BytesIO(file.file.read()), nrows=MAX_UPLOAD_ROWS + 1,
            usecols=lambda column: str(column).strip() in wanted, low_memory=False,
        )
    except Exception as exc:
        raise HTTPException(422, f"could not read the file as a CICFlowMeter CSV: {exc}") from exc
    if df.empty:
        raise HTTPException(422, "the file has no rows or none of the expected columns")

    truncated = len(df) > MAX_UPLOAD_ROWS
    return _classification_payload(model, df.iloc[:MAX_UPLOAD_ROWS], "upload", truncated)


class ExplainRequest(BaseModel):
    model: str
    x: list[float | None]


@app.post("/api/explain")
def explain(req: ExplainRequest):
    spec = _spec(req.model)
    if spec.kind != "tree":
        raise HTTPException(400, "per-flow explanations are available for the XGBoost models only")
    if len(req.x) != len(FEATURES):
        raise HTTPException(422, f"expected {len(FEATURES)} feature values, got {len(req.x)}")

    X = pd.DataFrame([[np.nan if v is None else v for v in req.x]], columns=FEATURES, dtype="float64")
    model = load_model(req.model)
    proba = model.predict_proba(X)[0]
    predicted = int(proba.argmax())

    contributions, bias = tree_contributions(model.estimator, X, predicted)
    scores = contributions.iloc[0]
    strongest = scores.reindex(scores.abs().sort_values(ascending=False).index)[:TOP_FEATURES]

    return {
        "predicted": spec.classes[predicted],
        "confidence": float(proba[predicted]),
        "alternatives": [
            {"class": spec.classes[j], "p": float(proba[j])} for j in np.argsort(-proba)[:TOP_ALTERNATIVES]
        ],
        "contributions": [
            {"feature": name, "value": _finite(X.iloc[0][name]), "contribution": float(score)}
            for name, score in strongest.items()
        ],
        "other_features": float(scores.sum() - strongest.sum()),
        "bias": float(bias[0]),
    }


class TickRequest(BaseModel):
    model: str
    n: int = Field(30, ge=1, le=300)
    seed: int = 0
    attack_share: float = Field(0.0, ge=0.0, le=1.0)
    vectors: list[str] | None = None


@app.post("/api/stream/tick")
def stream_tick(req: TickRequest):
    """One second of replayed traffic for the live monitor.

    Draws held-out flows (benign plus, at `attack_share`, the chosen attack vectors), scores them
    with the real model and returns the verdicts with the flow features worth showing. It is a
    replay of labeled dataset flows, not a capture of live traffic.
    """
    spec = _spec(req.model)
    pool = _sample_frame()
    rng = np.random.default_rng(req.seed)

    attacks = pool[pool["Label"].isin([v for v in (req.vectors or CLASSES) if v != "BENIGN"])]
    n_attack = int(rng.binomial(req.n, req.attack_share)) if len(attacks) else 0
    parts = [pool[pool["Label"] == "BENIGN"].sample(req.n - n_attack, replace=True, random_state=int(rng.integers(2**31)))]
    if n_attack:
        parts.append(attacks.sample(n_attack, replace=True, random_state=int(rng.integers(2**31))))
    drawn = pd.concat(parts).sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(drop=True)

    started = time.perf_counter()
    X = prepare_features(drawn)
    result = classify(load_model(req.model), X)
    truth = truth_labels(drawn, spec.level)

    flows = [
        {
            "prediction": str(result.predicted[i]),
            "confidence": float(result.confidence[i]),
            "truth": truth.iloc[i] if truth is not None and isinstance(truth.iloc[i], str) else None,
            "protocol": _finite(X["Protocol"].iloc[i]),
            "packets_per_s": _finite(X["Flow Packets/s"].iloc[i]),
            "bytes_per_s": _finite(X["Flow Bytes/s"].iloc[i]),
            "duration_us": _finite(X["Flow Duration"].iloc[i]),
        }
        for i in range(len(X))
    ]
    return {
        "model": req.model,
        "level": spec.level,
        "n_flows": len(X),
        "flows": flows,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }


@app.get("/api/results")
def results():
    thirteen = pd.read_csv(REPORTS / "final_summary_13class.csv", index_col=0)
    thirteen.columns = [
        c.replace(" | ", "_").replace("% of ceiling (macro-F1)", "pct_ceiling").replace("macro-F1", "macro_f1")
        .replace("train+predict minutes", "minutes") for c in thirteen.columns
    ]
    family = pd.read_csv(REPORTS / "final_summary_family.csv")
    family.columns = ["split", "macro_f1", "accuracy", "ceiling_macro_f1", "ceiling_accuracy", "vs_ceiling"]
    base = pd.read_csv(REPORTS / "final_summary_base_paper.csv")
    base.columns = ["split", "model", "accuracy", "macro_f1", "ceiling_accuracy", "ceiling_macro_f1"]

    def per_class(path: str, rename: dict | None = None) -> dict:
        table = pd.read_csv(REPORTS / path, index_col=0).rename(columns=rename or {})
        return {
            "classes": list(table.index),
            "columns": list(table.columns),
            "values": [[_finite(v) for v in row] for row in table.to_numpy()],
        }

    return {
        "ceilings": _records(pd.read_csv(REPORTS / "ceiling_summary.csv")),
        "thirteen_class": _records(thirteen.reset_index(names="model")),
        "family": _records(family),
        "base_paper": _records(base),
        "per_class_13": per_class("final_summary_per_class_13class.csv"),
        "per_class_family": per_class(
            "family_baseline_per_class_f1.csv",
            {
                "ceiling": "Ceiling | (no ports)",
                "XGBoost (family, random)": "XGBoost | random",
                "XGBoost (family, group)": "XGBoost | group",
            },
        ),
    }


@app.get("/api/sample/download")
def download_sample():
    return FileResponse(SAMPLE_PATH, media_type="text/csv", filename="sample_flows.csv")


if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
