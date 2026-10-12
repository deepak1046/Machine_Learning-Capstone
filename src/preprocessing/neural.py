"""Input preparation for the CNN-BiLSTM-Attention network (v2).

The first version fed the network max-imputed, standard-scaled raw features. Two things hurt it
(see reports/nn_inputs_experiment.csv): heavy-tailed features such as Flow Bytes/s reach about 1e9,
and ports were scaled as if they were magnitudes. v2 therefore (1) applies a signed log1p before
scaling and (2) also hands the two ports to the network as categories for an embedding layer.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.preprocessing.features import MaxImputer, signed_log1p

PORT_COLUMNS = ("Source Port", "Destination Port")
LOW_PORTS = 1024  # ports below this name a service and keep their own bucket
COARSE_BUCKETS = 64  # every higher port falls into one of these 1024-wide buckets
MISSING_BUCKET = LOW_PORTS + COARSE_BUCKETS
N_PORT_BUCKETS = MISSING_BUCKET + 1


def port_buckets(df: pd.DataFrame) -> np.ndarray:
    """(rows, 2) integer bucket per port: exact below 1024, coarse above, one bucket for missing."""
    out = []
    for column in PORT_COLUMNS:
        p = df[column].to_numpy(dtype="float64")
        bucket = np.where(p < LOW_PORTS, p, LOW_PORTS + np.minimum((p - LOW_PORTS) // 1024, COARSE_BUCKETS - 1))
        out.append(np.where(np.isnan(p), MISSING_BUCKET, bucket).astype("int32"))
    return np.stack(out, axis=1)


class NeuralPreprocessor(BaseEstimator, TransformerMixin):
    """max-impute -> signed log1p -> standard scale, plus port buckets.

    Fitted on training rows only. `transform` returns the dict the network expects:
    {"flow": (rows, features, 1) float32, "ports": (rows, 2) int32}.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        self.imputer_ = MaxImputer().fit(X)
        self.scaler_ = StandardScaler().fit(signed_log1p(self.imputer_.transform(X)))
        return self

    def transform(self, X: pd.DataFrame) -> dict:
        X = X[self.columns_]
        flow = self.scaler_.transform(signed_log1p(self.imputer_.transform(X))).astype("float32")
        return {"flow": flow.reshape(len(flow), flow.shape[1], 1), "ports": port_buckets(X)}
