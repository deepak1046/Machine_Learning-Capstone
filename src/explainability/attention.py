"""Attention weights of the CNN-BiLSTM-Attention network.

AttentionPooling scores every step of the BiLSTM output, softmaxes the scores over the sequence and
takes the weighted sum. The network feeds the 60 flow features in as a sequence of length 60 (no
pooling in the served model), so step i is the position of feature i after the convolutions and the
BiLSTM have mixed in its neighbors. The weights say where the pooling layer looked; they are not a
causal attribution (see shap_global.py for that).
"""

import keras
import numpy as np
import tensorflow as tf

from src.models.hybrid import AttentionPooling


def load_network(path, custom_objects=None) -> keras.Model:
    return keras.saving.load_model(
        path, custom_objects={"AttentionPooling": AttentionPooling, **(custom_objects or {})}, compile=False
    )


def attention_weights(model: keras.Model, X, batch_size: int = 2048) -> np.ndarray:
    """Softmax attention weights, shape (rows, steps).

    X is the preprocessed input: an array (rows, steps, 1) for the v1 network, or the
    {"flow": ..., "ports": ...} dict from NeuralPreprocessor for v2.
    """
    pool = next(layer for layer in model.layers if isinstance(layer, AttentionPooling))
    sequence = keras.Model(model.input, pool.input)  # the BiLSTM output the layer pools over
    n = len(next(iter(X.values())) if isinstance(X, dict) else X)
    out = []
    for start in range(0, n, batch_size):
        batch = {k: v[start : start + batch_size] for k, v in X.items()} if isinstance(X, dict) else X[start : start + batch_size]
        h = sequence(batch, training=False)
        out.append(tf.nn.softmax(pool.score(h), axis=1)[..., 0].numpy())
    return np.concatenate(out)
