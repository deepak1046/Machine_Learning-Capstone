import keras
import tensorflow as tf
from keras import layers


class AttentionPooling(layers.Layer):
    """Additive attention pooling: learns a per-timestep importance score and
    collapses the sequence to a single weighted context vector."""

    def build(self, input_shape):
        self.score = layers.Dense(1, activation="tanh")
        super().build(input_shape)

    def call(self, x):
        scores = self.score(x)  # (batch, steps, 1)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(x * weights, axis=1)  # (batch, channels)


def build_cnn_bilstm_attention(n_features: int, n_classes: int, pool: bool = False) -> keras.Model:
    """CNN front-end (local feature smoothing) feeding a BiLSTM (sequential
    dependencies across the feature sequence) with additive attention pooling,
    then a classification head.

    `pool=False` (default) keeps the full n_features-length sequence into the
    BiLSTM. An earlier version pooled 60 features down to 15 "steps" to keep
    CPU training fast, but that likely discarded fine-grained signal the
    confusable classes (NetBIOS/Portmap, DNS/LDAP/SNMP) depend on -- those are
    exactly the classes where this model underperformed XGBoost. `pool=True`
    restores the old 4x-downsampled behavior for comparison.
    """
    inputs = layers.Input(shape=(n_features, 1))

    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    if pool:
        x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    if pool:
        x = layers.MaxPooling1D(2)(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = AttentionPooling()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="cnn_bilstm_attention")
