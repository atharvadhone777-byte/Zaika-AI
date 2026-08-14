from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="recipe_ai")
def _pad_mask_fn(t):
    return tf.cast(tf.not_equal(t, 0), tf.float32)[..., tf.newaxis]


@register_keras_serializable(package="recipe_ai")
def _sum_over_time_fn(t):
    return tf.reduce_sum(t, axis=1)


@register_keras_serializable(package="recipe_ai")
def _count_real_ingredients_fn(t):
    return tf.reduce_sum(t, axis=1) + 1e-8


@register_keras_serializable(package="recipe_ai")
def _safe_divide_fn(t):
    return t[0] / t[1]


@register_keras_serializable(package="recipe_ai")
def _stack_embeddings_fn(t):
    return tf.stack(t, axis=1)


def build_encoder(
    vocab_size: int,
    embedding_dim: int = 128,
    max_len: int = 20,
    hidden_units: tuple[int, ...] = (256, 128),
    dropout_rate: float = 0.2,
    name: str = "ingredient_encoder",
) -> Model:

    ingredient_ids = layers.Input(shape=(max_len,), dtype="int32", name="ingredient_ids")
    embedded = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=False,
        name="ingredient_embedding",
    )(ingredient_ids)

    mask = layers.Lambda(_pad_mask_fn, output_shape=(max_len, 1), name="pad_mask")(ingredient_ids)

    masked_embedded = layers.Multiply(name="apply_mask")([embedded, mask])
    summed = layers.Lambda(_sum_over_time_fn, output_shape=(embedding_dim,), name="sum_embeddings")(masked_embedded)
    ingredient_counts = layers.Lambda(
        _count_real_ingredients_fn, output_shape=(1,), name="count_real_ingredients"
    )(mask)
    x = layers.Lambda(_safe_divide_fn, output_shape=(embedding_dim,), name="masked_mean_pool")(
        [summed, ingredient_counts]
    )

    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)

    x = layers.Dense(embedding_dim, name="embedding_projection")(x)

    outputs = layers.UnitNormalization(axis=-1, name="l2_normalize")(x)

    return Model(inputs=ingredient_ids, outputs=outputs, name=name)


def build_siamese_training_model(encoder: Model, max_len: int) -> Model:
    query_ids = layers.Input(shape=(max_len,), dtype="int32", name="query_ids")
    positive_ids = layers.Input(shape=(max_len,), dtype="int32", name="positive_ids")

    query_embedding = encoder(query_ids)
    positive_embedding = encoder(positive_ids)

    stacked = layers.Lambda(_stack_embeddings_fn, name="stack_embeddings")(
        [query_embedding, positive_embedding]
    )

    return Model(inputs=[query_ids, positive_ids], outputs=stacked, name="siamese_trainer")
