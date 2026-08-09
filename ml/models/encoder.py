"""
The retrieval model: a single, weight-shared encoder that maps a set of
ingredients (whether it's a user's pantry or a recipe's full ingredient
list) to a point in a shared embedding space.

Why weight-SHARED rather than two separate towers (a true "dual encoder"):
a classic dual encoder (e.g. DPR for question answering) uses separate
towers because the two sides are different modalities - a short question
vs. a long passage - and forcing them through the same weights would hurt
each side's representation. Here, both sides of the pair are the SAME
modality: an unordered set of ingredient tokens drawn from the same
vocabulary. Sharing weights is not just simpler, it's the more theoretically
correct choice - it forces the model to learn one consistent notion of
"what ingredients go together" rather than learning a query-side quirk and
a document-side quirk that happen to line up. This is exactly the kind of
distinction worth being able to draw in an interview: dual encoders aren't
inherently "two networks," the term describes two INPUT SIDES, and whether
they share weights is a separate, deliberate decision.

Pooling: masked global average pooling over ingredient embeddings, not a
recurrent layer (LSTM/GRU) and not a Transformer. An ingredient list has no
meaningful sequence order (["tomato","onion"] and ["onion","tomato"] are
the same recipe), so a sequence model would be spending parameters modeling
an order that carries no signal, and would need positional encodings that
actively mislead it. Average pooling over a *set* is the representation
that matches what the data actually is.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.saving import register_keras_serializable


# Named, registered functions instead of raw `lambda`s inside Lambda
# layers. Two things break with plain lambdas under Keras 3 + H5:
#   1. Keras 3 refuses to deserialize a Lambda layer whose function is a
#      bare Python lambda by default (arbitrary-code-execution guard).
#   2. Even with that guard overridden, a lambda's closure over module
#      globals (like `tf`) doesn't reliably survive the marshal/unmarshal
#      round trip used to persist it, causing `NameError: name 'tf' is
#      not defined` at load time.
# Registering named functions with `register_keras_serializable` fixes
# both: the Lambda layer stores a resolvable *name*, and the function
# lives in this module's own, always-imported namespace rather than in a
# marshalled closure.
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
    """
    Builds the shared encoder as a plain functional Keras Model (not a
    subclassed model). This is a deliberate export-compatibility choice:
    subclassed models can lose the ability to save cleanly to both
    SavedModel and H5 format, and the assignment requires both. A
    functional model with a fixed input signature saves reliably to
    either format and can be loaded and served directly without needing
    the original Python class definition available at load time.
    """
    ingredient_ids = layers.Input(shape=(max_len,), dtype="int32", name="ingredient_ids")

    # mask_zero=False here deliberately: Keras's automatic mask propagation
    # (mask_zero=True) inserts an internal mask-computation op into the
    # graph that the LEGACY H5 saver cannot always reliably deserialize
    # (it isn't a registered layer, so `load_model` fails with "Unknown
    # layer" on reload). Masking is instead computed explicitly below as
    # ordinary Keras layers, which is a few more lines but serializes
    # cleanly to both SavedModel and H5 - a worthwhile trade given the
    # assignment explicitly requires both export formats to work.
    embedded = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=False,
        name="ingredient_embedding",
    )(ingredient_ids)

    # Explicit mask: 1.0 where a real ingredient id is present, 0.0 at
    # <PAD> positions (id 0, see tokenizer.py). Building this as ordinary
    # layers (Lambda + Multiply), rather than relying on layer-level
    # masking, is what keeps the graph H5-serializable.
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

    # Final projection back to embedding_dim, then L2-normalize. Normalizing
    # is what makes dot product == cosine similarity, which is what both
    # the training loss and the FAISS IndexFlatIP at inference time assume -
    # keeping this consistent between training and serving avoids a subtle
    # train/serve skew bug.
    x = layers.Dense(embedding_dim, name="embedding_projection")(x)
    # UnitNormalization is a native Keras layer (unlike a Lambda wrapping
    # tf.math.l2_normalize), which is what makes this step reload cleanly
    # from H5 without needing a custom_object_scope at load time.
    outputs = layers.UnitNormalization(axis=-1, name="l2_normalize")(x)

    return Model(inputs=ingredient_ids, outputs=outputs, name=name)


def build_siamese_training_model(encoder: Model, max_len: int) -> Model:
    """
    Wraps the shared encoder in a two-input training model. Both inputs go
    through the SAME encoder instance (that's what makes the weights
    shared), and the two resulting embeddings are stacked into a single
    (batch, 2, embedding_dim) output so a standard Keras loss function -
    which receives the full batch of y_pred - can compute the in-batch
    contrastive loss across ALL pairs in the batch, not just within a
    single (query, positive) pair. See ml/training/losses.py.
    """
    query_ids = layers.Input(shape=(max_len,), dtype="int32", name="query_ids")
    positive_ids = layers.Input(shape=(max_len,), dtype="int32", name="positive_ids")

    query_embedding = encoder(query_ids)
    positive_embedding = encoder(positive_ids)

    stacked = layers.Lambda(_stack_embeddings_fn, name="stack_embeddings")(
        [query_embedding, positive_embedding]
    )

    return Model(inputs=[query_ids, positive_ids], outputs=stacked, name="siamese_trainer")
