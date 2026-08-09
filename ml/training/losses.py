"""
In-batch contrastive loss (InfoNCE, the same family used by CLIP and
SimCLR) for training the shared ingredient encoder.

Why contrastive learning rather than a supervised classification loss:
there's no fixed label set to classify into - "similar to which recipe"
is a relation between examples, not a category. Contrastive learning
turns every OTHER item in the same training batch into a free negative
example, without needing separately mined negative pairs. That's what
makes it practical here: with a few hundred (and eventually ~150K)
recipes, explicitly mining hard negatives would be its own subproject,
while in-batch negatives fall out of ordinary batching for free and get
harder (more informative) automatically as the embedding space improves
during training.
"""

from __future__ import annotations

import tensorflow as tf


def make_in_batch_infonce_loss(temperature: float = 0.07):
    """
    Returns a Keras-compatible loss fn(y_true, y_pred).

    y_pred: (batch, 2, embedding_dim) - index 0 is the query embedding,
    index 1 is the positive (full-recipe) embedding, as produced by
    build_siamese_training_model. y_true is unused (a dummy placeholder is
    passed at training time) because the "label" for each row is implicitly
    its own position in the batch - see `labels` below.

    Symmetric loss: computed once treating query->positive as the
    classification problem and once treating positive->query, then
    averaged. This is standard practice (CLIP does the same) because a
    query should retrieve its correct recipe AND a recipe should retrieve
    a query consistent with it - optimizing only one direction leaves the
    embedding space asymmetric in a way that hurts real retrieval, where
    only the query->recipe direction is used at serving time but the
    representation quality benefits measurably from the symmetric signal.
    """

    def loss_fn(y_true, y_pred):
        query_embeddings = y_pred[:, 0, :]
        positive_embeddings = y_pred[:, 1, :]

        # Embeddings are already L2-normalized by the encoder, so this dot
        # product is cosine similarity in [-1, 1]. Dividing by temperature
        # sharpens the softmax - a lower temperature makes the model more
        # confident/punishing about near-miss negatives, which matters more
        # as the effective vocabulary of "plausible" recipes grows.
        logits = tf.matmul(query_embeddings, positive_embeddings, transpose_b=True) / temperature

        batch_size = tf.shape(logits)[0]
        labels = tf.range(batch_size)  # row i's positive is column i, by construction of the batch

        loss_q_to_p = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
        loss_p_to_q = tf.keras.losses.sparse_categorical_crossentropy(
            labels, tf.transpose(logits), from_logits=True
        )
        return (loss_q_to_p + loss_p_to_q) / 2.0

    return loss_fn
