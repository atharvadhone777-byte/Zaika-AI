"""
Train/validation/test split, and the tf.data pipeline that will consume it
during training (added in ml/training/train.py once the model exists).

Split strategy: a plain random split (not stratified, not time-based) is
used deliberately. Two alternatives were considered and rejected:

  - Stratified by cuisine tag: rejected because a recipe can have multiple
    or zero cuisine tags, so "the" stratification key is ambiguous, and the
    retrieval task doesn't depend on cuisine balance to be evaluated fairly.
  - Time-based split (older recipes -> train, newest -> test): this is the
    right call for a system where recipes/trends drift over time (e.g. a
    demand-forecasting model), but our retrieval model's target is a
    largely time-invariant relationship (which ingredients co-occur in
    a coherent recipe), so a time split would only shrink the effective
    training data without buying protection against anything relevant.

A plain random split with a fixed seed is simpler, easier to defend, and
appropriate for what's actually being learned here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

from ml.config import RANDOM_SEED, SPLIT
from ml.data.tokenizer import IngredientVocabulary


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(SPLIT.train_frac + SPLIT.val_frac + SPLIT.test_frac - 1.0) < 1e-9

    rng = np.random.RandomState(RANDOM_SEED)
    shuffled_idx = rng.permutation(len(df))

    n_train = int(len(df) * SPLIT.train_frac)
    n_val = int(len(df) * SPLIT.val_frac)

    train_idx = shuffled_idx[:n_train]
    val_idx = shuffled_idx[n_train:n_train + n_val]
    test_idx = shuffled_idx[n_train + n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Training pair generation for the Siamese retrieval model
# ---------------------------------------------------------------------------

def _pad_or_truncate(ids: list[int], max_len: int) -> list[int]:
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [0] * (max_len - len(ids))  # 0 == <PAD>, see tokenizer.py


def make_training_pairs(
    df: pd.DataFrame,
    vocab: IngredientVocabulary,
    max_len: int,
    min_query_fraction: float = 0.3,
    max_query_fraction: float = 0.8,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds (query, positive) training pairs by simulating "a user who has
    SOME of this recipe's ingredients on hand". For each recipe, the query
    is a random subset (30-80% of its ingredients) and the positive is the
    recipe's full ingredient list.

    Why simulate queries this way rather than using real user query logs:
    this product has no query logs yet - it's pre-launch. Subsampling a
    recipe's own ingredient list is a reasonable proxy for "partial
    ingredient knowledge" because it's drawn from a real, coherent recipe
    (so the subset is a genuinely plausible pantry), and it requires no
    additional labeled data beyond what's already in the recipes table.
    The known limitation, worth stating directly: this only teaches the
    model "ingredients from the same recipe belong together" - it doesn't
    teach substitution relationships (e.g. that butter and margarine play
    similar roles) since those never co-occur as query/positive by this
    construction. That gap is intentionally left to the RAG assistant
    (rag/), which handles substitution via retrieval over a curated
    knowledge base rather than asking the embedding model to learn it
    implicitly - a good example of not forcing one model to do a job a
    different, simpler component is better suited for.
    """
    rng = np.random.RandomState(seed)
    queries, positives = [], []

    for ingredients in df["ingredients"]:
        if len(ingredients) < 2:
            continue
        frac = rng.uniform(min_query_fraction, max_query_fraction)
        k = max(1, int(round(len(ingredients) * frac)))
        query_subset = list(rng.choice(ingredients, size=k, replace=False))

        query_ids = _pad_or_truncate(vocab.encode(query_subset), max_len)
        positive_ids = _pad_or_truncate(vocab.encode(ingredients), max_len)

        queries.append(query_ids)
        positives.append(positive_ids)

    return np.array(queries, dtype="int32"), np.array(positives, dtype="int32")


def make_tf_dataset(
    query_ids: np.ndarray, positive_ids: np.ndarray, batch_size: int, shuffle: bool
) -> tf.data.Dataset:
    """
    Wraps arrays into a tf.data pipeline. drop_remainder=True on the
    training set specifically: the in-batch contrastive loss treats the
    batch size as the number of negative examples, so a smaller final
    batch would silently make the last training step's loss cheaper/easier
    - dropping it keeps every step's difficulty comparable.
    """
    dummy_labels = np.zeros(len(query_ids), dtype="int32")  # unused by the loss, see losses.py
    ds = tf.data.Dataset.from_tensor_slices(
        ({"query_ids": query_ids, "positive_ids": positive_ids}, dummy_labels)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(query_ids), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=shuffle)  # only drop on the (shuffled) train set
    return ds.prefetch(tf.data.AUTOTUNE)
