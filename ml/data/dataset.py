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

def _pad_or_truncate(ids: list[int], max_len: int) -> list[int]:
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [0] * (max_len - len(ids)) 

def make_training_pairs(
    df: pd.DataFrame,
    vocab: IngredientVocabulary,
    max_len: int,
    min_query_fraction: float = 0.3,
    max_query_fraction: float = 0.8,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:

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
    dummy_labels = np.zeros(len(query_ids), dtype="int32")  
    ds = tf.data.Dataset.from_tensor_slices(
        ({"query_ids": query_ids, "positive_ids": positive_ids}, dummy_labels)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(query_ids), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=shuffle) 
    return ds.prefetch(tf.data.AUTOTUNE)
