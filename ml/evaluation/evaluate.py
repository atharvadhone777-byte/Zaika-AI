"""
Evaluates the trained encoder on the held-out test split.

Corpus definition: the retrieval corpus is ALL recipes (train + val + test)
- at serving time, a real user's pantry should be able to retrieve ANY
recipe in the system, not just ones held out for testing. What's held out
for evaluation is the QUERIES: test-set recipes' ingredient subsets are
used as queries, and we check whether retrieval finds their true source
recipe in the full corpus. This mirrors how the system is actually used
in production and avoids the common mistake of evaluating retrieval
against an artificially shrunk corpus that makes the task easier than
it will be at serving time.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import tensorflow as tf

from ml.config import (
    CLEANED_RECIPES_PATH, TEST_SPLIT_PATH, VOCAB_PATH, MODELS_DIR, DOCS_DIR, MODEL,
)
from ml.data.dataset import make_training_pairs
from ml.data.tokenizer import IngredientVocabulary
from ml.evaluation.metrics import evaluate_retrieval
import ml.models.encoder  # noqa: F401 - import triggers @register_keras_serializable, required before load_model


def _load_parquet_with_ingredients(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["ingredients"] = df["ingredients"].apply(json.loads)
    return df


def _encode_all(encoder: tf.keras.Model, ingredient_lists: list[list[str]], vocab: IngredientVocabulary, max_len: int) -> np.ndarray:
    from ml.data.dataset import _pad_or_truncate
    ids = np.array([_pad_or_truncate(vocab.encode(ings), max_len) for ings in ingredient_lists], dtype="int32")
    return encoder.predict(ids, verbose=0)


def main():
    vocab = IngredientVocabulary.load(VOCAB_PATH)
    max_len = MODEL.max_ingredients_per_recipe

    # safe_mode=False: this H5 file contains Lambda layers (the explicit
    # masking math in ml/models/encoder.py), which Keras 3 refuses to
    # deserialize by default since a Lambda layer's Python function could
    # in principle contain arbitrary code. That's the right default for
    # loading models from an untrusted source; it's safe to override here
    # because this file is one we just trained ourselves in this same
    # pipeline, not a third-party artifact.
    encoder = tf.keras.models.load_model(str(MODELS_DIR / "encoder.h5"), safe_mode=False)

    corpus_df = _load_parquet_with_ingredients(CLEANED_RECIPES_PATH)
    test_df = _load_parquet_with_ingredients(TEST_SPLIT_PATH)

    print(f"corpus size: {len(corpus_df)}, test queries drawn from: {len(test_df)} recipes")

    corpus_embeddings = _encode_all(encoder, corpus_df["ingredients"].tolist(), vocab, max_len)

    # id -> position in the corpus array, needed to translate "this query's
    # true recipe id" into "which row of corpus_embeddings is relevant".
    id_to_corpus_idx = {rid: i for i, rid in enumerate(corpus_df["id"].tolist())}

    query_ids, _ = make_training_pairs(test_df, vocab, max_len, seed=123)  # different seed than training, fresh subsets
    # make_training_pairs drops recipes with <2 ingredients, so re-derive
    # the matching id list the same way to keep queries and relevant_indices aligned.
    valid_test_df = test_df[test_df["ingredients"].apply(len) >= 2].reset_index(drop=True)
    relevant_indices = [id_to_corpus_idx[rid] for rid in valid_test_df["id"].tolist()]

    query_embeddings = encoder.predict(query_ids, verbose=0)

    metrics = evaluate_retrieval(query_embeddings, corpus_embeddings, relevant_indices)

    print("\n=== Retrieval evaluation on test split ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DOCS_DIR / "evaluation_report.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {DOCS_DIR / 'evaluation_report.json'}")

    # A handful of qualitative examples - what a reviewer / interviewer
    # will actually want to see, not just the aggregate numbers.
    print("\n=== Qualitative examples ===")
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(len(valid_test_df), size=min(3, len(valid_test_df)), replace=False)
    similarity = query_embeddings @ corpus_embeddings.T
    for i in sample_idx:
        query_recipe = valid_test_df.iloc[i]
        top3 = np.argsort(-similarity[i])[:3]
        print(f"\nQuery (subset of): {query_recipe['name']!r} — true ingredients: {query_recipe['ingredients']}")
        for rank, corpus_i in enumerate(top3, start=1):
            row = corpus_df.iloc[corpus_i]
            marker = " <-- correct match" if corpus_i == relevant_indices[i] else ""
            print(f"  #{rank}: {row['name']!r} (sim={similarity[i, corpus_i]:.3f}){marker}")


if __name__ == "__main__":
    main()
