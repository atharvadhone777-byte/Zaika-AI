"""
Builds the FAISS index over the FULL recipe corpus (train+val+test) using
the trained encoder, and saves recipe metadata needed at serving time
(nutrition, steps, difficulty, etc.) as a lookup table. Run once after
training; the API loads the saved artifacts, it never re-encodes the
corpus at request time.
"""

from __future__ import annotations

import json

import pandas as pd
import tensorflow as tf

from ml.config import CLEANED_RECIPES_PATH, VOCAB_PATH, MODELS_DIR, MODEL
from ml.data.preprocessing import estimate_difficulty
from ml.data.tokenizer import IngredientVocabulary
from ml.inference.retriever import RecipeIndex
import ml.models.encoder  # noqa: F401 - registers custom Lambda functions before load_model


def main():
    vocab = IngredientVocabulary.load(VOCAB_PATH)
    max_len = MODEL.max_ingredients_per_recipe
    encoder = tf.keras.models.load_model(str(MODELS_DIR / "encoder.h5"), safe_mode=False)

    corpus_df = pd.read_parquet(CLEANED_RECIPES_PATH)
    corpus_df["ingredients"] = corpus_df["ingredients"].apply(json.loads)
    corpus_df["steps"] = corpus_df["steps"].apply(json.loads)
    corpus_df["tags"] = corpus_df["tags"].apply(json.loads)

    corpus_df["difficulty"] = corpus_df.apply(
        lambda r: estimate_difficulty(r["n_steps"], r["minutes"], r["n_ingredients"]), axis=1
    )

    from ml.data.dataset import _pad_or_truncate
    import numpy as np
    ids = np.array(
        [_pad_or_truncate(vocab.encode(ings), max_len) for ings in corpus_df["ingredients"]],
        dtype="int32",
    )
    embeddings = encoder.predict(ids, verbose=0)

    index = RecipeIndex(embedding_dim=embeddings.shape[1])
    index.build(embeddings, corpus_df["id"].tolist())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    index.save(MODELS_DIR / "recipe.index", MODELS_DIR / "recipe_ids.npy")

    # Metadata lookup table: everything the API needs to return for a
    # recipe_id without touching the embedding model again. Saved as
    # parquet (not re-reading recipes_clean.parquet at serve time) so the
    # serving path only depends on files inside models/v1/ - one
    # self-contained artifact directory, which is what actually gets
    # deployed/versioned together.
    metadata_cols = [
        "id", "name", "minutes", "n_steps", "steps", "ingredients", "n_ingredients",
        "difficulty", "description", "tags", "calories", "total_fat_pdv", "sugar_pdv",
        "sodium_pdv", "protein_pdv", "saturated_fat_pdv", "carbohydrates_pdv",
    ]
    metadata_df = corpus_df[metadata_cols].copy()
    metadata_df["steps"] = metadata_df["steps"].apply(json.dumps)
    metadata_df["ingredients"] = metadata_df["ingredients"].apply(json.dumps)
    metadata_df["tags"] = metadata_df["tags"].apply(json.dumps)
    metadata_df.to_parquet(MODELS_DIR / "recipe_metadata.parquet", index=False)

    print(f"Indexed {len(corpus_df)} recipes -> {MODELS_DIR / 'recipe.index'}")
    print(f"Metadata saved -> {MODELS_DIR / 'recipe_metadata.parquet'}")


if __name__ == "__main__":
    main()
