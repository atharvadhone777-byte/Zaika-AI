"""
RecipePredictor: loads all trained artifacts ONCE and exposes the
inference operations the API needs. This is the one class app/services/
is allowed to import from ml/ - everything else in ml/ is training-time
only. Keeping that boundary narrow (one class, a small method surface) is
what makes it easy to reason about what the API actually depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from ml.config import MODELS_DIR, VOCAB_PATH, MODEL
from ml.data.dataset import _pad_or_truncate
from ml.data.preprocessing import normalize_ingredient_list
from ml.data.tokenizer import IngredientVocabulary
from ml.inference.retriever import RecipeIndex
import ml.models.encoder  # noqa: F401 - registers custom Lambda functions before load_model


class RecipePredictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.vocab = IngredientVocabulary.load(VOCAB_PATH)
        self.max_len = MODEL.max_ingredients_per_recipe

        self.encoder = tf.keras.models.load_model(str(models_dir / "encoder.h5"), safe_mode=False)
        self.index = RecipeIndex.load(models_dir / "recipe.index", models_dir / "recipe_ids.npy")

        metadata_df = pd.read_parquet(models_dir / "recipe_metadata.parquet")
        metadata_df["steps"] = metadata_df["steps"].apply(json.loads)
        metadata_df["ingredients"] = metadata_df["ingredients"].apply(json.loads)
        self.metadata = metadata_df.set_index("id")

    def _embed(self, ingredients: list[str]) -> np.ndarray:
        ids = np.array([_pad_or_truncate(self.vocab.encode(ingredients), self.max_len)], dtype="int32")
        return self.encoder.predict(ids, verbose=0)[0]

    def _recipe_dict(self, recipe_id: int, match_score: float | None = None) -> dict:
        row = self.metadata.loc[recipe_id]
        return {
            "recipe_id": int(recipe_id),
            "title": row["name"],
            "match_score": round(float(match_score), 4) if match_score is not None else None,
            "required_ingredients": row["ingredients"],
            "cook_time_minutes": int(row["minutes"]),
            "difficulty": row["difficulty"],
            "steps": row["steps"],
            "description": row["description"] if pd.notna(row["description"]) else None,
            "nutrition": {
                "calories": round(float(row["calories"]), 1),
                "protein_g": round(float(row["protein_pdv"]), 1),   # PDV = % daily value, as stored by Food.com
                "fat_g": round(float(row["total_fat_pdv"]), 1),
                "carbs_g": round(float(row["carbohydrates_pdv"]), 1),
            },
        }

    def recommend(
        self,
        ingredients: list[str],
        top_k: int = 5,
        max_missing_ingredients: int | None = None,
        fetch_multiplier: int = 4,
    ) -> list[dict]:
        """
        Returns up to top_k recipes ranked by embedding similarity to the
        given ingredients, each annotated with which of its ingredients
        the user already has vs. is missing.

        fetch_multiplier: FAISS is queried for top_k * fetch_multiplier
        candidates rather than exactly top_k, because the max_missing_ingredients
        filter is applied AFTER retrieval - if we only fetched top_k and
        then filtered, a strict max_missing filter could leave fewer than
        top_k results even when good matches exist slightly further down
        the ranking. Over-fetching and filtering is simpler and more
        correct than trying to bake an arbitrary hard constraint into the
        similarity search itself.
        """
        normalized = normalize_ingredient_list(ingredients)
        if not normalized:
            return []

        query_embedding = self._embed(normalized)
        candidates = self.index.search(query_embedding, top_k=top_k * fetch_multiplier)

        have = set(normalized)
        results = []
        for recipe_id, score in candidates:
            recipe = self._recipe_dict(recipe_id, match_score=score)
            required = set(recipe["required_ingredients"])
            missing = sorted(required - have)
            recipe["missing_ingredients"] = missing

            if max_missing_ingredients is not None and len(missing) > max_missing_ingredients:
                continue

            recipe["similar_recipe_ids"] = self.similar_recipes(recipe_id, top_k=3)
            results.append(recipe)
            if len(results) >= top_k:
                break

        return results

    def similar_recipes(self, recipe_id: int, top_k: int = 3) -> list[int]:
        """Recipes whose full ingredient list embeds near this recipe's -
        reuses the same index and embedding space as ingredient-based
        recommendation, rather than needing a second similarity model."""
        recipe_ingredients = self.metadata.loc[recipe_id, "ingredients"]
        embedding = self._embed(recipe_ingredients)
        neighbors = self.index.search(embedding, top_k=top_k + 1)  # +1 since the recipe will match itself
        return [rid for rid, _ in neighbors if rid != recipe_id][:top_k]

    def get_recipe(self, recipe_id: int) -> dict | None:
        if recipe_id not in self.metadata.index:
            return None
        recipe = self._recipe_dict(recipe_id)
        # No user ingredient context here (unlike recommend()), so there's
        # nothing to compute "missing" against - explicitly empty rather
        # than absent, since the schema treats it as always-present.
        recipe["missing_ingredients"] = []
        recipe["similar_recipe_ids"] = self.similar_recipes(recipe_id, top_k=3)
        return recipe
