from __future__ import annotations

from ml.inference.predictor import RecipePredictor
from app.utils.exceptions import RecipeNotFoundError


class RecipeService:
    def __init__(self, predictor: RecipePredictor):
        self._predictor = predictor

    def recommend(self, ingredients: list[str], top_k: int, max_missing_ingredients: int | None) -> list[dict]:
        return self._predictor.recommend(
            ingredients=ingredients, top_k=top_k, max_missing_ingredients=max_missing_ingredients
        )

    def get_recipe(self, recipe_id: int) -> dict:
        recipe = self._predictor.get_recipe(recipe_id)
        if recipe is None:
            raise RecipeNotFoundError(recipe_id)
        return recipe
