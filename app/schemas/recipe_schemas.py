"""
Pydantic request/response models for /recommend and /generate-recipe.

Every field the frontend needs is modeled explicitly (not passed through
as a raw dict) - this is what gives FastAPI's auto-generated OpenAPI docs
real value, and what turns a malformed request into a clean 422 instead
of a 500 from deep inside the ML code.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class NutritionInfo(BaseModel):
    calories: float
    protein_g: float = Field(description="Protein, as % daily value (Food.com's PDV convention)")
    fat_g: float = Field(description="Total fat, as % daily value")
    carbs_g: float = Field(description="Carbohydrates, as % daily value")


class RecommendRequest(BaseModel):
    ingredients: list[str] = Field(min_length=1, description="Ingredients the user currently has")
    top_k: int = Field(default=5, ge=1, le=20)
    max_missing_ingredients: int | None = Field(
        default=None, ge=0, description="If set, only return recipes missing at most this many ingredients"
    )

    @field_validator("ingredients")
    @classmethod
    def _non_empty_strings(cls, value: list[str]) -> list[str]:
        cleaned = [v.strip() for v in value if v.strip()]
        if not cleaned:
            raise ValueError("ingredients must contain at least one non-empty string")
        return cleaned


class RecipeResult(BaseModel):
    recipe_id: int
    title: str
    match_score: float | None = None
    required_ingredients: list[str]
    missing_ingredients: list[str] = Field(default_factory=list)
    cook_time_minutes: int
    difficulty: str
    steps: list[str]
    description: str | None = None
    nutrition: NutritionInfo
    similar_recipe_ids: list[int]


class RecommendResponse(BaseModel):
    results: list[RecipeResult]
    count: int


class GenerateRecipeRequest(BaseModel):
    recipe_id: int


class GenerateRecipeResponse(BaseModel):
    recipe: RecipeResult
