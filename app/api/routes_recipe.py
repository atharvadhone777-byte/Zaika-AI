from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas.recipe_schemas import (
    RecommendRequest, RecommendResponse, GenerateRecipeRequest, GenerateRecipeResponse,
)
from app.services.recipe_service import RecipeService

router = APIRouter(prefix="/api/v1", tags=["recipes"])


def _service(request: Request) -> RecipeService:
    return request.app.state.recipe_service


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(payload: RecommendRequest, request: Request):
    results = _service(request).recommend(
        ingredients=payload.ingredients,
        top_k=payload.top_k,
        max_missing_ingredients=payload.max_missing_ingredients,
    )
    return {"results": results, "count": len(results)}


@router.post("/generate-recipe", response_model=GenerateRecipeResponse)
async def generate_recipe(payload: GenerateRecipeRequest, request: Request):
    recipe = _service(request).get_recipe(payload.recipe_id)
    return {"recipe": recipe}
