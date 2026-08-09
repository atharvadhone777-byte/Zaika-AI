from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas.chat_schemas import (
    SubstituteIngredientRequest, SubstituteIngredientResponse, ChatRequest, ChatResponse,
)
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/v1", tags=["chat"])


def _service(request: Request) -> ChatService:
    return request.app.state.chat_service


@router.post("/substitute-ingredient", response_model=SubstituteIngredientResponse)
async def substitute_ingredient(payload: SubstituteIngredientRequest, request: Request):
    return _service(request).substitute(ingredient=payload.ingredient, recipe_id=payload.recipe_id)


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    return _service(request).ask(question=payload.question, recipe_id=payload.recipe_id)
