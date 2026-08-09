from __future__ import annotations

from pydantic import BaseModel, Field


class SubstituteIngredientRequest(BaseModel):
    ingredient: str = Field(min_length=1)
    recipe_id: int | None = Field(default=None, description="If given, restricts retrieval to this recipe's context")
    constraint: str | None = Field(default=None, description="Optional dietary constraint, e.g. 'vegan', 'diabetic'")


class SubstituteIngredientResponse(BaseModel):
    ingredient: str
    suggestion: str
    confidence: str
    source: str | None


class ChatSource(BaseModel):
    id: str
    topic: str
    score: float


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    recipe_id: int | None = None


class ChatResponse(BaseModel):
    answer: str
    confidence: str
    sources: list[ChatSource]
