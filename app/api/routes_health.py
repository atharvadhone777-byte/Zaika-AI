from __future__ import annotations

import time

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])
_start_time = time.time()


@router.get("/")
async def root():
    return {
        "service": "ai-recipe-generator-api",
        "docs": "/docs",
        "endpoints": ["/api/v1/recommend", "/api/v1/generate-recipe", "/api/v1/substitute-ingredient", "/api/v1/chat", "/health"],
    }


@router.get("/health")
async def health(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    vector_store = getattr(request.app.state, "vector_store", None)
    return {
        "status": "ok" if predictor is not None else "starting",
        "model_loaded": predictor is not None,
        "rag_loaded": vector_store is not None,
        "index_size": len(predictor.index.recipe_ids) if predictor is not None else 0,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }
