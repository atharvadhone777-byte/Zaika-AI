from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.utils.logger import get_logger

logger = get_logger(__name__)


class RecipeNotFoundError(Exception):
    def __init__(self, recipe_id: int):
        self.recipe_id = recipe_id
        super().__init__(f"Recipe {recipe_id} not found")


class ModelNotReadyError(Exception):
    pass


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RecipeNotFoundError)
    async def recipe_not_found_handler(request: Request, exc: RecipeNotFoundError):
        return _error_response(404, "recipe_not_found", str(exc))

    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready_handler(request: Request, exc: ModelNotReadyError):
        return _error_response(503, "model_not_ready", "The model is still loading. Try again shortly.")

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
        return _error_response(500, "internal_error", "An unexpected error occurred.")
