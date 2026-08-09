"""
Application entrypoint. Model + FAISS index + RAG store are loaded ONCE
at startup (via the lifespan handler below), not per-request - loading a
Keras model and a FAISS index takes real time, and doing that on every
request would make latency unpredictable and largely pointless work.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_recipe, routes_chat, routes_health
from app.config.settings import settings
from app.services.chat_service import ChatService
from app.services.recipe_service import RecipeService
from app.utils.exceptions import register_exception_handlers
from app.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Imported lazily inside the lifespan handler (not at module import
    # time) so that importing app.main - e.g. for a quick route-listing
    # script, or in tests that monkeypatch the model - doesn't force a
    # multi-second TensorFlow + FAISS load as a side effect of the import
    # itself.
    from ml.inference.predictor import RecipePredictor
    from rag.vector_store import KnowledgeVectorStore

    logger.info("Loading model artifacts from %s ...", settings.models_dir)
    predictor = RecipePredictor(models_dir=settings.models_dir)
    vector_store = KnowledgeVectorStore.load(settings.models_dir / "rag_knowledge_store.joblib")
    logger.info("Loaded. Index size: %d recipes.", len(predictor.index.recipe_ids))

    app.state.predictor = predictor
    app.state.vector_store = vector_store
    app.state.recipe_service = RecipeService(predictor)
    app.state.chat_service = ChatService(vector_store)

    yield

    logger.info("Shutting down.")


app = FastAPI(
    title="AI Recipe Generator API",
    description="Ingredient-aware recipe recommendation with a RAG-backed cooking assistant.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(routes_health.router)
app.include_router(routes_recipe.router)
app.include_router(routes_chat.router)
