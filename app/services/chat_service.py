from __future__ import annotations

from rag import chatbot
from rag.vector_store import KnowledgeVectorStore


class ChatService:
    def __init__(self, vector_store: KnowledgeVectorStore):
        self._store = vector_store

    def ask(self, question: str, recipe_id: int | None) -> dict:
        return chatbot.generate_answer(question, self._store, recipe_id=recipe_id)

    def substitute(self, ingredient: str, recipe_id: int | None) -> dict:
        return chatbot.suggest_substitution(ingredient, self._store, recipe_id=recipe_id)
