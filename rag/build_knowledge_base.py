"""Builds the RAG vector store from the substitution KB + recipe-derived
chunks and saves it to models/v1/. Run after ml/inference/build_index.py
(needs the difficulty-annotated recipe metadata it produces)."""

from __future__ import annotations

import json

import pandas as pd

from ml.config import MODELS_DIR
from rag.knowledge_base import SUBSTITUTION_KB, build_recipe_chunks
from rag.vector_store import KnowledgeVectorStore


def main():
    metadata_df = pd.read_parquet(MODELS_DIR / "recipe_metadata.parquet")
    metadata_df["ingredients"] = metadata_df["ingredients"].apply(json.loads)
    metadata_df["tags"] = metadata_df["tags"].apply(json.loads)

    recipe_chunks = build_recipe_chunks(metadata_df)
    all_chunks = SUBSTITUTION_KB + recipe_chunks

    store = KnowledgeVectorStore()
    store.build(all_chunks)
    store.save(MODELS_DIR / "rag_knowledge_store.joblib")

    print(f"Built RAG store: {len(SUBSTITUTION_KB)} curated entries + "
          f"{len(recipe_chunks)} recipe chunks = {len(all_chunks)} total")
    print(f"Saved -> {MODELS_DIR / 'rag_knowledge_store.joblib'}")


if __name__ == "__main__":
    main()
