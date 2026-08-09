"""
FAISS index wrapper for nearest-neighbor recipe retrieval.

Index type: IndexFlatIP (exact inner-product search), not an approximate
index (IVF, HNSW). Rationale: approximate indexes trade recall for speed,
and that trade only pays off once brute-force search is actually too slow
- for ~180K recipes at 128 dimensions, exact search is comfortably
sub-10ms per query on CPU. Reaching for an approximate index here would be
solving a scaling problem that doesn't exist yet at this size, at the cost
of a tunable recall/speed knob that adds complexity with no current
benefit. This is exactly the kind of thing worth flagging as a stated,
deliberate "future improvement, not needed yet" rather than silently
picking the fancier-sounding option.

Inner product (not L2 distance) because the encoder's output is
L2-normalized (see ml/models/encoder.py), which makes inner product
mathematically equivalent to cosine similarity - consistent with the
similarity metric the contrastive training loss was optimized against.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class RecipeIndex:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.recipe_ids: list[int] = []  # index position -> recipe id

    def build(self, embeddings: np.ndarray, recipe_ids: list[int]) -> None:
        assert embeddings.shape[0] == len(recipe_ids)
        self.index.add(embeddings.astype("float32"))
        self.recipe_ids = list(recipe_ids)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Returns [(recipe_id, similarity_score), ...] sorted by descending similarity."""
        query = query_embedding.astype("float32").reshape(1, -1)
        scores, positions = self.index.search(query, top_k)
        results = []
        for pos, score in zip(positions[0], scores[0]):
            if pos == -1:  # FAISS pads with -1 if top_k exceeds index size
                continue
            results.append((self.recipe_ids[pos], float(score)))
        return results

    def save(self, index_path: Path, ids_path: Path) -> None:
        faiss.write_index(self.index, str(index_path))
        np.save(ids_path, np.array(self.recipe_ids, dtype="int64"))

    @classmethod
    def load(cls, index_path: Path, ids_path: Path) -> "RecipeIndex":
        index = faiss.read_index(str(index_path))
        obj = cls(embedding_dim=index.d)
        obj.index = index
        obj.recipe_ids = np.load(ids_path).tolist()
        return obj
