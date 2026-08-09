"""
Retrieval backend for the RAG assistant: TF-IDF + cosine similarity via
scikit-learn, not a neural sentence-embedding model.

Why TF-IDF instead of a pretrained sentence embedding model (e.g. a
sentence-transformers checkpoint): a good pretrained embedding model would
likely retrieve somewhat better on paraphrased questions, but it requires
downloading model weights from the internet at build/serve time, adding a
real external dependency (availability, licensing, download size) for a
knowledge base that's small (a few hundred to a few thousand short,
keyword-rich chunks - substitution facts and recipe summaries, not prose
with subtle paraphrasing to catch). For short, keyword-dense text like
this, TF-IDF's lexical matching captures the great majority of relevant
retrievals without that dependency. This is a real trade-off worth being
explicit about (see docs/ARCHITECTURE.md's "Future Improvements"), not a
default reached for without thought - swapping in a neural retriever later
only requires changing this one file, since `search()`'s interface
wouldn't need to change.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeVectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.chunk_vectors = None
        self.chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        # Topic is repeated to up-weight it relative to content: without
        # this, a question like "replace butter with oil" was observed
        # (see notebooks/eda notes / manual testing) to sometimes rank a
        # longer, tangentially-related chunk above the exact
        # "butter substitution" entry, simply because TF-IDF has more raw
        # term-overlap surface area to match against in the longer chunk.
        # Concatenating the topic twice biases matching toward chunks
        # whose TOPIC, not just incidental phrasing, matches the query.
        texts = [f"{c['topic']} {c['topic']} {c['content']}" for c in chunks]
        self.chunk_vectors = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3, recipe_id: int | None = None) -> list[dict]:
        """
        recipe_id: if given, restricts the search to that recipe's own
        chunk plus the general substitution KB - this is what lets
        /substitute-ingredient answer "for THIS recipe" questions instead
        of retrieving an unrelated recipe's chunk that happens to share
        keywords.
        """
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]

        if recipe_id is not None:
            allowed = {
                i for i, c in enumerate(self.chunks)
                if c.get("recipe_id") == recipe_id or "recipe_id" not in c
            }
            similarities = np.array([s if i in allowed else -1.0 for i, s in enumerate(similarities)])

        top_indices = np.argsort(-similarities)[:top_k]
        return [
            {**self.chunks[i], "score": round(float(similarities[i]), 4)}
            for i in top_indices if similarities[i] > 0
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "chunk_vectors": self.chunk_vectors, "chunks": self.chunks}, path)

    @classmethod
    def load(cls, path: Path) -> "KnowledgeVectorStore":
        data = joblib.load(path)
        store = cls()
        store.vectorizer = data["vectorizer"]
        store.chunk_vectors = data["chunk_vectors"]
        store.chunks = data["chunks"]
        return store
