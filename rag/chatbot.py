"""
The RAG assistant's answer synthesis.

Synthesis strategy: extractive, not generative. The answer is built by
selecting and lightly combining the highest-scoring retrieved chunk(s)
verbatim (with light formatting), rather than feeding retrieved chunks
into a generative LLM to paraphrase into a fluent free-form answer.

This is a real, worth-defending trade-off:
  - Pro: every word of the answer is traceable to a specific KB entry -
    zero hallucination risk, and the `source` field in the API response
    is always exactly correct, not a best-effort attribution.
  - Con: answers read a bit more like "here's the relevant fact" than a
    natural conversational reply a fine-tuned LLM would produce.

`generate_answer()` is written as the one clearly-marked place to plug in
an LLM call (OpenAI/Anthropic/local model) for more fluent synthesis
later - everything upstream of it (retrieval, source tracking) doesn't
change either way, which is the point of separating retrieval from
synthesis as distinct steps.
"""

from __future__ import annotations

from rag.vector_store import KnowledgeVectorStore


def generate_answer(
    question: str,
    vector_store: KnowledgeVectorStore,
    recipe_id: int | None = None,
    top_k: int = 2,
) -> dict:
    hits = vector_store.search(question, top_k=top_k, recipe_id=recipe_id)

    if not hits:
        return {
            "answer": "I don't have grounded information to answer that confidently. "
                      "Try asking about ingredient substitutions, dietary adaptations "
                      "(vegan/diabetic/lower-calorie), cooking techniques, or a specific recipe.",
            "confidence": "low",
            "sources": [],
        }

    best = hits[0]
    # Confidence is a direct, honest function of the retrieval score, not
    # a separately learned/guessed number - it tells the caller how
    # lexically close the question was to what's actually in the KB,
    # which is exactly what determines whether this extractive answer is
    # trustworthy.
    if best["score"] > 0.3:
        confidence = "high"
    elif best["score"] > 0.12:
        confidence = "medium"
    else:
        confidence = "low"

    answer = best["content"]
    if len(hits) > 1 and hits[1]["score"] > 0.12:
        answer += f" Related: {hits[1]['content']}"

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": [{"id": h["id"], "topic": h["topic"], "score": h["score"]} for h in hits],
    }


def suggest_substitution(ingredient: str, vector_store: KnowledgeVectorStore, recipe_id: int | None = None) -> dict:
    """Thin, more targeted wrapper around generate_answer for the
    /substitute-ingredient endpoint specifically - phrases the query to
    bias retrieval toward substitution-KB entries for that ingredient."""
    question = f"substitute substitution replace {ingredient}"
    result = generate_answer(question, vector_store, recipe_id=recipe_id, top_k=1)
    return {
        "ingredient": ingredient,
        "suggestion": result["answer"],
        "confidence": result["confidence"],
        "source": result["sources"][0]["id"] if result["sources"] else None,
    }
