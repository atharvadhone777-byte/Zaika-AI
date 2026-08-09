"""
Retrieval evaluation metrics: precision@k, recall@k, mean reciprocal rank (MRR).

Why these and not BLEU/ROUGE: BLEU/ROUGE measure n-gram overlap between
generated and reference TEXT, which only makes sense when the model's job
is to produce free text. This model's job is to rank candidates, so the
correct metrics are ranking metrics - the same family used to evaluate
real recommendation and search systems. Reporting BLEU here would be a
red flag in an interview: it would suggest a mismatch between what the
model does and what's being measured.
"""

from __future__ import annotations

import numpy as np


def precision_at_k(ranked_indices: np.ndarray, relevant_index: int, k: int) -> float:
    """Fraction of the top-k that are relevant. With exactly one relevant
    item per query (our setup: each query has exactly one source recipe),
    this is either 1/k (hit) or 0 (miss) - included anyway because
    precision@k is the standard metric name reviewers expect to see, and
    it generalizes cleanly if the definition of "relevant" is ever
    broadened (e.g. to "any recipe sharing >=80% of ingredients")."""
    top_k = ranked_indices[:k]
    hits = np.sum(top_k == relevant_index)
    return hits / k


def recall_at_k(ranked_indices: np.ndarray, relevant_index: int, k: int) -> float:
    """With one relevant item, recall@k is 1.0 if it's in the top-k, else 0.0."""
    top_k = ranked_indices[:k]
    return 1.0 if relevant_index in top_k else 0.0


def reciprocal_rank(ranked_indices: np.ndarray, relevant_index: int) -> float:
    """1/rank of the relevant item (1-indexed), or 0 if absent from the ranking."""
    positions = np.where(ranked_indices == relevant_index)[0]
    if len(positions) == 0:
        return 0.0
    return 1.0 / (positions[0] + 1)


def evaluate_retrieval(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    relevant_indices: list[int],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """
    query_embeddings: (n_queries, dim)
    corpus_embeddings: (n_corpus, dim) - the full recipe corpus being retrieved from
    relevant_indices: for each query, the index into corpus_embeddings of
        its true source recipe (queries were generated as subsets of that
        recipe's ingredients, see ml/data/dataset.py:make_training_pairs)

    Embeddings are assumed L2-normalized (true for this encoder's output),
    so a plain dot product is cosine similarity - brute-force here since
    the corpus is small enough that this is instant; ml/inference/retriever.py
    uses FAISS for the same computation at serving scale.
    """
    similarity = query_embeddings @ corpus_embeddings.T  # (n_queries, n_corpus)
    rankings = np.argsort(-similarity, axis=1)  # descending similarity -> ranked candidate indices

    results = {f"precision@{k}": [] for k in k_values}
    results.update({f"recall@{k}": [] for k in k_values})
    mrr_scores = []

    for i, relevant_idx in enumerate(relevant_indices):
        ranked = rankings[i]
        for k in k_values:
            results[f"precision@{k}"].append(precision_at_k(ranked, relevant_idx, k))
            results[f"recall@{k}"].append(recall_at_k(ranked, relevant_idx, k))
        mrr_scores.append(reciprocal_rank(ranked, relevant_idx))

    summary = {metric: round(float(np.mean(scores)), 4) for metric, scores in results.items()}
    summary["mrr"] = round(float(np.mean(mrr_scores)), 4)
    summary["n_queries"] = len(relevant_indices)
    summary["corpus_size"] = len(corpus_embeddings)
    return summary
