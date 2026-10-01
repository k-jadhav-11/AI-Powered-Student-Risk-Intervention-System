# ai_assistant/rerank.py
from __future__ import annotations
import math
from . import vectorstore


# Configuration
OVERFETCH_MULTIPLIER = 3


# Similarity Helpers
def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculates cosine similarity between two embedding vectors.
    Returns zero when either vector has no magnitude."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


# Candidate Re-ranking

def rerank(
    query_factors: list[str],
    candidates: list[dict],
    top_k: int = 2,
) -> dict:
    """Re-scores candidate resources against the student's top factors.
    Returns the highest-scoring resources with similarity scores and backend."""
    if not candidates:
        return {
            "suggestions": [],
            "source_ids": [],
            "scores": [],
            "backend": "no candidates",
        }

    embed_fn, backend = vectorstore.get_embedding_function()

    query_text = " ".join(query_factors) if query_factors else ""
    query_vec = embed_fn([query_text])[0]

    candidate_texts = [candidate["text"] for candidate in candidates]
    candidate_vecs = embed_fn(candidate_texts)

    scored = [
        (candidate, _cosine_similarity(query_vec, vector))
        for candidate, vector in zip(candidates, candidate_vecs)
    ]

    scored.sort(key=lambda pair: pair[1], reverse=True)
    top = scored[:top_k]

    return {
        "suggestions": [candidate["text"] for candidate, _ in top],
        "source_ids": [candidate["id"] for candidate, _ in top],
        "scores": [round(score, 4) for _, score in top],
        "backend": backend,
    }


# Exact Factor Matching

def rank_exact_matches(
    top_factors: list[str],
    exclude_ids: set[str] = frozenset(),
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Ranks resources by overlap with the student's top factors.
    Optional factor weights provide an additional relevance signal for ties."""
    scored = [
        (
            item,
            len(set(item["factor_tags"]) & set(top_factors)),
            (
                sum(
                    weights.get(factor, 0.0)
                    for factor in (set(item["factor_tags"]) & set(top_factors))
                )
                if weights
                else 0.0
            ),
        )
        for item in vectorstore.STARTER_REMEDIAL_CONTENT
        if (
            item["id"] not in exclude_ids
            and set(item["factor_tags"]) & set(top_factors)
        )
    ]

    scored.sort(
        key=lambda triple: (triple[1], triple[2]),
        reverse=True,
    )

    return [item for item, _, _ in scored]


# Combined Retrieval and Re-ranking

def rerank_remaining_after_tag_match(
    top_factors: list[str],
    tagged_ids: set[str],
    top_k: int = 2,
) -> dict:
    """Builds a wider candidate pool excluding exact tag matches, then re-ranks it.
    Returns the highest-scoring remaining resources for the intervention."""
    pool_size = min(
        len(vectorstore.STARTER_REMEDIAL_CONTENT),
        top_k * OVERFETCH_MULTIPLIER,
    )

    tag_matched = [
        item
        for item in vectorstore.STARTER_REMEDIAL_CONTENT
        if (
            item["id"] not in tagged_ids
            and set(item["factor_tags"]) & set(top_factors)
        )
    ]

    remaining_pool = [
        item
        for item in vectorstore.STARTER_REMEDIAL_CONTENT
        if item["id"] not in tagged_ids and item not in tag_matched
    ]

    candidates = (tag_matched + remaining_pool)[:pool_size]

    return rerank(
        top_factors,
        candidates,
        top_k=top_k,
    )