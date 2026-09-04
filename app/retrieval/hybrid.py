from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = 60,
    top_k: int = 15,
) -> list[dict[str, Any]]:

    scores = defaultdict(float)
    chunks = {}

    result_sources = defaultdict(list)

    for source_name, results in (
        ("vector", vector_results),
        ("bm25", bm25_results),
    ):

        for rank, result in enumerate(
            results,
            start=1,
        ):

            chunk = result["chunk"]
            chunk_id = chunk["chunk_id"]

            scores[chunk_id] += (
                1.0 / (k + rank)
            )

            chunks[chunk_id] = chunk
            result_sources[chunk_id].append(
                source_name
            )

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    return [
        {
            "rrf_score": score,
            "chunk": chunks[chunk_id],
            "retrieved_by": result_sources[chunk_id],
        }
        for chunk_id, score in ranked
    ]