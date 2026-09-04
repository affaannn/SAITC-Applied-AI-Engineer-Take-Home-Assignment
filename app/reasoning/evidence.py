from typing import Any


class EvidenceSelector:
    """
    Builds the final evidence candidate set without trusting
    any single retrieval or reranking signal.
    """

    def __init__(
        self,
        max_evidence: int = 8,
    ):
        self.max_evidence = max_evidence

    def select(
        self,
        reranked_results: list[dict[str, Any]],
        hybrid_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        selected = []
        seen = set()

        # First take reranked candidates.
        for result in reranked_results:
            chunk = result["chunk"]
            chunk_id = chunk["chunk_id"]

            if chunk_id in seen:
                continue

            selected.append(result)
            seen.add(chunk_id)

        # Then retain strong hybrid evidence that the reranker
        # may have incorrectly suppressed.
        for result in hybrid_results:
            chunk = result["chunk"]
            chunk_id = chunk["chunk_id"]

            if chunk_id in seen:
                continue

            selected.append(result)
            seen.add(chunk_id)

            if len(selected) >= self.max_evidence:
                break

        return selected[:self.max_evidence]