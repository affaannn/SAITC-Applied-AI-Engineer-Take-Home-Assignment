from typing import Any


class EvidenceSelector:
    """
    Selects final evidence without trusting a single retrieval signal.

    Strategy:
    1. Preserve the strongest reranked evidence.
    2. Add expanded/neighbor evidence that may contain fragmented
       table rows or surrounding context.
    3. Avoid duplicate chunks.
    """

    def __init__(
        self,
        max_evidence: int = 10,
    ):
        self.max_evidence = max_evidence

    def select(
        self,
        reranked_results: list[dict[str, Any]],
        expanded_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        selected = []
        seen = set()

        # Do not allow the reranker to completely dominate.
        # It has already shown that it can rank semantically related
        # but factually incorrect chunks highly.
        reranker_limit = min(
            4,
            self.max_evidence,
        )

        for result in reranked_results[:reranker_limit]:

            chunk = result["chunk"]
            chunk_id = chunk["chunk_id"]

            if chunk_id in seen:
                continue

            selected.append(result)
            seen.add(chunk_id)

        # Add expanded candidates.
        #
        # This is important for fragmented PDF tables where the value
        # and its corresponding label may exist in neighbouring chunks.
        for result in expanded_results:

            if len(selected) >= self.max_evidence:
                break

            chunk = result["chunk"]
            chunk_id = chunk["chunk_id"]

            if chunk_id in seen:
                continue

            selected.append(result)
            seen.add(chunk_id)

        return selected