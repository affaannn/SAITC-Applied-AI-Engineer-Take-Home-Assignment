from typing import Any


class ContextExpander:
    """
    Expands retrieved chunks with neighbouring chunks from
    the same document/page.

    This helps recover context fragmented by PDF tables and
    section boundaries without altering the original chunks.
    """

    def __init__(
        self,
        documents: list[dict[str, Any]],
    ):
        self.documents = documents

        self.chunk_position = {
            chunk["chunk_id"]: index
            for index, chunk in enumerate(documents)
        }

    def expand(
        self,
        results: list[dict[str, Any]],
        window: int = 1,
    ) -> list[dict[str, Any]]:

        expanded = []
        seen = set()

        for result in results:

            chunk = result["chunk"]
            chunk_id = chunk["chunk_id"]

            index = self.chunk_position.get(chunk_id)

            if index is None:
                continue

            start = max(
                0,
                index - window,
            )

            end = min(
                len(self.documents),
                index + window + 1,
            )

            for neighbour_index in range(start, end):

                neighbour = self.documents[neighbour_index]

                # Never cross document boundaries.
                if (
                    neighbour["document_id"]
                    != chunk["document_id"]
                ):
                    continue

                neighbour_id = neighbour["chunk_id"]

                if neighbour_id in seen:
                    continue

                seen.add(neighbour_id)

                expanded.append(
                    {
                        "score": result.get("rrf_score", result.get("score", 0.0),),
                        "chunk": neighbour,
                        "retrieved_from": chunk_id,
                        "is_neighbor": neighbour_id != chunk_id,
                    }
                )

        return expanded