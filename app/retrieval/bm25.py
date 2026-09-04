import re
from typing import Any

from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(
        self,
        documents: list[dict[str, Any]],
    ):
        self.documents = documents

        self.tokenized_documents = [
            self._tokenize(
                self._build_search_text(document)
            )
            for document in documents
        ]

        self.index = BM25Okapi(
            self.tokenized_documents
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    @staticmethod
    def _build_search_text(
        chunk: dict[str, Any],
    ) -> str:

        parts = [
            chunk.get("title", ""),
            chunk.get("section", ""),
            chunk.get("text", ""),
        ]

        return "\n".join(
            part
            for part in parts
            if part
        )

    def search(
        self,
        query: str,
        top_k: int = 15,
    ) -> list[dict[str, Any]]:

        tokens = self._tokenize(query)

        scores = self.index.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            {
                "score": float(scores[index]),
                "chunk": self.documents[index],
            }
            for index in ranked_indices
        ]