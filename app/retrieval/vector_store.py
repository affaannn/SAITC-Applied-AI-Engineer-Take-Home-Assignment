import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

class VectorStoreError(Exception):
    """Raised when vector index operations fail."""

class FAISSVectorStore:

    def __init__(
        self,
        dimension: int,
    ):
        self.dimension = dimension

        # Embeddings are normalized beforehand.
        # Inner product therefore behaves as cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)

        self.documents: list[dict[str, Any]] = []

    def add(
        self,
        embeddings: np.ndarray,
        documents: list[dict[str, Any]],
    ) -> None:

        if len(embeddings) != len(documents):
            raise VectorStoreError(
                "Number of embeddings does not match "
                "number of documents."
            )

        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Expected embedding dimension {self.dimension}, "
                f"received {embeddings.shape[1]}."
            )

        self.index.add(
            embeddings.astype("float32")
        )

        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:

        if self.index.ntotal == 0:
            return []

        top_k = min(
            top_k,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_embedding.astype("float32"),
            top_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index < 0:
                continue

            document = self.documents[index]

            results.append(
                {
                    "score": float(score),
                    "chunk": document,
                }
            )

        return results

    def save(
        self,
        index_path: str | Path,
        documents_path: str | Path,
    ) -> None:

        index_path = Path(index_path)
        documents_path = Path(documents_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        documents_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(index_path),
        )

        with documents_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.documents,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        documents_path: str | Path,
    ) -> "FAISSVectorStore":

        index_path = Path(index_path)
        documents_path = Path(documents_path)

        if not index_path.exists():
            raise VectorStoreError(
                f"FAISS index not found: {index_path}"
            )

        if not documents_path.exists():
            raise VectorStoreError(
                f"Document metadata not found: {documents_path}"
            )

        index = faiss.read_index(
            str(index_path)
        )

        with documents_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            documents = json.load(file)

        store = cls(
            dimension=index.d
        )

        store.index = index
        store.documents = documents

        return store