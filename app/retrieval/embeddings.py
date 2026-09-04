from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingModel:
    """
    Local embedding wrapper used for both document and query embeddings.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    @property
    def dimension(self) -> int:
        """
        Return embedding vector dimension.
        """
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Embed document chunks.

        Embeddings are normalized so FAISS inner product can
        be interpreted as cosine similarity.
        """
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.astype("float32")

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Embed a single search query.
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.astype("float32")
    
    def build_embedding_text(
    chunk: dict,
    ) -> str:
        """
        Add lightweight document context to the semantic representation
        without modifying the source text used for citations.
        """

        parts = []

        title = chunk.get("title")

        if title:
            parts.append(title)

        section = chunk.get("section")

        if section:
            parts.append(section)

        parts.append(chunk["text"])

        return "\n\n".join(parts)