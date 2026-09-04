import json
import time
from pathlib import Path

import yaml

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import (
    FAISSVectorStore,
)


def load_config(
    path: str = "config.yaml",
) -> dict:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


def load_chunks(
    path: str | Path,
) -> list[dict]:

    chunks = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            chunks.append(
                json.loads(line)
            )

    return chunks


def main():

    config = load_config()

    start_time = time.perf_counter()

    chunks_path = Path(
        config["output"]["chunks_file"]
    )

    chunks = load_chunks(
        chunks_path
    )

    print("=" * 45)
    print("VECTOR INDEX BUILD")
    print("=" * 45)

    print(
        f"Embedding model : "
        f"{config['embedding']['model']}"
    )

    print(
        f"Chunks          : "
        f"{len(chunks)}"
    )

    print()

    embedding_model = EmbeddingModel(
        model_name=config["embedding"]["model"]
    )

    texts = [
        EmbeddingModel.build_embedding_text(chunk)
        for chunk in chunks
    ]

    embedding_start = time.perf_counter()

    embeddings = embedding_model.embed_documents(
        texts,
        batch_size=config["embedding"]["batch_size"],
    )

    embedding_time = (
        time.perf_counter()
        - embedding_start
    )

    store = FAISSVectorStore(
        dimension=embedding_model.dimension
    )

    index_start = time.perf_counter()

    store.add(
        embeddings,
        chunks,
    )

    store.save(
        index_path=config["index"]["faiss_path"],
        documents_path=config["index"]["documents_path"],
    )

    index_time = (
        time.perf_counter()
        - index_start
    )

    total_time = (
        time.perf_counter()
        - start_time
    )

    print()
    print("=" * 45)
    print("VECTOR INDEX SUMMARY")
    print("=" * 45)

    print(
        f"Embedding dimension : "
        f"{embedding_model.dimension}"
    )

    print(
        f"Vectors indexed     : "
        f"{store.index.ntotal}"
    )

    print(
        f"Embedding time      : "
        f"{embedding_time:.2f} sec"
    )

    print(
        f"Index build time    : "
        f"{index_time:.3f} sec"
    )

    print(
        f"Total time          : "
        f"{total_time:.2f} sec"
    )

    print(
        f"FAISS index         : "
        f"{config['index']['faiss_path']}"
    )

    print("=" * 45)

if __name__ == "__main__":
    main()
