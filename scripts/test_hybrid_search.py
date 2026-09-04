import json
from pathlib import Path

import yaml

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.context_expander import ContextExpander

TEST_QUERIES = [
    "What is the company's annual leave policy?",
    "How much notice must an employee give during probation?",
    "What is the current price of Atlas Professional?",
    "What refund window applies to Atlas Enterprise customers?",
    "What is the vendor onboarding procedure?",
]

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_chunks(path: str | Path) -> list[dict]:
    chunks = []

    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


def print_results(title: str, results: list[dict], limit: int = 10):
    print()
    print("-" * 100)
    print(title)
    print("-" * 100)

    for rank, result in enumerate(results[:limit], start=1):
        chunk = result["chunk"]

        score = (
            result.get("rrf_score")
            or result.get("score")
            or 0.0
        )

        retrieved_by = result.get("retrieved_by", [])
        is_neighbor = result.get("is_neighbor", False)

        print()
        print(
            f"[{rank}] "
            f"score={score:.6f} | "
            f"neighbor={is_neighbor} | "
            f"retrieved_by={retrieved_by}"
        )

        print(
            f"{chunk['document_id']} | "
            f"page {chunk['page']} | "
            f"{chunk.get('section')}"
        )

        print(
            chunk["text"]
            .replace("\n", " ")
            [:500]
        )


def main():
    config = load_config()

    chunks = load_chunks(
        config["output"]["chunks_file"]
    )

    embedding_model = EmbeddingModel(
        model_name=config["embedding"]["model"]
    )

    vector_store = FAISSVectorStore.load(
        index_path=config["index"]["faiss_path"],
        documents_path=config["index"]["documents_path"],
    )

    bm25 = BM25Retriever(chunks)

    expander = ContextExpander(chunks)

    vector_top_k = config["retrieval"].get(
        "vector_top_k",
        15,
    )

    bm25_top_k = config["retrieval"].get(
        "bm25_top_k",
        15,
    )

    hybrid_top_k = config["retrieval"].get(
        "hybrid_top_k",
        15,
    )

    neighbor_window = config["retrieval"].get(
        "neighbor_window",
        1,
    )

    for query in TEST_QUERIES:
        print()
        print("=" * 100)
        print(f"QUERY: {query}")
        print("=" * 100)

        query_embedding = embedding_model.embed_query(query)

        vector_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=vector_top_k,
        )

        bm25_results = bm25.search(
            query=query,
            top_k=bm25_top_k,
        )

        fused_results = reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=hybrid_top_k,
        )

        expanded_results = expander.expand(
            results=fused_results,
            window=neighbor_window,
        )
    
        print_results(
            "TOP HYBRID RESULTS BEFORE NEIGHBOR EXPANSION",
            fused_results,
            limit=10,
        )

        print_results(
            "EXPANDED CANDIDATES",
            expanded_results,
            limit=20,
        )


if __name__ == "__main__":
    main()