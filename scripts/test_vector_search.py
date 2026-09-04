import yaml
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import FAISSVectorStore


TEST_QUERIES = [
    "What is the company's annual leave policy?",
    "How much notice must an employee give during probation?",
    "What is the current price of Atlas Professional?",
    "What refund window applies to Atlas Enterprise customers?",
    "What is the vendor onboarding procedure?",
]


def load_config(
    path: str = "config.yaml",
) -> dict:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


def main():

    config = load_config()

    embedding_model = EmbeddingModel(
        model_name=config["embedding"]["model"]
    )

    store = FAISSVectorStore.load(
        index_path=config["index"]["faiss_path"],
        documents_path=config["index"]["documents_path"],
    )

    for query in TEST_QUERIES:

        print()
        print("=" * 100)
        print(f"QUERY: {query}")
        print("=" * 100)

        query_embedding = (
            embedding_model.embed_query(query)
        )

        results = store.search(
            query_embedding=query_embedding,
            top_k=5,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            chunk = result["chunk"]

            print()
            print(
                f"[{rank}] "
                f"score={result['score']:.4f}"
            )

            print(
                f"{chunk['document_id']} | "
                f"page {chunk['page']} | "
                f"{chunk.get('section')}"
            )

            preview = (
                chunk["text"]
                .replace("\n", " ")
                [:350]
            )

            print(preview)


if __name__ == "__main__":
    main()