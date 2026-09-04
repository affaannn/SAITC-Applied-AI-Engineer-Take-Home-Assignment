from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
    ):
        self.model_name = model_name

        self.model = CrossEncoder(
            model_name,
            device=device,
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 6,
    ) -> list[dict]:

        if not candidates:
            return []

        pairs = [
            (
                query,
                candidate["chunk"]["text"],
            )
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked = []

        for candidate, score in zip(
            candidates,
            scores,
        ):
            item = candidate.copy()

            item["reranker_score"] = float(score)

            reranked.append(item)

        reranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        return reranked[:top_k]