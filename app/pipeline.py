import json
from pathlib import Path

import yaml

from app.reasoning.router import QueryRouter
from app.reasoning.ambiguity import AmbiguityDetector
from app.reasoning.evidence import EvidenceSelector
from app.reasoning.conflict import ConflictDetector

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.context_expander import ContextExpander
from app.retrieval.reranker import CrossEncoderReranker

from app.generation.llamacpp_client import LlamaCppClient
from app.generation.prompts import (
    SYSTEM_PROMPT,
    build_grounded_prompt,
)
from app.generation.citations import validate_citations
from app.reasoning.fact_resolver import FactResolver


class RAGPipeline:

    def __init__(
        self,
        config_path: str = "config.yaml",
    ):

        # ---------------------------------------------------------
        # CONFIG
        # ---------------------------------------------------------

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.config = yaml.safe_load(file)

        # ---------------------------------------------------------
        # CHUNKS
        # ---------------------------------------------------------

        self.chunks = self._load_chunks(
            self.config["output"]["chunks_file"]
        )

        # ---------------------------------------------------------
        # ROUTING / REASONING
        # ---------------------------------------------------------

        self.router = QueryRouter()

        self.ambiguity = AmbiguityDetector()

        self.conflict_detector = ConflictDetector()

        self.evidence_selector = EvidenceSelector(
            max_evidence=self.config
            .get("evidence", {})
            .get("max_chunks", 10)
        )
        self.fact_resolver = FactResolver(
        as_of_date=self.config.get(
            "as_of_date",
            "2026-08-27",
        )
    )
        # ---------------------------------------------------------
        # EMBEDDINGS
        # ---------------------------------------------------------

        self.embedding_model = EmbeddingModel(
            model_name=self.config[
                "embedding"
            ]["model"]
        )

        # ---------------------------------------------------------
        # VECTOR STORE
        # ---------------------------------------------------------

        self.vector_store = FAISSVectorStore.load(
            index_path=self.config[
                "index"
            ]["faiss_path"],
            documents_path=self.config[
                "index"
            ]["documents_path"],
        )

        # ---------------------------------------------------------
        # BM25
        # ---------------------------------------------------------

        self.bm25 = BM25Retriever(
            self.chunks
        )

        # ---------------------------------------------------------
        # NEIGHBOR EXPANSION
        # ---------------------------------------------------------

        self.expander = ContextExpander(
            self.chunks
        )

        # ---------------------------------------------------------
        # RERANKER
        # ---------------------------------------------------------

        self.reranker = CrossEncoderReranker(
            model_name=self.config[
                "reranker"
            ]["model"]
        )

        # ---------------------------------------------------------
        # GENERATION
        # ---------------------------------------------------------

        generation_config = self.config.get(
            "generation",
            {},
        )

        self.llm = LlamaCppClient(
            tokenizer_name=generation_config.get(
                "tokenizer",
                "Qwen/Qwen2.5-3B-Instruct",
            ),
            base_url=generation_config.get(
                "base_url",
                "http://127.0.0.1:8080",
            ),
            timeout=generation_config.get(
                "timeout",
                600,
            ),
        )

    # =============================================================
    # LOAD CHUNKS
    # =============================================================

    @staticmethod
    def _load_chunks(
        path: str,
    ) -> list[dict]:

        chunks = []

        with Path(path).open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                chunks.append(
                    json.loads(line)
                )

        return chunks

    # =============================================================
    # RETRIEVAL
    # =============================================================

    def retrieve(
        self,
        query: str,
    ) -> dict:

        retrieval_config = self.config.get(
            "retrieval",
            {},
        )

        # ---------------------------------------------------------
        # VECTOR
        # ---------------------------------------------------------

        query_embedding = (
            self.embedding_model
            .embed_query(query)
        )

        vector_results = (
            self.vector_store.search(
                query_embedding=query_embedding,
                top_k=retrieval_config.get(
                    "vector_top_k",
                    15,
                ),
            )
        )

        # ---------------------------------------------------------
        # BM25
        # ---------------------------------------------------------

        bm25_results = self.bm25.search(
            query=query,
            top_k=retrieval_config.get(
                "bm25_top_k",
                15,
            ),
        )

        # ---------------------------------------------------------
        # RRF
        # ---------------------------------------------------------

        hybrid_results = (
            reciprocal_rank_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results,
                top_k=retrieval_config.get(
                    "hybrid_top_k",
                    15,
                ),
            )
        )

        # ---------------------------------------------------------
        # NEIGHBOR EXPANSION
        # ---------------------------------------------------------

        expanded_results = self.expander.expand(
            results=hybrid_results,
            window=retrieval_config.get(
                "neighbor_window",
                1,
            ),
        )

        # ---------------------------------------------------------
        # RERANK
        # ---------------------------------------------------------

        reranked_results = self.reranker.rerank(
            query=query,
            candidates=expanded_results,
            top_k=self.config
            .get("reranker", {})
            .get("top_k", 6),
        )

        # ---------------------------------------------------------
        # FINAL EVIDENCE
        #
        # IMPORTANT:
        # Use EXPANDED results here, NOT raw hybrid results.
        # This preserves fragmented table rows.
        # ---------------------------------------------------------

        evidence = self.evidence_selector.select(
            reranked_results=reranked_results,
            expanded_results=expanded_results,
        )

        return {
            "vector": vector_results,
            "bm25": bm25_results,
            "hybrid": hybrid_results,
            "expanded": expanded_results,
            "reranked": reranked_results,
            "evidence": evidence,
        }

    # =============================================================
    # ANSWER
    # =============================================================

    def answer(
        self,
        query: str,
    ) -> dict:

        # ---------------------------------------------------------
        # AMBIGUITY
        # ---------------------------------------------------------

        ambiguous = (
            self.ambiguity
            .is_ambiguous(query)
        )

        # ---------------------------------------------------------
        # ROUTER
        # ---------------------------------------------------------

        decision = self.router.route(
            query=query,
            ambiguous=ambiguous,
        )

        # ---------------------------------------------------------
        # SYSTEM PROMPT EXTRACTION
        # ---------------------------------------------------------

        if decision.route == "refuse_prompt_extraction":

            return {
                "route": decision.route,
                "answer": (
                    "I can't provide system instructions "
                    "or hidden prompts."
                ),
            }

        # ---------------------------------------------------------
        # POLICY BYPASS
        # ---------------------------------------------------------

        if decision.route == "decline_policy_bypass":

            return {
                "route": decision.route,
                "answer": (
                    "I can't help bypass the documented "
                    "approval process. I can explain the "
                    "required approval procedure instead."
                ),
            }

        # ---------------------------------------------------------
        # AMBIGUOUS QUERY
        # ---------------------------------------------------------

        if decision.route == "clarify":

            return {
                "route": decision.route,
                "answer": (
                    "Which limit do you mean? For example, "
                    "the API rate limit, file upload limit, "
                    "storage limit, document limit, or "
                    "another limit?"
                ),
            }

        # ---------------------------------------------------------
        # RETRIEVAL
        # ---------------------------------------------------------

        retrieval = self.retrieve(
            query
        )
        deterministic_result = (
            self.fact_resolver.resolve(
                query=query,
                evidence=retrieval["expanded"],
            )
        )

        if deterministic_result:

            citation_info = {
                "citations": deterministic_result[
                    "citations"
                ],
                "invalid_citations": [],
                "valid": True,
            }

            return {
                "route": (
                    "deterministic_fact_resolution"
                ),
                "answer": deterministic_result[
                    "answer"
                ],
                "conflict": {
                    "conflict_detected":
                        deterministic_result[
                            "conflict_detected"
                        ]
                },
                "citations": citation_info,
                "tokens": {
                    "prompt": 0,
                    "completion": 0,
                    "total": 0,
                },
                "generation_latency_sec": 0.0,
                "documents_used":
                    deterministic_result[
                        "documents_used"
                    ],
                "resolution_type":
                    deterministic_result[
                        "resolution_type"
                    ],
            }
        evidence = retrieval[
            "evidence"
        ]

        # ---------------------------------------------------------
        # NO EVIDENCE
        # ---------------------------------------------------------

        if not evidence:

            return {
                "route": decision.route,
                "answer": (
                    "The supplied documents do not contain "
                    "enough information to answer this reliably."
                ),
            }

        # ---------------------------------------------------------
        # CONFLICT ANALYSIS
        # ---------------------------------------------------------

        conflict_result = (
            self.conflict_detector.detect(
                evidence
            )
        )

        # ---------------------------------------------------------
        # GENERATION CONFIG
        # ---------------------------------------------------------

        generation_config = self.config.get(
            "generation",
            {},
        )

        max_context_chunks = (
            generation_config.get(
                "max_context_chunks",
                8,
            )
        )

        # ---------------------------------------------------------
        # PROMPT
        # ---------------------------------------------------------

        prompt = build_grounded_prompt(
            query=query,
            evidence=evidence[
                :max_context_chunks
            ],
            as_of_date=self.config.get(
                "as_of_date",
                "2026-08-27",
            ),
        )

        # ---------------------------------------------------------
        # GENERATION
        # ---------------------------------------------------------

        generation = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=generation_config.get(
                "temperature",
                0.0,
            ),
            max_tokens=generation_config.get(
                "max_tokens",
                220,
            ),
        )

        # ---------------------------------------------------------
        # CITATION VALIDATION
        # ---------------------------------------------------------

        citation_check = validate_citations(
            answer=generation["text"],
            evidence=evidence[
                :max_context_chunks
            ],
        )

        # ---------------------------------------------------------
        # DOCUMENTS USED
        # ---------------------------------------------------------

        documents_used = []

        for item in evidence[
            :max_context_chunks
        ]:

            document_id = (
                item["chunk"]
                .get("document_id")
            )

            if (
                document_id
                and document_id
                not in documents_used
            ):
                documents_used.append(
                    document_id
                )

        # ---------------------------------------------------------
        # FINAL RESPONSE
        # ---------------------------------------------------------

        return {
            "route": decision.route,

            "answer": generation[
                "text"
            ],

            "conflict": conflict_result,

            "citations": citation_check,

            "tokens": {
                "prompt": generation[
                    "prompt_tokens"
                ],
                "completion": generation[
                    "completion_tokens"
                ],
                "total": generation[
                    "total_tokens"
                ],
            },

            "generation_latency_sec": generation[
                "latency_sec"
            ],

            "documents_used": documents_used,

            # Helpful during development/evaluation.
            "evidence_chunk_ids": [
                item["chunk"]["chunk_id"]
                for item in evidence[
                    :max_context_chunks
                ]
            ],
        }