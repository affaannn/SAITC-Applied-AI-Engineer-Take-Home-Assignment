import csv
import json
import time
from pathlib import Path
from typing import Any

from app.pipeline import RAGPipeline


QUESTIONS_PATH = Path(
    "evaluation/questions.json"
)

RESULTS_CSV_PATH = Path(
    "evaluation/results.csv"
)

RESULTS_JSON_PATH = Path(
    "evaluation/results.json"
)


def load_questions() -> list[dict[str, Any]]:
    with QUESTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_text(
    value: str,
) -> str:
    return (
        value
        .lower()
        .replace(",", "")
        .strip()
    )


def check_expected_content(
    answer: str,
    expected_values: list[str],
) -> bool:
    """
    Lightweight smoke-test check.

    This is intentionally not treated as a full factual
    correctness metric. It only verifies that at least one
    expected signal is present.
    """

    if not expected_values:
        return True

    normalized_answer = normalize_text(
        answer
    )

    return any(
        normalize_text(value)
        in normalized_answer
        for value in expected_values
    )


def check_expected_documents(
    documents_used: list[str],
    expected_docs: list[str],
) -> tuple[bool, float]:
    """
    Calculate expected-document recall.

    Example:
    expected = [A, B]
    retrieved = [A]
    recall = 0.5
    """

    if not expected_docs:
        return True, 1.0

    found = set(
        documents_used
    )

    expected = set(
        expected_docs
    )

    matched = (
        expected
        & found
    )

    recall = (
        len(matched)
        / len(expected)
    )

    return (
        recall == 1.0,
        recall,
    )


def check_route(
    actual_route: str,
    expected_routes: list[str],
) -> bool:

    if not expected_routes:
        return True

    return (
        actual_route
        in expected_routes
    )


def get_conflict_flag(
    result: dict,
) -> bool:

    conflict = result.get(
        "conflict",
        {}
    )

    if isinstance(
        conflict,
        dict,
    ):
        return bool(
            conflict.get(
                "conflict_detected",
                False,
            )
        )

    return False


def evaluate_question(
    pipeline: RAGPipeline,
    item: dict[str, Any],
) -> dict[str, Any]:

    question = item[
        "question"
    ]

    print()
    print("=" * 100)
    print(
        f"{item['id']} | "
        f"{item['category']}"
    )
    print("=" * 100)
    print(question)
    print()

    total_start = time.perf_counter()

    try:
        result = pipeline.answer(
            question
        )

        total_latency = (
            time.perf_counter()
            - total_start
        )

        answer = result.get(
            "answer",
            "",
        )

        route = result.get(
            "route",
            "unknown",
        )

        documents_used = result.get(
            "documents_used",
            [],
        )

        tokens = result.get(
            "tokens",
            {
                "prompt": 0,
                "completion": 0,
                "total": 0,
            },
        )

        citation_info = result.get(
            "citations",
            {},
        )

        citation_valid = (
            citation_info.get(
                "valid",
                True,
            )
            if isinstance(
                citation_info,
                dict,
            )
            else True
        )

        content_check = (
            check_expected_content(
                answer=answer,
                expected_values=item.get(
                    "must_contain_any",
                    [],
                ),
            )
        )

        docs_ok, doc_recall = (
            check_expected_documents(
                documents_used=documents_used,
                expected_docs=item.get(
                    "expected_docs",
                    [],
                ),
            )
        )

        route_ok = check_route(
            actual_route=route,
            expected_routes=item.get(
                "expected_route",
                [],
            ),
        )

        conflict_detected = (
            get_conflict_flag(
                result
            )
        )

        smoke_pass = (
            content_check
            and docs_ok
            and route_ok
            and citation_valid
        )

        print(
            f"Route       : {route}"
        )
        print(
            f"Documents   : {documents_used}"
        )
        print(
            f"Doc recall  : {doc_recall:.2f}"
        )
        print(
            f"Citations OK: {citation_valid}"
        )
        print(
            f"Content check: {content_check}"
        )
        print(
            f"Smoke pass  : {smoke_pass}"
        )

        print()
        print("ANSWER")
        print("-" * 100)
        print(answer)

        print()
        print(
            "Tokens       : "
            f"{tokens}"
        )

        print(
            f"Total latency: "
            f"{total_latency:.2f}s"
        )

        return {
            "id": item[
                "id"
            ],
            "category": item[
                "category"
            ],
            "question": question,
            "route": route,
            "expected_route": " | ".join(
                item.get(
                    "expected_route",
                    [],
                )
            ),
            "answer": answer,
            "documents_used": " | ".join(
                documents_used
            ),
            "expected_docs": " | ".join(
                item.get(
                    "expected_docs",
                    [],
                )
            ),
            "document_recall": round(
                doc_recall,
                3,
            ),
            "route_check": route_ok,
            "content_check": (
                content_check
            ),
            "citation_valid": (
                citation_valid
            ),
            "conflict_detected": (
                conflict_detected
            ),
            "prompt_tokens": tokens.get(
                "prompt",
                0,
            ),
            "completion_tokens": (
                tokens.get(
                    "completion",
                    0,
                )
            ),
            "total_tokens": tokens.get(
                "total",
                0,
            ),
            "generation_latency_sec": (
                result.get(
                    "generation_latency_sec",
                    0.0,
                )
            ),
            "total_latency_sec": round(
                total_latency,
                3,
            ),
            "smoke_pass": smoke_pass,
            "error": "",
        }

    except Exception as exc:

        total_latency = (
            time.perf_counter()
            - total_start
        )

        print(
            f"ERROR: {exc}"
        )

        return {
            "id": item[
                "id"
            ],
            "category": item[
                "category"
            ],
            "question": question,
            "route": "error",
            "expected_route": " | ".join(
                item.get(
                    "expected_route",
                    [],
                )
            ),
            "answer": "",
            "documents_used": "",
            "expected_docs": " | ".join(
                item.get(
                    "expected_docs",
                    [],
                )
            ),
            "document_recall": 0.0,
            "route_check": False,
            "content_check": False,
            "citation_valid": False,
            "conflict_detected": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "generation_latency_sec": 0.0,
            "total_latency_sec": round(
                total_latency,
                3,
            ),
            "smoke_pass": False,
            "error": str(exc),
        }


def save_csv(
    results: list[dict[str, Any]],
) -> None:

    RESULTS_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not results:
        return

    fieldnames = list(
        results[0].keys()
    )

    with RESULTS_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


def save_json(
    results: list[dict[str, Any]],
) -> None:

    with RESULTS_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_summary(
    results: list[dict[str, Any]],
) -> None:

    total = len(
        results
    )

    passed = sum(
        bool(
            result[
                "smoke_pass"
            ]
        )
        for result in results
    )

    citation_valid = sum(
        bool(
            result[
                "citation_valid"
            ]
        )
        for result in results
    )

    average_document_recall = (
        sum(
            float(
                result[
                    "document_recall"
                ]
            )
            for result in results
        )
        / total
        if total
        else 0.0
    )

    total_tokens = sum(
        int(
            result[
                "total_tokens"
            ]
        )
        for result in results
    )

    generated_questions = [
        result
        for result in results
        if int(
            result[
                "total_tokens"
            ]
        ) > 0
    ]

    average_generated_tokens = (
        total_tokens
        / len(
            generated_questions
        )
        if generated_questions
        else 0.0
    )

    average_latency = (
        sum(
            float(
                result[
                    "total_latency_sec"
                ]
            )
            for result in results
        )
        / total
        if total
        else 0.0
    )

    print()
    print("=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)

    print(
        f"Questions              : {total}"
    )

    print(
        f"Smoke tests passed     : "
        f"{passed}/{total}"
    )

    print(
        f"Citation-valid answers : "
        f"{citation_valid}/{total}"
    )

    print(
        f"Average document recall: "
        f"{average_document_recall:.3f}"
    )

    print(
        f"Total generation tokens: "
        f"{total_tokens}"
    )

    print(
        f"Avg tokens/generated Q : "
        f"{average_generated_tokens:.1f}"
    )

    print(
        f"Average total latency  : "
        f"{average_latency:.2f}s"
    )

    print()
    print(
        f"CSV  : {RESULTS_CSV_PATH}"
    )

    print(
        f"JSON : {RESULTS_JSON_PATH}"
    )

    print("=" * 100)


def main():

    questions = load_questions()

    print("=" * 100)
    print("SAITC RAG EVALUATION")
    print("=" * 100)
    print(
        f"Questions: {len(questions)}"
    )

    print()
    print(
        "Loading pipeline..."
    )

    pipeline = RAGPipeline()

    print(
        "Pipeline ready."
    )

    results = []

    for item in questions:

        result = evaluate_question(
            pipeline=pipeline,
            item=item,
        )

        results.append(
            result
        )

        # Save incrementally so a slow/crashed model
        # does not lose earlier results.
        save_csv(
            results
        )

        save_json(
            results
        )

    print_summary(
        results
    )


if __name__ == "__main__":
    main()