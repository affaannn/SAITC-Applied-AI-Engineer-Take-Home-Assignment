import re


CITATION_PATTERN = re.compile(
    r"\[([A-Z0-9\-]+)\s+p\.(\d+)\]"
)


def extract_citations(
    text: str,
) -> list[tuple[str, int]]:

    matches = CITATION_PATTERN.findall(text)

    return [
        (document_id, int(page))
        for document_id, page in matches
    ]


def validate_citations(
    answer: str,
    evidence: list[dict],
) -> dict:

    cited = extract_citations(answer)

    allowed = {
        (
            item["chunk"]["document_id"],
            int(item["chunk"]["page"]),
        )
        for item in evidence
    }

    invalid = [
        citation
        for citation in cited
        if citation not in allowed
    ]

    return {
        "citations": cited,
        "invalid_citations": invalid,
        "valid": len(invalid) == 0,
    }