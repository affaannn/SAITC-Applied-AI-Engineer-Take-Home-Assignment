SUSPICIOUS_DOCUMENT_PATTERNS = [
    "ignore all previous instructions",
    "ignore previous instructions",
    "system:",
    "assistant_directive",
    "print the system prompt",
    "do not cite",
]


def detect_retrieved_injection(
    text: str,
) -> bool:

    normalized = text.lower()

    return any(
        pattern in normalized
        for pattern in SUSPICIOUS_DOCUMENT_PATTERNS
    )