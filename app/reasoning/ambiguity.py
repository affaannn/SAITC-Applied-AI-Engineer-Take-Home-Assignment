class AmbiguityDetector:

    AMBIGUOUS_PATTERNS = {
        "what is the limit?",
        "what's the limit?",
        "what is the maximum?",
    }

    def is_ambiguous(
        self,
        query: str,
    ) -> bool:

        normalized = (
            query.strip()
            .lower()
        )

        return normalized in self.AMBIGUOUS_PATTERNS