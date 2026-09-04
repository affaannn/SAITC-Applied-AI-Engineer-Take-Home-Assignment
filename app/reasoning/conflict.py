import re
from datetime import date
from typing import Any


MONEY_PATTERN = re.compile(
    r"SAR\s?[\d,]+(?:\.\d+)?",
    re.IGNORECASE,
)

DAYS_PATTERN = re.compile(
    r"\b\d+\s+(?:calendar|working)?\s*days?\b",
    re.IGNORECASE,
)


class ConflictDetector:

    def detect(
        self,
        evidence: list[dict[str, Any]],
    ) -> dict:

        values = []

        for item in evidence:
            chunk = item["chunk"]
            text = chunk["text"]

            money_values = MONEY_PATTERN.findall(text)
            day_values = DAYS_PATTERN.findall(text)

            extracted = money_values + day_values

            for value in extracted:
                values.append(
                    {
                        "value": value,
                        "document_id": chunk["document_id"],
                        "effective_date": chunk.get(
                            "effective_date"
                        ),
                        "chunk_id": chunk["chunk_id"],
                    }
                )

        unique_values = {
            item["value"].lower()
            for item in values
        }

        return {
            "conflict_detected": len(unique_values) > 1,
            "candidate_values": values,
        }
    
class VersionResolver:

    @staticmethod
    def resolve_latest(
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        def effective_date(item):
            value = (
                item["chunk"]
                .get("effective_date")
            )

            if not value:
                return ""

            return value

        return sorted(
            evidence,
            key=effective_date,
            reverse=True,
        )