import hashlib
import re
from typing import Any

from transformers import AutoTokenizer


DEFAULT_TOKENIZER = "BAAI/bge-small-en-v1.5"


class DocumentChunker:

    SECTION_PATTERN = re.compile(
        r"^(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>.+)$"
    )

    def __init__(
        self,
        tokenizer_name: str = DEFAULT_TOKENIZER,
        target_tokens: int = 300,
        max_tokens: int = 450,
        overlap_tokens: int = 50,
        min_chunk_tokens: int = 40,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )

        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    def token_count(self, text: str) -> int:
        """
        Count tokens without passing text through the model.
        """
        return len(self.tokenizer.tokenize(text))
    
    def _merge_small_sections(
        self,
        sections: list[dict],
    ) -> list[dict]:
        """
        Merge abnormally small neighbouring sections where doing
        so remains safely below the target chunk size.
        """
        if not sections:
            return []

        merged = []

        for section in sections:

            text = section["text"]
            tokens = self.token_count(text)

            if (
                merged
                and tokens < self.min_chunk_tokens
            ):
                previous = merged[-1]

                combined_text = (
                    previous["text"]
                    + "\n\n"
                    + text
                )

                if (
                    self.token_count(combined_text)
                    <= self.target_tokens
                ):
                    previous["text"] = combined_text

                    if section.get("section"):
                        previous.setdefault(
                            "subsections", []
                        ).append(section["section"])

                    continue

            merged.append(section.copy())

        return merged

    @classmethod
    def _is_section_heading(cls, line: str) -> bool:
        """
        Detect likely document section headings while avoiding
        numbered procedural/list items.

        Examples accepted:
            1. Purpose
            3. Notice periods
            4.2 Annual leave entitlement
            2. Refund windows by plan

        Examples rejected:
            1. Day 30 check-in. Informal.
            2. Leave already taken in that year is deducted.
        """
        line = line.strip()

        match = cls.SECTION_PATTERN.match(line)

        if not match:
            return False

        title = match.group("title").strip()

        # Document headings in this corpus are short.
        if len(title) > 80:
            return False

        # Numbered procedural/list entries often contain full
        # sentences or multiple clauses.
        if ". " in title:
            return False

        # Very tiny numeric fragments are not useful sections.
        if len(title.split()) < 2:
            return False

        return True

    def _split_sections(
        self,
        text: str,
    ) -> list[dict[str, str | None]]:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        sections = []
        current_lines = []
        current_heading = None

        for line in lines:

            if self._is_section_heading(line):

                if current_lines:
                    sections.append(
                        {
                            "section": current_heading,
                            "text": "\n".join(current_lines),
                        }
                    )

                current_heading = line
                current_lines = [line]

            else:
                current_lines.append(line)

        if current_lines:
            sections.append(
                {
                    "section": current_heading,
                    "text": "\n".join(current_lines),
                }
            )

        return sections

    def _split_large_section(
        self,
        text: str,
    ) -> list[str]:
        """
        Split only when a logical section exceeds max_tokens.
        Prefer paragraph / line boundaries over arbitrary token cuts.
        """

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        pieces = []
        current = []

        for line in lines:

            candidate = "\n".join(current + [line])

            if (
                self.token_count(candidate) > self.max_tokens
                and current
            ):
                pieces.append("\n".join(current))

                current = self._build_overlap(current)
                current.append(line)

            else:
                current.append(line)

        if current:
            pieces.append("\n".join(current))

        return pieces

    def _build_overlap(
        self,
        lines: list[str],
    ) -> list[str]:

        overlap = []
        total = 0

        for line in reversed(lines):

            line_tokens = self.token_count(line)

            if total + line_tokens > self.overlap_tokens:
                break

            overlap.insert(0, line)
            total += line_tokens

        return overlap

    @staticmethod
    def _make_chunk_id(
        document_id: str,
        page: int,
        chunk_index: int,
        text: str,
    ) -> str:

        digest = hashlib.sha1(
            text.encode("utf-8")
        ).hexdigest()[:8]

        return (
            f"{document_id}"
            f"_p{page:02d}"
            f"_c{chunk_index:03d}"
            f"_{digest}"
        )

    def _build_chunk(
        self,
        page_record: dict[str, Any],
        text: str,
        section: str | None,
        chunk_index: int,
    ) -> dict[str, Any]:

        chunk = {
            key: value
            for key, value in page_record.items()
            if key != "text"
        }

        chunk["chunk_id"] = self._make_chunk_id(
            document_id=page_record["document_id"],
            page=page_record["page"],
            chunk_index=chunk_index,
            text=text,
        )

        chunk["section"] = section
        chunk["text"] = text
        chunk["token_count"] = self.token_count(text)

        return chunk

    def chunk_page(
        self,
        page_record: dict[str, Any],
    ) -> list[dict[str, Any]]:

        sections = self._split_sections(
            page_record["text"]
        )

        chunks = []
        chunk_index = 1

        for section in sections:

            section_text = section["text"]
            section_heading = section["section"]

            if self.token_count(section_text) <= self.max_tokens:

                chunks.append(
                    self._build_chunk(
                        page_record=page_record,
                        text=section_text,
                        section=section_heading,
                        chunk_index=chunk_index,
                    )
                )

                chunk_index += 1

            else:

                pieces = self._split_large_section(
                    section_text
                )

                for piece in pieces:

                    chunks.append(
                        self._build_chunk(
                            page_record=page_record,
                            text=piece,
                            section=section_heading,
                            chunk_index=chunk_index,
                        )
                    )

                    chunk_index += 1

        return chunks