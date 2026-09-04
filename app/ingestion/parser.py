from pathlib import Path
from typing import Any
import pymupdf

class PDFParsingError(Exception):
    """Raised when a PDF cannot be parsed."""


def clean_text(text: str) -> str:
    """
    Apply conservative whitespace cleaning.

    Do NOT aggressively rewrite document content because
    formatting and wording may matter for retrieval.
    """
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)


def parse_pdf(
    pdf_path: str | Path,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Parse a PDF into page-level records.

    Each page inherits document-level metadata from the
    corpus manifest.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise PDFParsingError(
            f"PDF not found: {pdf_path}"
        )

    pages = []

    try:
        document = pymupdf.open(pdf_path)

        for page_number, page in enumerate(document, start=1):
            raw_text = page.get_text("text")
            text = clean_text(raw_text)

            if not text:
                continue

            pages.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text,

                    "document_id": metadata["document_id"],
                    "title": metadata["title"],
                    "version": metadata["version"],
                    "effective_date": (
                        metadata["effective_date"].isoformat()
                    ),
                    "owner": metadata["owner"],
                    "classification": metadata["classification"],
                    "supersedes": metadata.get("supersedes"),
                }
            )

        document.close()

    except Exception as exc:
        raise PDFParsingError(
            f"Failed to parse {pdf_path.name}: {exc}"
        ) from exc

    return pages