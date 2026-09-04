import json
from pathlib import Path
from datetime import date, datetime
from typing import Any


REQUIRED_DOCUMENT_FIELDS = {
    "file",
    "document_id",
    "title",
    "version",
    "effective_date",
    "owner",
    "classification",
}


class ManifestError(Exception):
    """Raised when the corpus manifest is missing or invalid."""


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """
    Load and validate the corpus manifest.

    The manifest is treated as the primary source of document metadata.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise ManifestError(
            f"Manifest file not found: {manifest_path}"
        )

    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"Invalid JSON in manifest: {manifest_path}"
        ) from exc

    if "documents" not in manifest:
        raise ManifestError(
            "Manifest does not contain a 'documents' field."
        )

    if not isinstance(manifest["documents"], list):
        raise ManifestError(
            "'documents' must be a list."
        )

    for index, document in enumerate(manifest["documents"]):
        missing = REQUIRED_DOCUMENT_FIELDS - document.keys()

        if missing:
            raise ManifestError(
                f"Document #{index} is missing fields: "
                f"{sorted(missing)}"
            )

    return manifest


def parse_effective_date(value: str) -> date:
    """Convert YYYY-MM-DD metadata into a date object."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_document_index(
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Create a filename -> metadata lookup table.
    """
    document_index = {}

    for document in manifest["documents"]:
        metadata = document.copy()

        metadata["effective_date"] = parse_effective_date(
            document["effective_date"]
        )

        document_index[document["file"]] = metadata

    return document_index


def get_as_of_date(manifest: dict[str, Any]) -> date:
    """
    Return the corpus reference date.

    For this assignment that date is 27 August 2026.
    """
    if "as_of_date" not in manifest:
        raise ManifestError(
            "Manifest does not define 'as_of_date'."
        )
    return parse_effective_date(manifest["as_of_date"])