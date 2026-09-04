import json
from pathlib import Path
import yaml
import time
import statistics

from app.ingestion.metadata import (
    build_document_index,
    load_manifest,
)
from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import DocumentChunker


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    start_time = time.perf_counter()
    config = load_config()

    data_dir = Path(
        config["corpus"]["data_dir"]
    )

    manifest_path = Path(
        config["corpus"]["manifest"]
    )

    output_path = Path(
        config["output"]["chunks_file"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest(manifest_path)

    document_index = build_document_index(
        manifest
    )

    chunker = DocumentChunker(
        tokenizer_name=config["chunking"]["tokenizer"],
        target_tokens=config["chunking"]["target_tokens"],
        max_tokens=config["chunking"]["max_tokens"],
        overlap_tokens=config["chunking"]["overlap_tokens"],
        min_chunk_tokens=config["chunking"]["min_tokens"],
    )

    chunks = []

    for filename, metadata in document_index.items():

        pdf_path = data_dir / filename

        print(f"Ingesting: {filename}")

        pages = parse_pdf(
            pdf_path=pdf_path,
            metadata=metadata,
        )

        for page in pages:
            page_chunks = chunker.chunk_page(page)
            chunks.extend(page_chunks)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for chunk in chunks:

            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )

    token_counts = [chunk["token_count"] for chunk in chunks]
    tiny_chunks = sum(
    count < config["chunking"]["min_tokens"]
    for count in token_counts
    )

    small_chunks = sum(
        40 <= count < 100
        for count in token_counts
    )

    medium_chunks = sum(
        100 <= count <= 300
        for count in token_counts
    )

    large_chunks = sum(
        count > 300
        for count in token_counts
    )

    print()
    print("=" * 45)
    print("INGESTION SUMMARY")
    print("=" * 45)

    print(f"Documents        : {len(document_index)}")
    print(f"Pages processed  : {sum(len(parse_pdf(data_dir / f, m)) for f, m in document_index.items())}")
    print(f"Chunks generated : {len(chunks)}")

    print()
    print("Token statistics")
    print(f"Min tokens       : {min(token_counts)}")
    print(f"Max tokens       : {max(token_counts)}")
    print(
        f"Average tokens   : "
        f"{sum(token_counts) / len(token_counts):.1f}"
    )

    oversized = sum(
        count > config["chunking"]["max_tokens"]
        for count in token_counts
    )

    print(f"Oversized chunks : {oversized}")
    print("Chunk distribution")
    print(f"< 40 tokens      : {tiny_chunks}")
    print(f"40–99 tokens     : {small_chunks}")
    print(f"100–300 tokens   : {medium_chunks}")
    print(f"> 300 tokens      : {large_chunks}")
    print(
    f"Median tokens    : "
    f"{statistics.median(token_counts):.1f}")

    print()
    print(f"Output            : {output_path}")
    elapsed = time.perf_counter() - start_time

    print(f"Ingestion time   : {elapsed:.2f} sec")
    print("=" * 45)


if __name__ == "__main__":
    main()