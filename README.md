# SAITC-Applied-AI-Engineer-Take-Home-Assignment

## Implementation

### 1. Corpus ingestion
The corpus contains 13 text-based PDFs and a machine-readable
metadata manifest.

The ingestion pipeline uses PyMuPDF for text extraction and
`corpus_manifest.json` as the primary source of document metadata.

Metadata including document ID, version, effective date, owner,
classification, and supersession information is attached to every
retrievable chunk. This allows downstream retrieval and reasoning
components to distinguish newer and superseded documents.

The assignment reference date is fixed through configuration as
`2026-08-27`, matching the corpus instructions.

Retrieved document text is preserved as-is during ingestion,
including text that resembles instructions to an AI assistant.
Such content is treated as untrusted evidence rather than executable
instructions by the downstream safety and generation layers.

Chunking is section-aware with a bounded token size rather than
pure fixed-width splitting. This was chosen to preserve policy
clauses, tables, headings, and nearby explanatory text.

#### 2. Chunking iteration

The initial ingestion implementation produced 26 chunks for the
13-document corpus, effectively resulting in approximately one chunk
per PDF page.

Inspection showed that this grouped multiple independent policy
sections into the same embedding unit. For example, annual leave
entitlement, accrual, requesting leave and carry-over could appear
inside a single chunk.

The chunking strategy was therefore changed to section-first
chunking. Numbered document sections are preserved as independent
retrieval units where possible, with token-based splitting used only
when an individual section exceeds the configured maximum size.

The maximum chunk size is kept below the embedding model's context
limit to avoid silent truncation during embedding generation.

Repeated page headers, footers and metadata blocks are excluded from
semantic text because equivalent values are stored as structured
metadata. Deliberately adversarial document content is not removed.

#### 3. Chunk granularity refinement

The first section-aware implementation over-segmented the corpus,
producing 169 chunks with an average size of only 46.4 tokens and
some chunks containing as few as two tokens.

Inspection indicated that numbered procedural steps were sometimes
being interpreted as document section headings. Heading detection
was therefore tightened and a conservative minimum chunk threshold
was introduced.

Very small neighbouring fragments are merged only when the resulting
chunk remains below the target size. Legitimately concise clauses
and tables are still preserved independently where they represent a
useful retrieval unit.

This resulted from iterative inspection of the corpus rather than
selecting a fixed chunk size in isolation.

Known limitation:
  Table / cross-page fragmentation
  - handled downstream using retrieval expansion
<p align="center">
  <a href="[https://YOUR_USERNAME.github.io/YOUR_REPO/rag_pipeline_flowchart.html](https://github.com/affaannn/SAITC-Applied-AI-Engineer-Take-Home-Assignment/blob/main/rag_pipeline_flowchart.html">
    <img src="./docs/rag_pipeline_flowchart.svg" alt="Hybrid RAG Assistant Pipeline" width="900" />
  </a>
</p>
