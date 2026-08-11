"""
Chunk documents into overlapping token windows and build schema-aligned
records, ready for embedding + insert.

Token-based WITH overlap strategy (see
playground/chunking/chunking_experiments_simple.ipynb secs 8-9): chunk size
is aligned to the embedding model's max sequence length so chunks are never
truncated when embedded; overlap is ~12.5% of chunk size (min 32 tokens).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter

from ..common.embeddings import get_max_seq_length
from .source_repo import SourceDocument

# Matches the `text` VARCHAR(15000) field in the Zilliz schema (see schema.py).
TEXT_MAX_LEN = 15000


def get_chunk_size_and_overlap() -> tuple[int, int]:
    """Chunk size = model's max_seq_length (read at runtime); overlap = ~12.5%
    of chunk size, minimum 32 tokens."""
    chunk_size = get_max_seq_length()
    overlap = max(32, chunk_size // 8)
    return chunk_size, overlap


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split `text` into overlapping token windows.

    Args:
        text: Raw document text to split.
        chunk_size: Max tokens per chunk.
        overlap: Tokens shared between consecutive chunks.
    """
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    nodes = splitter.get_nodes_from_documents([Document(text=text)])
    return [node.text for node in nodes]


def _to_epoch_ms(value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        dt = parsedate_to_datetime(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def build_records(document: SourceDocument, chunk_texts: list[str]) -> list[dict]:
    """Build schema-aligned records for one document's chunks.

    `text_vector` is intentionally omitted here — it's attached later, once
    all chunks across all documents are embedded together in one batch.

    Args:
        document: The source document the chunks were split from.
        chunk_texts: Ordered chunk texts for this document.
    """
    meta = document.meta
    content_type = (meta.get("content_type") or "").split(";")[0].strip()[:20]

    records = []
    for order, text in enumerate(chunk_texts):
        records.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "document_id": document.doc_id,
                "chunk_order": int(order),
                "base_url": meta.get("base_url", ""),
                "canonical_url": meta.get("canonical_url", ""),
                "crawl_date": _to_epoch_ms(meta.get("crawl_dt")),
                "doc_last_modified": _to_epoch_ms(meta.get("doc_last_modified_dt")),
                "content_type": content_type,
                "content_source_type": meta.get("content_source_type", ""),
                "scheme_type": meta.get("scheme_type", ""),
                "scheme_name": meta.get("scheme_name", ""),
                "language": meta.get("lang", ""),
                "text": (text or "")[:TEXT_MAX_LEN],
                "doc_version": str(meta.get("doc_version", "")),
                "is_active": _as_bool(meta.get("is_active", True)),
            }
        )
    return records
