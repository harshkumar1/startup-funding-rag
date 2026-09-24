from __future__ import annotations

import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter

from ..common.embeddings import get_max_seq_length
from .source_repo import SourceDocument

TEXT_MAX_LEN = 15000


def get_chunk_size_and_overlap() -> tuple[int, int]:
    chunk_size = get_max_seq_length()
    overlap = max(32, chunk_size // 8)
    return chunk_size, overlap


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
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
