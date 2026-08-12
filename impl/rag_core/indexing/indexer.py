from __future__ import annotations

import logging

from ..common.config import load_settings
from ..common.embeddings import embed_texts, get_vector_dim
from ..common.vector_store import get_client
from .chunking import build_records, chunk_text, get_chunk_size_and_overlap
from .schema import create_vector_index, recreate_collection
from .source_repo import load_all_documents

logger = logging.getLogger(__name__)

_INSERT_BATCH_SIZE = 500


def index_all() -> dict:
    logger.info("Ingest started.")
    settings = load_settings()
    vector_dim = get_vector_dim()
    chunk_size, overlap = get_chunk_size_and_overlap()
    logger.info(
        "Resolved embedding config: vector_dim=%d chunk_size=%d overlap=%d",
        vector_dim,
        chunk_size,
        overlap,
    )

    logger.info("Recreating collection %r...", settings.collection_name)
    recreate_collection(vector_dim)

    logger.info("Fetching source documents...")
    documents = load_all_documents()
    logger.info("Fetched %d source documents.", len(documents))

    logger.info("Chunking %d documents...", len(documents))
    records: list[dict] = []
    for document in documents:
        chunks = chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        records.extend(build_records(document, chunks))
    logger.info("Produced %d chunks from %d documents.", len(records), len(documents))

    if not records:
        raise RuntimeError("No chunks produced from source documents.")

    logger.info("Embedding %d chunks...", len(records))
    texts = [record["text"] for record in records]
    vectors = embed_texts(texts)
    for record, vector in zip(records, vectors):
        record["text_vector"] = vector

    client = get_client()
    total_batches = (len(records) + _INSERT_BATCH_SIZE - 1) // _INSERT_BATCH_SIZE
    logger.info("Inserting %d chunks in %d batch(es)...", len(records), total_batches)
    for batch_num, start in enumerate(range(0, len(records), _INSERT_BATCH_SIZE), start=1):
        batch = records[start : start + _INSERT_BATCH_SIZE]
        client.insert(collection_name=settings.collection_name, data=batch)
        logger.info("Inserted batch %d/%d (%d records).", batch_num, total_batches, len(batch))

    logger.info("Creating vector index...")
    create_vector_index()

    logger.info(
        "Ingest complete: %d documents, %d chunks.", len(documents), len(records)
    )
    return {
        "collection": settings.collection_name,
        "documents_ingested": len(documents),
        "chunks_inserted": len(records),
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
    }
