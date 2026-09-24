from __future__ import annotations

import logging

from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

from ..common.config import load_settings
from ..common.embeddings import embed_text
from ..common.vector_store import get_client
from .rerank import rerank_chunks

logger = logging.getLogger(__name__)

_OUTPUT_FIELDS = [
    "chunk_id",
    "document_id",
    "chunk_order",
    "canonical_url",
    "scheme_type",
    "scheme_name",
    "content_type",
    "language",
    "text",
]


def _run_search(
    client: MilvusClient, collection_name: str, query_vector: list[float], top_k: int
) -> list:
    return client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=top_k,
        filter="is_active == true",
        output_fields=_OUTPUT_FIELDS,
    )


def search_chunks(query_vector: list[float], top_k: int) -> list[dict]:
    settings = load_settings()
    client = get_client()

    try:
        results = _run_search(client, settings.collection_name, query_vector, top_k)
    except MilvusException as exc:
        if "not loaded" not in str(exc).lower():
            raise
        logger.warning(
            "Collection %r not loaded (likely auto-released after "
            "inactivity by Zilliz); loading and retrying search once.",
            settings.collection_name,
        )
        client.load_collection(settings.collection_name)
        results = _run_search(client, settings.collection_name, query_vector, top_k)

    hits = results[0] if results else []
    return [{"score": hit["distance"], **hit["entity"]} for hit in hits]


def retrieve_chunks(query: str, top_k: int) -> list[dict]:
    """
    Embed, vector-search a wider candidate set, then rerank down to top_k.

    Same pattern as sample/grok_retrieval_service.py: fetch cfg.top_k (8)
    hits, then keep rerank_top_n. Here the API `top_k` is the count after
    rerank; candidate width comes from RERANK_CANDIDATES (default 8).
    Rerank failures fall back to the original vector order (sample behavior).
    """
    settings = load_settings()
    vector = embed_text(query)
    candidate_k = top_k
    if settings.rerank:
        candidate_k = max(settings.rerank_candidates, top_k)

    chunks = search_chunks(vector, top_k=candidate_k)
    if not settings.rerank or not chunks:
        return chunks[:top_k]

    before_ids = [chunk.get("chunk_id") for chunk in chunks]
    logger.info("Rerank BEFORE (vector order, %d hits): %s", len(chunks), before_ids)
    try:
        reranked = rerank_chunks(query, chunks, top_n=top_k)
    except Exception:
        logger.exception("Reranking failed, using original vector hits")
        return chunks[:top_k]

    after_ids = [chunk.get("chunk_id") for chunk in reranked]
    logger.info("Rerank AFTER (cross-encoder order, %d hits): %s", len(reranked), after_ids)
    return reranked
