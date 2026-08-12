from __future__ import annotations

import logging

from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

from ..common.config import load_settings
from ..common.vector_store import get_client

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
