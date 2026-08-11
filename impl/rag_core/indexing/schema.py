"""
`rag_chunks` collection schema + index management (Zilliz Cloud).

Mirrors playground/schema_design/schema_design_notebook.ipynb sec 7 (schema)
and sec 12 (AUTOINDEX + COSINE index). `text_vector` dim is passed in by the
caller, read from the embedding model at runtime — never hardcoded.
"""

from __future__ import annotations

import logging

from pymilvus import DataType, MilvusClient

from ..common.config import load_settings
from ..common.vector_store import get_client

logger = logging.getLogger(__name__)


def _build_schema(vector_dim: int) -> MilvusClient:
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=36, description="Chunk Id")
    schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=36, description="Document Id")
    schema.add_field(field_name="chunk_order", datatype=DataType.INT32, description="Chunk order number helps in reconstructing document sequence")
    schema.add_field(field_name="base_url", datatype=DataType.VARCHAR, max_length=512, description="Base url of the source website")
    schema.add_field(field_name="canonical_url", datatype=DataType.VARCHAR, max_length=512, description="Current url of the document")
    schema.add_field(field_name="crawl_date", datatype=DataType.INT64, description="Date when the document was fetched")
    schema.add_field(field_name="doc_last_modified", datatype=DataType.INT64, description="Document Last Modified")
    schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=20, description="Type of data whether it is text/image/mixed")
    schema.add_field(field_name="content_source_type", datatype=DataType.VARCHAR, max_length=50, description="Type of source e.g., webpage, PDF, announcement")
    schema.add_field(field_name="scheme_type", datatype=DataType.VARCHAR, max_length=50, description="Whether it is govt scheme or angel investor etc")
    schema.add_field(field_name="scheme_name", datatype=DataType.VARCHAR, max_length=50, description="Name of govt scheme or investor")
    schema.add_field(field_name="language", datatype=DataType.VARCHAR, max_length=15, description="Language of the document")
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=15000, description="Chunk original text")
    schema.add_field(field_name="text_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim, description="Chunk embedding vector")
    schema.add_field(field_name="doc_version", datatype=DataType.VARCHAR, max_length=5, description="Version of the document")
    schema.add_field(field_name="is_active", datatype=DataType.BOOL, description="Flag to mark if the document is active")
    return schema


def recreate_collection(vector_dim: int) -> None:
    """Drop the collection if it exists, then recreate it with a fresh schema.

    WARNING: destructive — deletes all existing indexed data.

    Args:
        vector_dim: Embedding dimension for `text_vector`, read from the
            embedding model at runtime (see embeddings.get_vector_dim).
    """
    settings = load_settings()
    client = get_client()

    if client.has_collection(settings.collection_name):
        logger.info("Dropping existing collection %r...", settings.collection_name)
        client.drop_collection(settings.collection_name)

    logger.info(
        "Creating collection %r (vector_dim=%d)...", settings.collection_name, vector_dim
    )
    client.create_collection(
        collection_name=settings.collection_name,
        schema=_build_schema(vector_dim),
    )


def create_vector_index() -> None:
    """Create the AUTOINDEX + COSINE index on `text_vector` (Zilliz Cloud
    recommendation) if one doesn't already exist, then load the collection
    into memory so it's immediately searchable.

    Loading is done unconditionally (even when the index already existed)
    since Zilliz Cloud (serverless/free tier) auto-releases idle collections
    from memory to save resources — an existing index doesn't guarantee the
    collection is currently loaded. `load_collection` is a cheap no-op if
    it's already loaded. (`retrieval/search.py` also self-heals by loading
    on-demand if a search hits a "not loaded" error later, e.g. after a long
    idle period post-indexing.)
    """
    settings = load_settings()
    client = get_client()

    if client.list_indexes(settings.collection_name):
        logger.info(
            "Index already exists on %r, skipping.", settings.collection_name
        )
    else:
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="text_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        client.create_index(collection_name=settings.collection_name, index_params=index_params)
        logger.info("Created AUTOINDEX/COSINE index on %r.text_vector.", settings.collection_name)

    logger.info("Loading collection %r into memory...", settings.collection_name)
    client.load_collection(settings.collection_name)
