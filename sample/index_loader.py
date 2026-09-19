"""
index_loader.py

CLI tool to load Milvus-backed VectorStoreIndex and start RetrievalService.
Wires in a reranker by default.
"""

import argparse
import logging

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

from src.vector_storage.milvus_client import MilvusVectorClient
from src.vector_storage.custom_vector_store import CustomMilvusVectorStore
from src.retrieve.config.retrieval_config import RetrievalConfig

# from retrieve.retrieval_service_old import make_service_from_index


# ---------------------------
# Logger setup
# ---------------------------
logger = logging.getLogger("index_loader")
if not logger.handlers:
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def build_index(
    collection_name: str,
    dim: int,
    host: str,
    port: str,
    embed_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex backed by Milvus + CustomMilvusVectorStore.
    """
    # Initialize embedder
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    # Milvus client
    milvus_client = MilvusVectorClient(
        logger=logger,
        collection_name=collection_name,
        dim=dim,
        host=host,
        port=port,
    )

    # Custom vector store
    vector_store = CustomMilvusVectorStore(
        logger=logger,
        milvus_client=milvus_client,
        embed_model=embed_model,
        batch_size=64,  # experiment: efficiency for higher batch size?
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index
    index = VectorStoreIndex.from_vector_store(
        storage_context=storage_context,
        vector_store=vector_store,
        embed_model=embed_model,
    )

    return index
