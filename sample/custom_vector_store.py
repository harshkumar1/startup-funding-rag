import uuid
from logging import Logger
from typing import Iterable, List, Union
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.schema import TextNode, NodeWithScore

from src.vector_storage.milvus_client import MilvusVectorClient


def _flatten(items: Iterable):
    for x in items:
        if isinstance(x, (list, tuple)):
            yield from _flatten(x)
        else:
            yield x


def _coerce_to_record(item: Union[TextNode, dict]) -> dict:
    """
    Convert TextNode or dict into a Milvus-ready record.
    Ensures `text` and `chunk_id`.
    """
    if isinstance(item, TextNode):
        txt = getattr(item, "text", None)
        metadata = getattr(item, "metadata", {}) or {}
        rec = {**metadata}
        rec.setdefault("text", txt)
    elif isinstance(item, dict):
        rec = dict(item)  # shallow copy
    else:
        raise TypeError(f"Unsupported item type: {type(item)}")

    # normalize text field
    text_val = rec.get("text") or rec.get("chunk_text")
    if not text_val or not isinstance(text_val, str) or not text_val.strip():
        return {}

    rec["text"] = text_val.strip()
    rec.setdefault("chunk_id", str(uuid.uuid4()))
    # return {k: v for k, v in rec.items() if k in ALLOWED_FIELDS}
    return rec  # Todo: why it has file_path


class CustomMilvusVectorStore(VectorStore):
    def __init__(
        self,
        logger: Logger,
        milvus_client: MilvusVectorClient,
        embed_model,
        batch_size: int = 64,
    ):
        """
        :param milvus_client: Your Milvus client wrapper
        :param embed_model: Embedding model (e.g. HuggingFaceEmbedding)
        :param batch_size: Batch insert size for Milvus
        """
        self.logger = logger
        self.milvus_client = milvus_client
        self.embed_model = embed_model
        self.batch_size = batch_size
        self.dimension = 768

    @property
    def stores_text(self) -> bool:
        """LlamaIndex uses this to decide if text is stored directly in Milvus."""
        return True  # since we always store text in 'text' field

    @property
    def stores_embedding(self) -> bool:
        """Indicates if embeddings are stored in Milvus."""
        return True  # since we store them in 'text_vector'

    def add(self, nodes) -> List[str]:
        """
        Accepts list of dict
        """
        try:
            if not nodes:
                self.logger.warning("add(): no nodes provided.")
                return []

            # Generate embeddings if missing, and drop any record whose
            # embedding ends up the wrong dimension (it would fail/corrupt
            # the batch insert otherwise).
            valid_nodes = []
            for node in nodes:
                if "text_vector" not in node or node["text_vector"] is None:
                    try:
                        node["text_vector"] = self.embed_model.get_text_embedding(
                            node["text"]
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Failed to embed text for record {node.get('chunk_id')}: {e}"
                        )
                        node["text_vector"] = [0.0] * self.dimension  # safe fallback

                # validate embedding dimension
                if len(node["text_vector"]) != self.dimension:
                    self.logger.error(
                        f"Embedding size mismatch for {node.get('chunk_id')} "
                        f"(got {len(node['text_vector'])}, expected {self.dimension}). "
                        f"Dropping record from this insert."
                    )
                    continue

                valid_nodes.append(node)

            nodes = valid_nodes
            if not nodes:
                self.logger.warning(
                    "add(): no valid nodes left after embedding/validation."
                )
                return []

            # Batch insert into Milvus
            ids: List[str] = []
            for i in range(0, len(nodes), self.batch_size):
                batch = nodes[i : i + self.batch_size]
                try:
                    inserted_ids = self.milvus_client.insert_nodes(nodes=batch)
                    if not inserted_ids:
                        self.logger.error(
                            f"Milvus insert returned no ids for batch {i // self.batch_size} "
                            f"(size {len(batch)}) - treating as failed, not counting as inserted."
                        )
                        continue
                    if len(inserted_ids) != len(batch):
                        self.logger.warning(
                            f"Milvus insert returned {len(inserted_ids)} ids for a batch of "
                            f"{len(batch)} - partial insert, verify data."
                        )
                    ids.extend(inserted_ids)
                    self.logger.info(
                        f"Inserted batch of {len(inserted_ids)} into Milvus."
                    )
                except Exception as e:
                    self.logger.error(
                        f"Milvus insert failed for batch {i//self.batch_size}: {e}"
                    )
            self.milvus_client.client.flush(
                collection_name=self.milvus_client.collection_name
            )
            self.logger.info(
                f"Flushed collection {self.milvus_client.collection_name} after batch insert."
            )

            # self.logger.info(f"Inserted total {len(ids)} records into Milvus.")
            return ids

        except Exception as e:
            self.logger.error(
                f"add(): fatal error inserting nodes into Milvus: {e}", exc_info=True
            )
            return []

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """
        Run similarity search on Milvus and return results as LlamaIndex-compatible Nodes.
        """
        try:
            if query.query_embedding is None:
                raise ValueError("Query embedding is missing in VectorStoreQuery.")

            hits = self.milvus_client.search(
                query_embedding=query.query_embedding, top_k=query.similarity_top_k
            )
            if not hits or not hits[0]:
                self.logger.warning("No results returned from Milvus search.")
                return VectorStoreQueryResult(nodes=[], ids=[], similarities=[])

            nodes: List[TextNode] = []
            ids: List[str] = []
            sims: List[float] = []
            for hit in hits:  # Milvus returns list[list[hit]]
                entity = hit.get("entity", hit)
                text = entity.get("text")
                if not text:
                    self.logger.warning(f"Skipping hit with no text: {entity}")
                    continue

                score = hit.get("score", 0.0)
                metadata = {k: v for k, v in entity.items() if k != "text"}

                node = TextNode(text=text, metadata=metadata)
                nodes.append(node)  # NodeWithScore(node=node, score=score))
                ids.append(hit.get("id", entity.get("chunk_id")))
                sims.append(score)

            self.logger.info(f"Retrieved {len(nodes)} nodes from Milvus.")
            return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=sims)

        except Exception as e:
            self.logger.error(f"Error during Milvus query: {e}", exc_info=True)
            return VectorStoreQueryResult(nodes=[], ids=[], similarities=[])
