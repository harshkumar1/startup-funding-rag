from logging import Logger
from typing import List, Dict, Any, Optional
from pymilvus import (
    MilvusClient,
    Collection,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    MilvusException,
    connections,
)

# from vector_storage.storage_common.base_client import BaseVectorClient
# from vector_storage.storage_common.exception_handler import handle_milvus_exception

# Aligned with playground notebooks. In notebooks prefer reading dim from the loaded model.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class MilvusStorage:
    """Handles schema and index creation for Milvus."""

    @staticmethod
    def create_schema():
        schema_fields = [
            # Identification details
            FieldSchema(
                name="chunk_id",
                dtype=DataType.VARCHAR,
                description="Chunk Id",
                is_primary=True,
                max_length=36,
            ),
            FieldSchema(
                name="document_id",
                dtype=DataType.VARCHAR,
                description="Document Id",
                max_length=36,
            ),
            # Ordernal details ##base_url = https://seedfund.startupindia.gov.in,  canonical_url = https://seedfund.startupindia.gov.in/about.html
            FieldSchema(
                name="chunk_order",
                dtype=DataType.INT32,
                description="Chunk order number helps in reconstructing document sequence",
            ),
            # Document tracing
            FieldSchema(
                name="base_url",
                dtype=DataType.VARCHAR,
                description="Base url of the source website",
                max_length=512,
            ),
            FieldSchema(
                name="canonical_url",
                dtype=DataType.VARCHAR,
                description="Current url of the document",
                max_length=512,
            ),
            # Temporal metadata
            FieldSchema(
                name="crawl_date",
                dtype=DataType.INT64,
                description="Date when the document was fetched",
            ),
            FieldSchema(
                name="doc_last_modified",
                dtype=DataType.INT64,
                description="Document Last Modified",
            ),
            # Content Classification
            FieldSchema(
                name="content_type",
                dtype=DataType.VARCHAR,
                description="Type of data whether it is text/image/mixed",
                max_length=20,
            ),
            FieldSchema(
                name="content_source_type",
                dtype=DataType.VARCHAR,
                description="Type of source e.g., webpage, PDF, announcement",
                max_length=50,
            ),
            FieldSchema(
                name="scheme_type",
                dtype=DataType.VARCHAR,
                description="Whether it is govt scheme or angel investor etc",
                max_length=50,
            ),
            FieldSchema(
                name="scheme_name",
                dtype=DataType.VARCHAR,
                description="Name of govt scheme or investor",
                max_length=50,
            ),
            # FieldSchema(
            #                 name='funding_stage',
            #                 dtype=DataType.ARRAY,
            #                 element_type=DataType.VARCHAR,
            #                 description=(
            #                     'Stage of funding whether it is seed ' \
            #                     'funding or for early traction etc'
            #                 ),
            #                 max_capacity=40,
            #                 max_length=100
            #             ),
            # FieldSchema(
            #                 name='eligible_sectors',
            #                 dtype=DataType.ARRAY,
            #                 element_type=DataType.VARCHAR,
            #                 description='Details of sectors for which funding is available',
            #                 max_capacity=40,
            #                 max_length=100
            #             ),
            # Language details
            FieldSchema(
                name="language",
                dtype=DataType.VARCHAR,
                description="Language of the document",
                max_length=15,
            ),
            # Content
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                description="Chunk original text",
                max_length=15000,
            ),
            FieldSchema(
                name="text_vector",
                dtype=DataType.FLOAT_VECTOR,
                description="Chunk embedding vector",
                dim=EMBEDDING_DIM,
            ),  # from EMBEDDING_MODEL (all-MiniLM-L6-v2 => 384)
            # FieldSchema(name="chunk_image_url", dtype=DataType.VARCHAR, description='Url of image where image is stored', max_length=512),
            # FieldSchema(name="chunk_image_vector", dtype=DataType.FLOAT_VECTOR, dim=512, description='Image embedded vector'), # CLIP embedding dim
            # Version control
            FieldSchema(
                name="doc_version",
                dtype=DataType.VARCHAR,
                max_length=5,
                description="Version of the document",
            ),
            FieldSchema(
                name="is_active",
                dtype=DataType.BOOL,
                description="Flag to mark if the document is active",
            ),
        ]
        return schema_fields

    @staticmethod
    def define_index(collection: CollectionSchema):
        """
        For Vector Fields:
        text_vector → EMBEDDING_DIM (384 for all-MiniLM-L6-v2)
        chunk_image_vector → 512 dim
        Recommended index type: HNSW or IVF_FLAT
        HNSW → great for high recall, stable for large collections
        IVF_FLAT → good for large-scale datasets with filtering

        Indexes for Categorical fields
        Inverted index is generally best for these fields in Milvus and similar vector databases, as it is optimized for fast filtering on categorical/metadata fields.
        Hash index can also work well for very low-cardinality fields (like is_active), but offers no major advantage over inverted for your use case.
        Binary tree is rarely used for categorical fields in modern vector databases;
        it’s more relevant for range/ordered queries on numeric data.
        """
        try:
            collection.create_index(
                field_name="text_vector",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
            """
            collection.create_index(
                field_name="chunk_image_vector",
                index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
            )
            """
            for date_field in ["crawl_date", "doc_last_modified"]:
                collection.create_index(
                    field_name=date_field, index_params={"index_type": "INVERTED"}
                )

            for schema_field in [
                "scheme_type",
                "language",
                "is_active",
            ]:  ##, 'eligible_sectors']: Removing to create index as its type is ARRAY.
                collection.create_index(field_name=schema_field, index_type="INVERTED")
        except Exception as e:
            raise MilvusException(message=f"Failed to create index: {e}", code=1)


class MilvusVectorClient:
    """Milvus implementation of BaseVectorClient."""

    def __init__(
        self,
        logger: Logger,
        collection_name: str,
        dim: int,
        host: str = "localhost",
        port: str = "19530",
    ):
        self.logger = logger
        self.collection_name = collection_name
        self.dim = dim
        self.host = host
        self.port = port

        try:
            self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")
            connections.connect(alias="default", host=self.host, port=self.port)
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def create_collection(self, overwrite: bool = False):
        """Create collection schema for Milvus."""
        try:
            if utility.has_collection(self.collection_name):
                if overwrite:
                    utility.drop_collection(self.collection_name)
                    self.logger.info(
                        f"Dropped existing collection: {self.collection_name}"
                    )
                else:
                    self.logger.info(f"Collection already exists: {self.collection_name}")
                    return
            schema = MilvusStorage.create_schema()
            funds_schema = CollectionSchema(
                fields=schema, description="Generic document embeddings"
            )
            collection = Collection(name=self.collection_name, schema=funds_schema)

            MilvusStorage.define_index(collection)
            self.logger.info(f"Created collection {self.collection_name} with indexes")
        # except MilvusException as e:  # Catch specific exceptions if possible
        # handle_milvus_exception(e=e, logger=self.logger)
        except Exception as e:  # Fallback for generic errors
            # handle_milvus_exception(e=e, logger=self.logger)
            print(e.with_traceback)

    def insert_nodes(self, nodes: List[Dict[str, Any]]) -> List[str]:
        """Insert records into Milvus."""
        try:
            if not nodes:
                self.logger.warning("insert_nodes: empty input list")
                return []

            # Validate required fields
            for rec in nodes:
                if "chunk_id" not in rec or "text_vector" not in rec:
                    raise ValueError(
                        f"Invalid record missing chunk_id or embedding: {rec}"
                    )

            res = self.client.insert(collection_name=self.collection_name, data=nodes)
            ids = [rec["chunk_id"] for rec in nodes]
            self.logger.info(f"Inserted {len(ids)} records into {self.collection_name}")
            return ids

        except Exception as e:
            self.logger.error(f"Insert failed for {len(nodes)} records: {e}")
            return []

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """Search Milvus with query embedding and return hits."""
        try:
            if not query_embedding or not isinstance(query_embedding, list):
                raise ValueError("Invalid query embedding")

            res = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                anns_field="text_vector",
                search_params={"metric_type": "COSINE", "params": {"ef": 128}},
                limit=top_k,
                output_fields=["chunk_id", "text", "funding_stage", "scheme_type"],
            )
            hits = []
            for hit in res[0]:
                entity = hit.get("entity", hit)
                hits.append(
                    {
                        "id": hit.get("id", entity.get("chunk_id")),
                        "score": 1 - hit.get("distance", 0.0),
                        "text": entity.get("text", ""),
                        **{
                            k: v
                            for k, v in entity.items()
                            if k not in ("text", "text_vector")
                        },
                    }
                )
            return hits

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics (total rows, index status, etc.)."""
        try:
            if not utility.has_collection(self.collection_name):
                raise ValueError(f"Collection {self.collection_name} does not exist")

            stats = self.client.get_collection_stats(self.collection_name)
            self.logger.info(f"Stats for {self.collection_name}: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get stats for {self.collection_name}: {e}")
            return {}

    def peek_chunks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the first `limit` chunks from the collection.
        This is useful for debugging to confirm data is stored correctly.
        """
        try:
            if not utility.has_collection(self.collection_name):
                raise ValueError(f"Collection {self.collection_name} does not exist")

            res = self.client.query(
                collection_name=self.collection_name,
                # filter=" ",  # no filter, return all
                output_fields=[
                    "chunk_id",
                    "document_id",
                    "text",
                    "funding_stage",
                    "scheme_type",
                ],
                limit=limit,
            )
            self.logger.info(f"Peeked {len(res)} chunks from {self.collection_name}")
            return res
        except Exception as e:
            self.logger.error(f"Failed to peek chunks from {self.collection_name}: {e}")
            return []

    def drop_collection(self, name: str):
        try:
            if utility.has_collection(name):
                utility.drop_collection(name)
                self.logger.info(f"Dropped collection: {name}")
        except Exception as e:
            self.logger.error(f"Failed to drop collection {name}: {e}")
