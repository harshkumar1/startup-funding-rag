import logging
from dataclasses import dataclass

# ---------------------------
# DataStore Config
# ---------------------------
@dataclass
class DataStoreConfig:
    collection: str = "funds_collection"
    dim: int = 768
    host: str = "localhost"
    port: str = "19530"