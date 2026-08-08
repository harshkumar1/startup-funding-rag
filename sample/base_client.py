from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseVectorClient(ABC):
    """Abstract base class for vector DB clients."""

    @abstractmethod
    def insert_nodes(self, nodes: List[Dict[str, Any]]) -> List[str]:
        """Insert documents into the vector DB and return their IDs."""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict]:
        """Search the vector DB and return a list of hits with text + metadata."""
        pass

    @abstractmethod
    def create_collection(self, overwrite: bool = False):
        """Create collection with schema (optionally overwrite)."""
        pass

    @abstractmethod
    def drop_collection(self, name: str):
        """Drop a collection if it exists."""
        pass
