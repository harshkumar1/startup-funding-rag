from __future__ import annotations

from pymilvus import MilvusClient

from .config import load_settings

_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        settings = load_settings()
        token = f"{settings.zilliz_user}:{settings.zilliz_password}"
        _client = MilvusClient(uri=settings.zilliz_uri, token=token)
    return _client
