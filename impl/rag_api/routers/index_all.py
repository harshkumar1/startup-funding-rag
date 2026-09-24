"""POST /index-all — drop + recreate schema, index every doc from scratch."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag_core.indexing.indexer import index_all as run_index_all

router = APIRouter()


class IndexAllResponse(BaseModel):
    status: str
    detail: str
    collection: str | None = None
    documents_ingested: int | None = None
    chunks_inserted: int | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None


@router.post("/index-all", response_model=IndexAllResponse)
def index_all() -> IndexAllResponse:
    try:
        result = run_index_all()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IndexAllResponse(status="ok", detail="Indexing complete.", **result)
