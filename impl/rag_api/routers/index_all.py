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
    """Full rebuild: drop the collection, recreate its schema/index, and
    index every document from the source repo (harshkumar1/website-scrapper).

    WARNING: destructive — deletes all existing indexed data.

    Steps: drop + recreate the `rag_chunks` schema (see
    playground/schema_design/schema_design_notebook.ipynb) -> fetch every doc
    from the source repo (data/markdown/*.md + data/raw_data.csv) -> chunk
    each with the token-based-with-overlap strategy (see
    playground/chunking/chunking_experiments_simple.ipynb) -> embed -> insert
    -> create the vector index.
    """
    try:
        result = run_index_all()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IndexAllResponse(status="ok", detail="Indexing complete.", **result)
