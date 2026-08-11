"""POST /index-doc — commit one new doc to the source repo, then index it (stub)."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class IndexDocRequest(BaseModel):
    document_id: str = Field(
        ..., description="Markdown filename / doc_id, e.g. 'scheme-123'."
    )
    markdown_content: str = Field(..., description="Raw markdown content of the document.")
    base_url: str = Field(..., description="Base URL the doc was/would be scraped from.")
    canonical_url: str = Field(..., description="Canonical URL of the doc.")
    crawl_date: str = Field(..., description="ISO date the doc was crawled.")
    doc_last_modified: str | None = Field(
        None, description="Last-modified date of the source doc, if known."
    )
    content_type: str = Field(..., description="e.g. 'scheme_page', 'faq', ...")
    content_source_type: str = Field(
        ..., description="e.g. 'government', 'startup', ..."
    )
    scheme_type: str | None = Field(None, description="Scheme category, if applicable.")
    scheme_name: str | None = Field(None, description="Scheme name, if applicable.")
    language: str = Field("en", description="Document language code.")


class IndexDocResponse(BaseModel):
    status: str
    detail: str


@router.post("/index-doc", response_model=IndexDocResponse)
def index_doc(request: IndexDocRequest) -> IndexDocResponse:
    """Add a single new document: commit it to the source GitHub repo
    (harshkumar1/website-scrapper), then chunk -> embed -> insert it into
    Zilliz without touching existing data.

    Steps (TODO):
      1. Commit `markdown_content` to
         harshkumar1/website-scrapper:data/markdown/{document_id}.md via the
         GitHub API, and append a row to data/raw_data.csv with the metadata
         fields on this request.
      2. Chunk the markdown content.
      3. Embed each chunk and insert into the `rag_chunks` collection.
    """
    return IndexDocResponse(
        status="not_implemented",
        detail=f"TODO: implement index_doc() for document_id={request.document_id!r}.",
    )
