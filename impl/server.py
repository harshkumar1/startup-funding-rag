"""
RAG API server (FastAPI) entrypoint.

Endpoints (see rag_api/routers/):
  1. POST /search       — retrieve relevant chunks for a question, then
                          generate a Groq-based answer grounded in them
                          (implemented)
  2. POST /index-all     — drop + recreate the collection schema, then index
                          every doc from the source repo from scratch (implemented)
  3. POST /index-doc     — commit one new doc to the source GitHub repo, then
                          chunk/embed/insert just that doc (stub)

This is the REST API backing the RAG pipeline. A separate `rag_mcp` project
(planned, not yet built — see AGENTS.md) will wrap these endpoints and expose
them as MCP tools. Built incrementally: skeleton -> deploy -> implement
endpoint by endpoint.

Composition root: lives at the top of impl/ (not inside a package) since it
just wires together rag_api/ (HTTP layer) and rag_core/ (business logic).

Run locally (from impl/):
  python server.py

Docs: http://127.0.0.1:8080/docs  (Cloud Run: $URL/docs)
"""

from __future__ import annotations

import logging
import os

import uvicorn
from fastapi import FastAPI

from rag_api.routers import index_all, index_doc, search

# Configured before anything else logs — rag_core's per-stage/per-batch
# progress logging (indexer.py, embeddings.py, source_repo.py, schema.py)
# relies on the root logger having a handler, otherwise those calls are
# silently dropped. uvicorn's own dictConfig (set up inside uvicorn.run(),
# below) only touches its own "uvicorn"/"uvicorn.access"/"uvicorn.error"
# loggers and leaves `disable_existing_loggers=False`, so this still applies.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="startup-funding-rag",
    description=(
        "RAG API. Use /search for retrieval + Groq-generated answers, "
        "/index-all to rebuild the whole index from the source repo, and "
        "/index-doc to add a single new document."
    ),
)

app.include_router(search.router)
app.include_router(index_all.router)
app.include_router(index_doc.router)


def main() -> None:
    host = os.getenv("API_HOST", "0.0.0.0")
    # Cloud Run injects PORT; fall back to API_PORT, then 8080 for local dev.
    port = int(os.getenv("PORT") or os.getenv("API_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
