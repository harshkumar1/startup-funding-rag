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

This is the REST API backing the RAG pipeline. MCP tools wrapping /search and
/index-all are mounted at /mcp (see mcp_server.py).

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
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from mcp_server import mcp_http_app
from rag_api.routers import index_all, index_doc, search

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="startup-funding-rag",
    description=(
        "RAG API. Use /search for retrieval + Groq-generated answers, "
        "/index-all to rebuild the whole index from the source repo, and "
        "/index-doc to add a single new document. MCP streamable HTTP is at /mcp."
    ),
    lifespan=mcp_http_app.lifespan,
    # Behind Hugging Face, a slash redirect uses Location: http://... which
    # Claude cannot follow. Keep /mcp and /mcp/ on the same HTTPS URL.
    redirect_slashes=False,
)


@app.middleware("http")
async def mcp_path_slash(request: Request, call_next):
    if request.scope.get("path") == "/mcp":
        request.scope["path"] = "/mcp/"
    return await call_next(request)

app.include_router(search.router)
app.include_router(index_all.router)
app.include_router(index_doc.router)
app.mount("/mcp", mcp_http_app)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


def main() -> None:
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("PORT") or os.getenv("API_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
