"""MCP tools wrapping the RAG REST APIs (`POST /search`, `POST /index-all`).

Used two ways:
  - Mounted on the FastAPI app at `/mcp` (streamable HTTP, Cloud Run / Docker).
  - Stdio: `python mcp_server.py` for a local Claude Desktop command server.
"""

from __future__ import annotations

from fastapi import HTTPException
from fastmcp import FastMCP

from rag_api.routers.index_all import index_all as index_all_http
from rag_api.routers.search import SearchRequest, search as search_http

mcp = FastMCP("startup-funding-rag")


@mcp.tool
def search(query: str, top_k: int = 5, generate_answer: bool = True) -> dict:
    """Search the startup-funding knowledge base and optionally generate a grounded answer.

    Same behavior as POST /search.
    """
    response = search_http(
        SearchRequest(query=query, top_k=top_k, generate_answer=generate_answer)
    )
    return response.model_dump()


@mcp.tool
def index_all() -> dict:
    """Drop and rebuild the entire vector index from the source GitHub repo.

    Same behavior as POST /index-all. Destructive.
    """
    try:
        return index_all_http().model_dump()
    except HTTPException as exc:
        return {"status": "error", "detail": exc.detail}


# path="/" because server.py mounts this app at /mcp (avoids /mcp/mcp).
# Host guard off so Claude Desktop can hit http://127.0.0.1:8080/mcp while
# uvicorn listens on 0.0.0.0 (Docker / Cloud Run).
mcp_http_app = mcp.http_app(
    path="/",
    stateless_http=True,
    host_origin_protection=False,
)


if __name__ == "__main__":
    mcp.run()
