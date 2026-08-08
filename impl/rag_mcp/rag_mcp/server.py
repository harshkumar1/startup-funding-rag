"""
RAG MCP server (FastMCP over HTTP) — skeleton.

Tools:
  1. query             — retrieve relevant chunks for a question
  2. ingest_new_docs    — incrementally ingest new documents
  3. reingest_all       — wipe and rebuild the entire index

This is intentionally a skeleton: each tool returns a stub response so the
server can be deployed to Cloud Run first, then filled in with real
chunking/embedding/vector-store logic afterwards.

Run locally (from the repo root, i.e. this rag_mcp/ directory):
  python -m rag_mcp.server

Endpoint: http://127.0.0.1:8080/mcp  (Cloud Run: $URL/mcp)
"""

from __future__ import annotations

import os

from fastmcp import FastMCP

mcp = FastMCP(
    name="rag-mcp",
    instructions=(
        "RAG MCP skeleton. Use query for retrieval, ingest_new_docs to add "
        "documents, and reingest_all to rebuild the whole index from scratch."
    ),
)


@mcp.tool
def query(query: str, top_k: int = 5) -> str:
    """
    Retrieve the most relevant chunks for a natural-language query.

    Args:
        query: User question / search text.
        top_k: Number of chunks to return (default 5).
    """
    return f"TODO: implement query(). Received query={query!r}, top_k={top_k}."


@mcp.tool
def ingest_new_docs(documents_json: str) -> str:
    """
    Incrementally ingest new documents (chunk -> embed -> insert) without
    touching existing data.

    Args:
        documents_json: JSON array of documents to ingest.
    """
    return f"TODO: implement ingest_new_docs(). Received {len(documents_json)} chars of documents_json."


@mcp.tool
def reingest_all() -> str:
    """
    Full rebuild: wipe the existing index and ingest everything from scratch.

    WARNING: destructive — will delete existing indexed data once implemented.
    """
    return "TODO: implement reingest_all()."


def main() -> None:
    host = os.getenv("MCP_HOST", "0.0.0.0")
    # Cloud Run injects PORT; fall back to MCP_PORT, then 8080 for local dev.
    port = int(os.getenv("PORT") or os.getenv("MCP_PORT", "8080"))
    mcp.run(transport="http", host=host, port=port)


if __name__ == "__main__":
    main()
