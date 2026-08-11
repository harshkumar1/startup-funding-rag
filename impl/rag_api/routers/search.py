"""POST /search — retrieve relevant chunks and (optionally) generate an
answer for a question (implemented)."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from rag_core.common.embeddings import embed_text
from rag_core.retrieval.generation import generate_answer
from rag_core.retrieval.search import search_chunks

logger = logging.getLogger(__name__)

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(..., description="User question / search text.")
    top_k: int = Field(5, description="Number of chunks to return.")
    generate_answer: bool = Field(
        True,
        description=(
            "If true, also generate a natural-language answer from the "
            "retrieved chunks via Groq. If false, only `results` is returned "
            "(no GROQ_API_KEY required)."
        ),
    )


class GenerationInfo(BaseModel):
    """The exact generation config + rendered prompt behind `answer` —
    included in the response itself (not just the logs) so the model,
    temperature, max_tokens, reasoning_format, system prompt, and rendered
    user prompt are all visible together in one place."""

    model: str
    temperature: float
    max_tokens: int
    reasoning_format: str
    system_prompt: str
    user_prompt: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[dict]
    answer: str | None = Field(
        None, description="Generated answer, or null if generate_answer was false."
    )
    generation: GenerationInfo | None = Field(
        None,
        description=(
            "Generation config + rendered prompt used to produce `answer`, "
            "or null if generate_answer was false."
        ),
    )


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Retrieve the most relevant chunks for a natural-language query, then
    (by default) generate an answer grounded in those chunks.

    Embeds the query with the same model used to build the collection, runs
    a vector search against Zilliz Cloud for the nearest chunks, and — unless
    `generate_answer` is false — passes them to Groq
    (`rag_core.retrieval.generation.generate_answer`) to produce a final
    natural-language answer.
    """
    vector = embed_text(request.query)
    chunks = search_chunks(vector, top_k=request.top_k)

    answer = None
    generation_info = None
    if request.generate_answer:
        result = generate_answer(request.query, chunks)
        answer = result.answer
        generation_info = GenerationInfo(
            model=result.model,
            temperature=result.temperature,
            max_tokens=result.max_tokens,
            reasoning_format=result.reasoning_format,
            system_prompt=result.system_prompt,
            user_prompt=result.user_prompt,
        )

    response = SearchResponse(
        query=request.query,
        top_k=request.top_k,
        results=chunks,
        answer=answer,
        generation=generation_info,
    )

    # Logged unconditionally (retrieval-only calls included) so the full
    # response — chunks, answer, and the exact generation config/prompt that
    # produced it — is always visible together in the logs, not just in the
    # HTTP response body.
    logger.info("Search response:\n%s", response.model_dump_json(indent=2))
    return response
