"""
Rerank vector hits with a cross-encoder.

Sample uses NodeReranker (src/retrieve/reranking.py, not in this repo) with
cross-encoder/ms-marco-MiniLM-L-6-v2 and keep top_n. This module does the same
step over plain chunk dicts, scoring via the Hugging Face Inference API (same
as embeddings — no local torch).
"""

from __future__ import annotations

import logging

import requests

from ..common.config import load_settings

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_S = 30
_INFERENCE_URL_TEMPLATE = "https://router.huggingface.co/hf-inference/models/{model}"


def _auth_headers() -> dict[str, str]:
    settings = load_settings()
    return {"Authorization": f"Bearer {settings.hf_token}"}


def _score_from_item(item: object) -> float:
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, dict):
        return float(item.get("score", 0.0))
    if isinstance(item, list) and item:
        return _score_from_item(item[0])
    raise RuntimeError(f"Unexpected cross-encoder score item: {item!r}")


def _parse_scores(payload: object, n: int) -> list[float]:
    if not isinstance(payload, list) or len(payload) != n:
        raise RuntimeError(f"Unexpected cross-encoder response: {payload!r}")
    return [_score_from_item(item) for item in payload]


def rerank_chunks(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    """Reorder chunks by cross-encoder relevance; keep the first top_n."""
    if not chunks or top_n <= 0:
        return []

    settings = load_settings()
    url = _INFERENCE_URL_TEMPLATE.format(model=settings.rerank_model)
    pairs = [[query, chunk.get("text") or ""] for chunk in chunks]
    response = requests.post(
        url,
        headers=_auth_headers(),
        json={"inputs": pairs},
        timeout=_REQUEST_TIMEOUT_S,
    )
    if not response.ok:
        raise RuntimeError(
            f"HF Inference API rerank request failed "
            f"({response.status_code}): {response.text}"
        )

    scores = _parse_scores(response.json(), len(chunks))
    ranked: list[dict] = []
    for chunk, rerank_score in zip(chunks, scores):
        updated = dict(chunk)
        updated["vector_score"] = chunk.get("score")
        updated["rerank_score"] = rerank_score
        updated["score"] = rerank_score
        ranked.append(updated)
    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return ranked[:top_n]
