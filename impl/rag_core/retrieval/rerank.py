"""
Rerank vector hits with a cross-encoder.

Sample uses NodeReranker (src/retrieve/reranking.py, not in this repo) with
cross-encoder/ms-marco-MiniLM-L-6-v2 and keep top_n. Hugging Face Inference
does not host that model (`Model not supported by provider hf-inference`),
so this runs the same CrossEncoder locally (CPU), like sample.
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from ..common.config import load_settings

logger = logging.getLogger(__name__)

# Hub id is L6 (no hyphen). Sample config uses L-6; try both.
_MODEL_ALIASES = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L6-v2",
}

_model: CrossEncoder | None = None
_model_id: str | None = None


def _load_cross_encoder(model_id: str) -> CrossEncoder:
    candidates = [model_id]
    alias = _MODEL_ALIASES.get(model_id)
    if alias:
        candidates.append(alias)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            logger.info("Loading cross-encoder %r for rerank.", candidate)
            return CrossEncoder(candidate)
        except Exception as exc:
            last_error = exc
            logger.warning("Failed to load cross-encoder %r: %s", candidate, exc)
    raise RuntimeError(
        f"Could not load rerank model {model_id!r}"
    ) from last_error


def _get_model() -> CrossEncoder:
    global _model, _model_id
    settings = load_settings()
    if _model is None or _model_id != settings.rerank_model:
        _model = _load_cross_encoder(settings.rerank_model)
        _model_id = settings.rerank_model
    return _model


def rerank_chunks(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    """Reorder chunks by cross-encoder relevance; keep the first top_n."""
    if not chunks or top_n <= 0:
        return []

    pairs = [(query, chunk.get("text") or "") for chunk in chunks]
    scores = _get_model().predict(pairs)

    ranked: list[dict] = []
    for chunk, rerank_score in zip(chunks, scores):
        updated = dict(chunk)
        updated["vector_score"] = chunk.get("score")
        updated["rerank_score"] = float(rerank_score)
        updated["score"] = float(rerank_score)
        ranked.append(updated)
    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return ranked[:top_n]
