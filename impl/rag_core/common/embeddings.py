from __future__ import annotations

import logging

import requests

from .config import load_settings

logger = logging.getLogger(__name__)

_INFERENCE_URL_TEMPLATE = (
    "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
)
_CONFIG_URL_TEMPLATE = "https://huggingface.co/{model}/raw/main/sentence_bert_config.json"
_REQUEST_TIMEOUT_S = 30
_EMBED_BATCH_SIZE = 32

_vector_dim: int | None = None
_max_seq_length: int | None = None


def _auth_headers() -> dict[str, str]:
    settings = load_settings()
    return {"Authorization": f"Bearer {settings.hf_token}"}


def embed_texts(texts: list[str]) -> list[list[float]]:
    settings = load_settings()
    url = _INFERENCE_URL_TEMPLATE.format(model=settings.embedding_model)

    total_batches = (len(texts) + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE
    if total_batches > 1:
        logger.info("Embedding %d texts in %d batch(es)...", len(texts), total_batches)

    vectors: list[list[float]] = []
    for batch_num, start in enumerate(range(0, len(texts), _EMBED_BATCH_SIZE), start=1):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        response = requests.post(
            url,
            headers=_auth_headers(),
            json={"inputs": batch},
            timeout=_REQUEST_TIMEOUT_S,
        )
        if not response.ok:
            raise RuntimeError(
                f"HF Inference API embedding request failed "
                f"({response.status_code}): {response.text}"
            )
        vectors.extend(response.json())
        if total_batches > 1:
            logger.info("Embedded batch %d/%d (%d texts).", batch_num, total_batches, len(batch))
    return vectors


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def get_vector_dim() -> int:
    global _vector_dim
    if _vector_dim is None:
        _vector_dim = len(embed_text("dimension probe"))
    return _vector_dim


def get_max_seq_length() -> int:
    global _max_seq_length
    if _max_seq_length is None:
        settings = load_settings()
        url = _CONFIG_URL_TEMPLATE.format(model=settings.embedding_model)
        response = requests.get(url, timeout=_REQUEST_TIMEOUT_S)
        if not response.ok:
            raise RuntimeError(
                f"Failed to fetch sentence_bert_config.json for "
                f"{settings.embedding_model} ({response.status_code}): {response.text}"
            )
        _max_seq_length = int(response.json()["max_seq_length"])
    return _max_seq_length
