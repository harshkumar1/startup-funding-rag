from __future__ import annotations

import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from string import Template

from groq import Groq

from ..common.config import load_settings

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "generation_config.json"
_CONFIG_SEARCH_MAX_LEVELS = 6

_DEFAULT_CONFIG = {
    "model": "llama-3.3-70b-versatile",
    "temperature": 0.2,
    "max_tokens": 1024,
    "reasoning_format": "",
    "system_prompt": (
        "You are a funding assistant that helps startups find government "
        "schemes, grants, and investor programs based on the provided context.\n"
        "- Answer only using the information in the provided context — never rely on outside knowledge.\n"
        "- If the context doesn't contain the answer, say so plainly instead of guessing.\n"
        "- Be concise and clear. Reference the relevant source URL(s) from the context when useful.\n"
    ),
    "user_prompt_template": "Context:\n$context\n\nQuestion:\n$query",
}

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_client: Groq | None = None
_semaphore: threading.BoundedSemaphore | None = None


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    model: str
    temperature: float
    max_tokens: int
    reasoning_format: str
    system_prompt: str
    user_prompt: str


def _find_config_file() -> Path | None:
    explicit = os.getenv("GENERATION_CONFIG_PATH", "")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise RuntimeError(f"GENERATION_CONFIG_PATH set but file not found: {path}")
        return path

    directory = Path.cwd()
    for _ in range(_CONFIG_SEARCH_MAX_LEVELS):
        candidate = directory / _CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        if directory.parent == directory:
            break
        directory = directory.parent
    return None


def _load_generation_config() -> dict:
    config = dict(_DEFAULT_CONFIG)
    path = _find_config_file()
    if path is None:
        logger.warning("No %s found; using built-in generation defaults.", _CONFIG_FILENAME)
        return config

    try:
        overrides = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error(
            "Failed to read/parse %s (%s); using built-in generation defaults.", path, exc
        )
        return config

    unknown_keys = set(overrides) - set(_DEFAULT_CONFIG)
    if unknown_keys:
        logger.warning("Ignoring unknown key(s) in %s: %s", path, sorted(unknown_keys))
    config.update({k: v for k, v in overrides.items() if k in _DEFAULT_CONFIG})
    return config


def _get_client() -> Groq:
    global _client
    if _client is None:
        settings = load_settings()
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set (required to "
                "generate an answer via Groq)."
            )
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def _get_semaphore() -> threading.BoundedSemaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = threading.BoundedSemaphore(load_settings().max_concurrent_generations)
    return _semaphore


def _build_user_prompt(template: str, query: str, chunks: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[{i + 1}] (source: {chunk.get('canonical_url') or 'unknown'})\n{chunk.get('text', '')}"
        for i, chunk in enumerate(chunks)
    )
    return Template(template).safe_substitute(context=context_block, query=query)


def _strip_think_block(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    if not cleaned and "<think>" in text:
        logger.warning(
            "Generation response was truncated inside a <think> block with "
            "no answer text after it — likely max_tokens is too low."
        )
        return ""
    return cleaned


def _fallback_answer(chunks: list[dict], reason: str) -> str:
    logger.warning("Falling back to raw chunks for generation (%s).", reason)
    snippets = "\n\n".join(chunk.get("text", "")[:300] for chunk in chunks[:3])
    return (
        "I couldn't generate a complete answer right now. Here are the most "
        f"relevant excerpts I found:\n\n{snippets}"
    )


def generate_answer(query: str, chunks: list[dict]) -> GenerationResult:
    gen_config = _load_generation_config()

    def _result(answer: str, user_prompt: str = "") -> GenerationResult:
        return GenerationResult(
            answer=answer,
            model=gen_config["model"],
            temperature=gen_config["temperature"],
            max_tokens=gen_config["max_tokens"],
            reasoning_format=gen_config["reasoning_format"],
            system_prompt=gen_config["system_prompt"],
            user_prompt=user_prompt,
        )

    if not chunks:
        return _result("No relevant context found to answer this question.")

    user_prompt = _build_user_prompt(gen_config["user_prompt_template"], query, chunks)
    settings = load_settings()

    def _call_groq() -> str:
        client = _get_client()
        kwargs = {}
        if gen_config["reasoning_format"]:
            kwargs["reasoning_format"] = gen_config["reasoning_format"]
        response = client.chat.completions.create(
            model=gen_config["model"],
            messages=[
                {"role": "system", "content": gen_config["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=gen_config["temperature"],
            max_tokens=gen_config["max_tokens"],
            **kwargs,
        )
        return _strip_think_block(response.choices[0].message.content or "")

    semaphore = _get_semaphore()
    if not semaphore.acquire(timeout=settings.generation_timeout_seconds):
        return _result(
            _fallback_answer(chunks, "timed out waiting for a generation concurrency slot"),
            user_prompt,
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_groq)
            answer = future.result(timeout=settings.generation_timeout_seconds)
    except FutureTimeoutError:
        return _result(
            _fallback_answer(
                chunks, f"Groq call exceeded {settings.generation_timeout_seconds}s timeout"
            ),
            user_prompt,
        )
    except RuntimeError as exc:
        return _result(_fallback_answer(chunks, str(exc)), user_prompt)
    except Exception as exc:
        logger.exception("Groq generation failed.")
        return _result(_fallback_answer(chunks, f"Groq call raised {exc!r}"), user_prompt)
    finally:
        semaphore.release()

    return _result(answer or _fallback_answer(chunks, "empty completion from Groq"), user_prompt)
