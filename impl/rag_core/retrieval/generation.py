"""
LLM-based answer generation — the RAG "generate" stage.

Takes the user's question plus chunks already retrieved by
`rag_core.retrieval.search.search_chunks` and produces a grounded
natural-language answer via Groq (see AGENTS.md: Groq, llama-3.x family,
not OpenAI/local).

Answer-quality tuning knobs (system prompt, user prompt template, model,
temperature, max_tokens, reasoning_format) intentionally do NOT live in this
code or in env vars — they're read fresh from `generation_config.json` (see
`_load_generation_config`) on every single call, with no in-process caching.
That means editing the file (typically a bind-mounted override — see
impl/README.MD) takes effect on the very next `/search` request, with no
server restart or image rebuild needed. Only infra-level settings that
aren't about answer *quality* (`GROQ_API_KEY`, concurrency/timeout bounds)
stay in env vars via `common/config.py`, since those are ops knobs, not
things you'd iterate on while tuning retrieval/generation quality.

`generate_answer()` returns a `GenerationResult` bundling the answer
together with the exact config + rendered prompt that produced it, rather
than logging them separately here — the `/search` router serializes the
whole thing into its response (and logs that single JSON blob), so
everything about a generation call is visible in one place.

Other patterns ported from sample/full_impl/{grok_retrieval_service.py,
system_prompt.py,user_prompt.py}:
- A context-only system prompt (answer strictly from the provided chunks).
- Bounded LLM concurrency (`threading.BoundedSemaphore`) plus a wall-clock
  generation timeout enforced via a short-lived `ThreadPoolExecutor`, rather
  than letting a slow/hung generation call block a request indefinitely.
- Never raises for LLM-side failures (missing config, timeout, API error,
  empty completion) — always falls back to a usable message built from the
  retrieved chunks so `/search` still returns something meaningful.
"""

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

# Built-in fallback, used only if generation_config.json can't be found or
# fails to parse — keeps /search answering (rather than hard-failing every
# request) even with a missing file or one that's briefly malformed mid-edit.
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
    """The answer plus the exact config + rendered prompt used to produce
    it — bundled together so a caller (the `/search` router) can serialize
    everything into one response/log entry instead of it being scattered."""

    answer: str
    model: str
    temperature: float
    max_tokens: int
    reasoning_format: str
    system_prompt: str
    user_prompt: str


def _find_config_file() -> Path | None:
    """Resolve `generation_config.json`, re-resolved on every call (not just
    the contents) in case an operator moves/replaces it at runtime.

    `GENERATION_CONFIG_PATH` wins if set (must point to an existing file).
    Otherwise search upward from cwd, same pattern as
    `common.config._find_creds_file` — this finds the default file baked
    into the Docker image at `/app/generation_config.json` (WORKDIR), or a
    bind-mounted override at that same path.
    """
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
    """Read the generation-tuning config fresh from disk (see module
    docstring — deliberately no caching) and merge it over `_DEFAULT_CONFIG`."""
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
    # `Template.safe_substitute` (not an f-string/`.format()`) so literal
    # `{`/`}` characters in scraped chunk text can never be misread as
    # format placeholders or raise a KeyError/IndexError.
    return Template(template).safe_substitute(context=context_block, query=query)


def _strip_think_block(text: str) -> str:
    """Strip a leaked `<think>...</think>` block (belt-and-suspenders in case
    a reasoning model is configured and `reasoning_format` doesn't hide it,
    e.g. an older API version)."""
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
    """Generate a natural-language answer to `query`, grounded in `chunks`.

    Args:
        query: The user's original question.
        chunks: Retrieved chunks (see `search_chunks`), each expected to
            have at least a `text` field.

    Returns:
        A `GenerationResult` with the answer plus the exact model/temperature/
        max_tokens/reasoning_format/system_prompt/user_prompt used to
        produce it (see `_load_generation_config` for where those come
        from) — never raises for LLM-side failures.
    """
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
    except Exception as exc:  # noqa: BLE001 — any Groq/API failure should degrade, not 500
        logger.exception("Groq generation failed.")
        return _result(_fallback_answer(chunks, f"Groq call raised {exc!r}"), user_prompt)
    finally:
        semaphore.release()

    return _result(answer or _fallback_answer(chunks, "empty completion from Groq"), user_prompt)
