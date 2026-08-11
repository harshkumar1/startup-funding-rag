"""
Settings for retrieval / ingestion business logic (Zilliz + embedding model).

Populated once from environment variables via `load_settings()`. Deliberately
separate from server bootstrap config (API_HOST/PORT in server.py) so the
server can still start and deploy before ZILLIZ_* secrets are configured;
`load_settings()` is only called lazily, when an endpoint actually needs the
vector store or embedding model.

Local-dev convenience: if a `creds.config` file is found (see
`_find_creds_file`), its values are loaded into `os.environ` *before* the
env vars below are read — but only to fill in gaps, never overriding an
already-set env var. This keeps deployment (Cloud Run/GKE env vars / Secret
Manager) as the source of truth and requires zero code change once
`impl/` is split into its own repo (the file simply won't be found).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_CREDS_FILENAME = "creds.config"
_CREDS_SEARCH_MAX_LEVELS = 6

# creds.config key -> env var it fills in (only if that env var isn't already set).
_CREDS_FILE_KEY_MAP = {
    "ZILLIZ_URI": "ZILLIZ_URI",
    "User": "ZILLIZ_USER",
    "Password": "ZILLIZ_PASSWORD",
    "GROQ_API_KEY": "GROQ_API_KEY",
    "HF_TOKEN": "HF_TOKEN",
}

_CREDS_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.*)$")


def _find_creds_file() -> Path | None:
    """Return the creds file to use, or None if not configured/found.

    `CREDS_CONFIG_PATH` wins if set (must point to an existing file).
    Otherwise, search upward from the current working directory for a file
    named `creds.config` (handles running from `impl/` or the repo
    root), giving up after a few levels.
    """
    explicit = os.getenv("CREDS_CONFIG_PATH", "")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise RuntimeError(f"CREDS_CONFIG_PATH set but file not found: {path}")
        return path

    directory = Path.cwd()
    for _ in range(_CREDS_SEARCH_MAX_LEVELS):
        candidate = directory / _CREDS_FILENAME
        if candidate.is_file():
            return candidate
        if directory.parent == directory:
            break
        directory = directory.parent
    return None


def _strip_quotes(value: str) -> str:
    """Strip one layer of matching surrounding quotes, e.g. `"foo"` -> `foo`."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _read_creds_file(path: Path) -> dict[str, str]:
    """Parse a simple `KEY: value` / `KEY = value` per-line credentials file."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _CREDS_LINE_RE.match(line)
        if match:
            values[match.group(1)] = _strip_quotes(match.group(2).strip())
    return values


def _apply_creds_file() -> None:
    """Fill in any unset mapped env vars from the creds file, if one is found."""
    path = _find_creds_file()
    if path is None:
        return
    values = _read_creds_file(path)
    for file_key, env_var in _CREDS_FILE_KEY_MAP.items():
        if file_key in values and not os.getenv(env_var):
            os.environ[env_var] = values[file_key]


@dataclass(frozen=True)
class Settings:
    zilliz_uri: str
    zilliz_user: str
    zilliz_password: str
    collection_name: str
    embedding_model: str
    hf_token: str
    source_repo: str
    source_branch: str
    source_docs_subdir: str
    source_csv_path: str
    source_local_dir: str
    groq_api_key: str
    generation_timeout_seconds: int
    max_concurrent_generations: int


def load_settings() -> Settings:
    _apply_creds_file()

    zilliz_uri = os.getenv("ZILLIZ_URI", "")
    zilliz_user = os.getenv("ZILLIZ_USER", "")
    zilliz_password = os.getenv("ZILLIZ_PASSWORD", "")
    hf_token = os.getenv("HF_TOKEN", "")

    if not zilliz_uri:
        raise RuntimeError("ZILLIZ_URI environment variable is not set.")
    if not zilliz_user:
        raise RuntimeError("ZILLIZ_USER environment variable is not set.")
    if not zilliz_password:
        raise RuntimeError("ZILLIZ_PASSWORD environment variable is not set.")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set (required to call the "
            "Hugging Face Inference API for embeddings)."
        )

    return Settings(
        zilliz_uri=zilliz_uri,
        zilliz_user=zilliz_user,
        zilliz_password=zilliz_password,
        collection_name=os.getenv("COLLECTION_NAME", "rag_chunks"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        hf_token=hf_token,
        # Public source-of-truth repo for scraped docs (see AGENTS.md). Not a
        # secret — overridable in case the source repo ever moves.
        source_repo=os.getenv("GITHUB_SOURCE_REPO", "harshkumar1/website-scrapper"),
        source_branch=os.getenv("GITHUB_SOURCE_BRANCH", "main"),
        source_docs_subdir=os.getenv("GITHUB_SOURCE_DOCS_SUBDIR", "data/markdown"),
        source_csv_path=os.getenv("GITHUB_SOURCE_CSV_PATH", "data/raw_data.csv"),
        # If set, /index-all reads from this local checkout of the source
        # repo (source_docs_subdir / source_csv_path resolved under it)
        # instead of fetching from GitHub. Local dev only.
        source_local_dir=os.getenv("SOURCE_LOCAL_DIR", ""),
        # Groq (RAG "generate" stage — rag_core/retrieval/generation.py). Not
        # validated here (unlike zilliz_uri/hf_token above) since only /search
        # with generate_answer=true needs it — /index-all and a
        # retrieval-only /search must still work without it configured;
        # generation.py raises its own clear error if used while unset.
        #
        # Answer-quality knobs (model, temperature, max_tokens,
        # reasoning_format, system/user prompts) deliberately are NOT here —
        # env vars require a container restart to change. They live in
        # generation_config.json instead, re-read from disk on every
        # generate_answer() call, so tuning them doesn't need a restart or
        # rebuild. Only true ops/infra knobs stay as env vars below.
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        generation_timeout_seconds=int(os.getenv("GENERATION_TIMEOUT_SECONDS", "30")),
        max_concurrent_generations=int(os.getenv("MAX_CONCURRENT_GENERATIONS", "4")),
    )
