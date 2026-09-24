from __future__ import annotations

import csv
import io
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ..common.config import Settings, load_settings

logger = logging.getLogger(__name__)

_USER_AGENT = "startup-funding-rag-ingest"
_MAX_WORKERS = 8
_REQUEST_TIMEOUT_S = 20
_FETCH_LOG_INTERVAL = 10


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    text: str
    meta: dict[str, str]


def _http_get_bytes(url: str, accept: str | None = None) -> bytes:
    headers = {"User-Agent": _USER_AGENT}
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_S) as response:
        return response.read()


def _fetch_raw_data_csv(settings: Settings) -> dict[str, dict[str, str]]:
    """Download the metadata CSV and index rows by `doc_id`."""
    url = (
        f"https://raw.githubusercontent.com/{settings.source_repo}/"
        f"{settings.source_branch}/{settings.source_csv_path}"
    )
    csv_text = _http_get_bytes(url).decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(csv_text))
    return {row["doc_id"]: row for row in reader}


def _list_markdown_files(settings: Settings) -> list[dict]:
    """List every `.md` file under the docs subdir via the GitHub contents API."""
    url = (
        f"https://api.github.com/repos/{settings.source_repo}/contents/"
        f"{settings.source_docs_subdir}?ref={settings.source_branch}"
    )
    entries = json.loads(_http_get_bytes(url, accept="application/vnd.github+json"))
    return [e for e in entries if e.get("type") == "file" and e["name"].endswith(".md")]


def _load_github_documents(settings: Settings) -> list[SourceDocument]:
    meta_by_doc_id = _fetch_raw_data_csv(settings)
    files = _list_markdown_files(settings)
    total = len(files)
    logger.info(
        "Fetching %d markdown files from %s (%d workers)...",
        total,
        settings.source_repo,
        _MAX_WORKERS,
    )

    def _fetch_one(entry: dict) -> SourceDocument | None:
        doc_id = entry["name"][: -len(".md")]
        meta = meta_by_doc_id.get(doc_id)
        if meta is None:
            return None
        text = _http_get_bytes(entry["download_url"]).decode("utf-8", errors="ignore")
        return SourceDocument(doc_id=doc_id, text=text, meta=meta)

    documents: list[SourceDocument] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        for i, result in enumerate(pool.map(_fetch_one, files), start=1):
            if result is not None:
                documents.append(result)
            if i % _FETCH_LOG_INTERVAL == 0 or i == total:
                logger.info("Fetched %d/%d source files.", i, total)

    if not documents:
        raise RuntimeError(
            f"No documents joined from {settings.source_repo}/{settings.source_docs_subdir} "
            f"+ {settings.source_csv_path} (check repo/branch/paths)."
        )
    return documents


def _load_local_documents(settings: Settings) -> list[SourceDocument]:
    base = Path(settings.source_local_dir)
    csv_path = base / settings.source_csv_path
    docs_dir = base / settings.source_docs_subdir

    if not csv_path.is_file():
        raise RuntimeError(f"SOURCE_LOCAL_DIR set but CSV not found: {csv_path}")
    if not docs_dir.is_dir():
        raise RuntimeError(f"SOURCE_LOCAL_DIR set but markdown dir not found: {docs_dir}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        meta_by_doc_id = {row["doc_id"]: row for row in csv.DictReader(f)}

    documents: list[SourceDocument] = []
    for path in sorted(docs_dir.glob("*.md")):
        meta = meta_by_doc_id.get(path.stem)
        if meta is None:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        documents.append(SourceDocument(doc_id=path.stem, text=text, meta=meta))

    if not documents:
        raise RuntimeError(
            f"No documents joined from local dir {base} "
            f"({settings.source_docs_subdir} + {settings.source_csv_path})."
        )
    return documents


def load_all_documents() -> list[SourceDocument]:
    settings = load_settings()
    if settings.source_local_dir:
        return _load_local_documents(settings)
    return _load_github_documents(settings)
