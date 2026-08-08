# AGENTS.md

Context for any human or AI agent working in this repo. Read this before making
structural changes, adding dependencies, or touching anything under `impl/`.

## What this repo is

A RAG (retrieval-augmented generation) system over scraped website content
(government/startup-scheme pages). Data flows: **scrape → chunk → embed →
store in Zilliz → retrieve → answer with an LLM**. The repo currently contains
experimentation notebooks that settled on a concrete architecture, plus a
skeleton MCP server (`impl/rag_mcp`) that is meant to become the productionized
version of that pipeline.

## Repo layout

```
rag/
├── AGENTS.md                  # this file
├── impl/rag_mcp/               # the actual project — MCP server (see below)
├── playground/                 # exploratory notebooks; source of truth for current architecture decisions
│   ├── chunking/                # chunking strategy experiments
│   ├── schema_design/           # Zilliz schema, embedding, indexing, insert
│   ├── retrieval/                # retrieval/query experiments
│   ├── transformers/             # embedding-model experiments
│   └── Integration.md            # how scraper → chunking → schema notebooks fit together
├── sample/                      # OLDER reference prototype — different stack, see caveat below
├── docs/                        # background reading (Milvus indexing guide, index-type demos)
└── output/                      # local scratch artifacts (e.g. selected_chunks.json), gitignored content
```

## Architecture decisions made so far (from `playground/`)

These are the decisions the current/next implementation should follow unless
explicitly changed:

- **Source data**: scraped pages live in a separate GitHub repo,
  `harshkumar1/website-scrapper` — markdown per page (`data/markdown/{doc_id}.md`)
  plus metadata (`data/raw_data.csv`). `doc_id` (the markdown filename) is the
  join key between the two.
- **Vector store**: **Zilliz Cloud** (managed Milvus), not self-hosted Milvus.
  Connect via `pymilvus.MilvusClient(uri=ZILLIZ_URI, token=f"{user}:{password}")`.
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` → 384-dim
  vectors. `VECTOR_DIM` and `MAX_SEQ_LENGTH` (256) should always be read off the
  model at runtime, never hardcoded — chunk size is deliberately aligned to
  `MAX_SEQ_LENGTH`.
- **Vector index**: `AUTOINDEX` + `COSINE` metric (Zilliz Cloud recommendation).
- **Collection**: `rag_chunks`, 16 fields — `chunk_id`, `document_id`,
  `chunk_order`, `base_url`, `canonical_url`, `crawl_date`, `doc_last_modified`,
  `content_type`, `content_source_type`, `scheme_type`, `scheme_name`,
  `language`, `text`, `text_vector`, `doc_version`, `is_active`. See
  `playground/Integration.md` for the full field-to-source mapping.
- **Chunking strategies evaluated**: token-based with overlap, token-based
  without overlap, sentence-based, semantic. Each strategy cell in
  `chunking_experiments_simple.ipynb` fills the same `chunks` shape so
  downstream code doesn't care which one ran.
- **LLM for generation**: **Groq** (`llama-3.x` family models), via the `groq`
  Python SDK — not OpenAI, not a local model, for the current direction.

## `sample/` is a legacy reference, not the current direction — don't copy it blindly

`sample/full_impl/` is an **earlier prototype** built on a different stack:
LlamaIndex + **Ollama** (local LLM) + **self-hosted Milvus** (host/port, not
Zilliz Cloud) + a 768-dim embedding model. That stack has since been replaced
by the decisions above (Zilliz Cloud, MiniLM/384-dim, Groq). Useful for
patterns (see below), but don't reuse its config values or assume it reflects
where this project is headed.

Patterns worth reusing from `sample/full_impl/` when building out real logic:
- Abstract base class for vector store clients (`base_client.py`,
  `BaseVectorClient`) so the store implementation can be swapped.
- Batch insert with per-record embedding-dimension validation and graceful
  skip-on-failure rather than hard-failing a whole batch
  (`custom_vector_store.py`).
- Bounding LLM concurrency with a semaphore + wall-clock timeout via a
  short-lived `ThreadPoolExecutor`, rather than letting generation calls hang
  (`retrieval_service.py`).
- `@dataclass` config objects instead of raw dicts (`datastore_config.py`).

## `impl/rag_mcp/` — the actual project

This is structured as a **self-contained mini-repo** nested inside the
monorepo, so it can be split out into its own GitHub repo later with zero path
changes:

```
impl/rag_mcp/                # treat this as a future standalone repo root
├── Dockerfile                 # build context = this dir, no path prefixes
├── cloudbuild.yaml
├── deploy_cloud_run.sh        # one-shot Cloud Run deploy script
├── requirements.txt
├── .env.example
├── .dockerignore
└── rag_mcp/                   # the importable Python package
    ├── __init__.py
    └── server.py               # FastMCP server, run via `python -m rag_mcp.server`
```

**Rule**: everything inside `impl/rag_mcp/` must stay self-relative (no
`impl/rag_mcp/...` path prefixes in the Dockerfile/cloudbuild/scripts) so the
directory works unmodified as its own git repo root.

### MCP tools (currently skeleton stubs — see `rag_mcp/server.py`)

1. `query(query: str, top_k: int = 5)` — retrieve top-k relevant chunks.
2. `ingest_new_docs(documents_json: str)` — incremental chunk → embed → insert,
   without touching existing data.
3. `reingest_all()` — destructive full rebuild of the index.

Built with **FastMCP** over HTTP transport. Deployment target is **Google
Cloud Run** (`PORT` env var drives the listen port; Cloud Run injects it).

Current state: each tool is a stub that returns a `TODO:` string. The
deliberate order of operations for this project is **skeleton → deploy →
implement**, i.e. get the empty server deployed and reachable first, then fill
in real chunking/embedding/vector-store logic tool by tool, redeploying
incrementally. Don't add real business logic and infra changes in the same
step if it can be avoided — keep deploys small and verifiable.

## Coding standards / preferred patterns

- **Never hardcode secrets, ever** — not even as a `os.getenv("X", "") or
  "literal-fallback"` pattern. Load everything from env vars with no
  literal fallback; raise a clear error if unset. (We had to scrub a real
  Zilliz password and cluster URI that had been hardcoded as notebook
  fallback values — see git history / past incidents. Treat this as a hard
  rule, not a suggestion.)
- Config via `@dataclass(frozen=True)` settings objects populated from
  `os.getenv(...)` in a single `load_settings()`-style function, not scattered
  `os.getenv` calls through business logic.
- Type hints on all function signatures; `from __future__ import annotations`
  at the top of new modules.
- FastMCP tool functions: always document `Args:` in the docstring — FastMCP
  surfaces these to MCP clients as the tool's parameter descriptions.
- Prefer plain `pymilvus.MilvusClient` over LlamaIndex abstractions for new
  code (matches the `playground/schema_design` direction, not `sample/`).
- Minimal dependencies — `impl/rag_mcp/requirements.txt` should only contain
  what's actually imported; don't add a framework "just in case."

## CI/CD

- `.github/workflows/security-scan.yml` — Trivy scan, see Security section below.
- `.github/workflows/docker-publish.yml` — builds `impl/rag_mcp/Dockerfile`
  (context = `impl/rag_mcp/`) and pushes to **GitHub Container Registry**
  (`ghcr.io/<owner>/rag-mcp`, tags: short commit SHA + `latest`). Uses the
  built-in `GITHUB_TOKEN` — no registry secrets to configure. This is
  separate from the Cloud Run deploy path (`deploy_cloud_run.sh`, which builds
  via Google Cloud Build instead) — GHCR publishing is for having a
  versioned, pullable image on GitHub itself; Cloud Run deploys still go
  through `cloudbuild.yaml`/`deploy_cloud_run.sh` today rather than pulling
  from GHCR. If these two build paths diverge, that's worth reconciling later
  (e.g. point Cloud Run at the GHCR image instead of rebuilding via Cloud
  Build).
  - **Gated on Trivy**: triggered via `workflow_run` off the `Security Scan`
    workflow's `completed` event, and only actually builds/pushes when
    `conclusion == 'success'` and `head_branch == 'main'` (or on manual
    `workflow_dispatch`, which bypasses the gate deliberately). This means it
    fires on *every* successful Trivy run on `main`, not just ones that
    touched `impl/rag_mcp/**` — `workflow_run` doesn't support path filters
    the way `push` does. Acceptable for now since `cache-from`/`cache-to:
    type=gha` makes a no-op rebuild cheap; revisit with a changed-files check
    inside the job if that stops being true.

## Security practices in this repo

- `.gitignore` blocks `*password*`, `.env`/`**/.env`, `*.pem`,
  `__pycache__/`, `.venv/`/`venv/`, `.DS_Store`, `.ipynb_checkpoints/`.
  `username-password.txt` at the repo root deliberately matches the
  `*password*` glob — it holds real local dev credentials and must never be
  un-ignored.
- `.github/workflows/security-scan.yml` runs **Trivy** (`fs` scan —
  vulnerabilities + secrets + misconfig) on every push/PR to `main` and on
  manual dispatch. This is the primary automated safety net and works
  regardless of repo visibility (doesn't depend on GitHub Advanced Security).
- `.github/dependabot.yml` covers `pip` deps in `impl/rag_mcp` and
  `github-actions` deps, weekly.
- This repo is **private**, on a **personal** GitHub account
  (`harshkumar1/rag`) — repo-level secret scanning, push protection, and
  CodeQL default setup are **not available** (those require a paid GitHub
  Advanced Security license, only purchasable on Team/Enterprise org plans).
  Don't assume those protections exist; the Trivy workflow is what's actually
  covering this repo. If it ever moves to a public repo or an org account,
  revisit enabling those natively.
- Before committing notebooks: check both the **source cells** and the
  **stored outputs** for secrets — we found real credentials leaked into
  printed cell outputs from a previous run, not just source code.

## Git / repo setup notes

- This folder (`rag/`) is its own git repo, separate from the
  home-directory-rooted git repo it happens to live inside on disk. Don't
  assume `git` commands here interact with any outer repo.
- Pushes to `github.com` from inside Cursor on this machine are blocked by an
  org-managed guardrail hook (`intuit-git-push-guard`). GitHub-side operations
  (repo creation, pushes) are done manually by the human in a plain terminal,
  not by an agent.
