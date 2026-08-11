# AGENTS.md

Context for any human or AI agent working in this repo. Read this before making
structural changes, adding dependencies, or touching anything under `impl/`.

## What this repo is

A RAG (retrieval-augmented generation) system over scraped website content
(government/startup-scheme pages). 

Data flows: 
**scrape → chunk → embed → store in Zilliz → retrieve → answer with an LLM**. 

`playground/` holds the experimentation notebooks; 
`impl/` has the REST API server.

`impl/rag_mcp/` **planned but not yet built**: a thin MCP-tool wrapper around this REST API to expose them as MCP tools

## Repo layout
```
rag/
├── AGENTS.md                  # this file
├── impl/                       # the actual project — REST API server (see below)
├── (impl/rag_mcp/)              # PLANNED, not yet built — future MCP-tool wrapper around impl/'s REST API
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

- **Source data**: scraped pages live in a separate GitHub repo, `harshkumar1/website-scrapper` — markdown per page (`data/markdown/{doc_id}.md`), plus metadata (`data/raw_data.csv`). `doc_id` (the markdown filename) is the join key between the two.

- **Vector store**: **Zilliz Cloud** (managed Milvus), not self-hosted Milvus. Connect via `pymilvus.MilvusClient(uri=ZILLIZ_URI, token=f"{user}:{password}")`.

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` → 384-dim vectors, called via the **Hugging Face Inference API** (`HF_TOKEN`), not downloaded/run locally — keeps the service free of `torch`/`sentence-transformers` as dependencies. `VECTOR_DIM` and `MAX_SEQ_LENGTH` (256) are always read off the model at runtime, never hardcoded — `VECTOR_DIM` via a probe embedding call, `MAX_SEQ_LENGTH` via the model's `sentence_bert_config.json` on the HFHub. Chunk size is deliberately aligned to `MAX_SEQ_LENGTH`.

- **Vector index**: `AUTOINDEX` + `COSINE` metric (Zilliz Cloud recommendation).

- **Collection**: `rag_chunks`, 16 fields — `chunk_id`, `document_id`,
  `chunk_order`, `base_url`, `canonical_url`, `crawl_date`, `doc_last_modified`,
  `content_type`, `content_source_type`, `scheme_type`, `scheme_name`,
  `language`, `text`, `text_vector`, `doc_version`, `is_active`. See
  `playground/Integration.md` for the full field-to-source mapping.

- **Chunking**: token-based with overlap (`chunk_size = embedder.max_seq_length`, `overlap = max(32, chunk_size // 8)`, ~12.5%). 
Other strategies (no-overlap, sentence-based, semantic) were evaluated in `chunking_experiments_simple.ipynb`.

- **LLM for generation**: **Groq** (`llama-3.x` family models), via the `groq` Python SDK — not OpenAI, not a local model.

## `sample/` is a reference implementation

## `impl/` the actual project

Structured as a **self-contained mini-repo**, so it can be split out into its own GitHub repo later with zero path changes — everything inside `impl/` must stay self-relative (no `impl/...` path prefixes in the
Dockerfile/cloudbuild/scripts).

```
impl/                        # treat this as a future standalone repo root
├── Dockerfile                 # build context = this dir, no path prefixes
├── server.py                  # composition root: creates the FastAPI app, run via `python server.py`
├── cloudbuild.yaml
├── deploy_cloud_run.sh        # one-shot Cloud Run deploy script
├── requirements.txt
├── .env.example
├── .dockerignore
├── generation_config.json      # answer-generation tuning knobs — hot-reloaded, see impl/README.MD
├── rag_api/                   # HTTP layer only — imports rag_core for business logic
│   └── routers/                 # one APIRouter module per endpoint (request/response models live here too)
│       ├── search.py               # POST /search
│       ├── index_all.py             # POST /index-all
│       └── index_doc.py              # POST /index-doc
└── rag_core/                  # framework-agnostic business logic (no FastAPI/HTTP imports)
    ├── common/                    # shared by both indexing/ and retrieval/
    │   ├── config.py                 # @dataclass(frozen=True) Settings, load_settings()
    │   ├── embeddings.py              # HF Inference API client, embed_text/embed_texts, dim/max_seq_length helpers
    │   └── vector_store.py            # lazy MilvusClient — get_client() only
    ├── indexing/                  # everything behind /index-all (and /index-doc, once built)
    │   ├── schema.py                  # rag_chunks schema (drop/create) + AUTOINDEX/COSINE index
    │   ├── source_repo.py              # fetch docs+metadata from harshkumar1/website-scrapper (GitHub API)
    │   ├── chunking.py                  # token-with-overlap chunking + schema-aligned record building
    │   └── indexer.py                    # index_all(): schema -> fetch -> chunk -> embed -> insert -> index
    └── retrieval/                 # everything behind /search
        ├── search.py                  # search_chunks() — vector search against Zilliz
        └── generation.py               # generate_answer() — Groq chat completion grounded in retrieved chunks
```

**Pattern**: each endpoint gets its own `APIRouter` module under `rag_api/routers/` (FastAPI's "Bigger Applications" pattern) — `server.py` stays a thin composition root wiring `rag_api`'s routers onto a `FastAPI()` instance. Request/response Pydantic models live next to their route handler rather than a separate `schemas.py`. Business logic lives in `rag_core/` — imported by `rag_api/routers/*`, with zero imports the other way — so it's independently testable and reusable from the future `rag_mcp` wrapper. Within `rag_core/`, only put a module in `common/` if both `indexing/` and `retrieval/` actually import it (today: settings, the embedding client, the Milvus client factory) — anything used by only one side belongs in that side's package.

### API endpoints (see `server.py`)

1. `POST /search` — retrieve top-k relevant chunks for a question, then generate a natural-language answer grounded in them. 
**Implemented**
   (embed query -> `rag_core.retrieval.search.search_chunks` against Zilliz
   -> `rag_core.retrieval.generation.generate_answer` via Groq). Generation
   can be skipped per-request (`generate_answer: false`) to get raw chunks
   only, e.g. for callers without `GROQ_API_KEY` configured.

2. `POST /index-all` — drop the `rag_chunks` collection, recreate its schema, fetch every doc from the source repo, chunk -> embed -> insert, then create the vector index. 
**Implemented**
   (`rag_core.indexing.indexer.index_all`). Destructive — wipes existing data every run.

3. `POST /index-doc` — commit one new doc's markdown + metadata to `harshkumar1/website-scrapper` via the GitHub API, then chunk/embed/insert just that doc without touching existing data. Stub.

Built with **FastAPI** + **uvicorn**, run via `python server.py`. Docker image / GHCR repo / Cloud Run service are all named `startup-funding-rag`. 

**Google Cloud Run** to be implemented

`/index-all` notes:
- `indexing/schema.create_vector_index()` calls `load_collection()` after
  creating/confirming the index, since Zilliz Cloud (serverless/free tier)
  doesn't load a collection into memory just because it has an index, and
  auto-releases idle collections later to save resources.
  `retrieval/search.search_chunks()` self-heals from that: on a "not
  loaded" `MilvusException` it loads the collection and retries once.
- `indexing/source_repo.py` fetches docs via the GitHub REST API + raw
  content CDN (no `git` binary needed in the container), with bounded
  concurrency. Inserts are batched (500 records/call) for Zilliz's payload
  limit. Per-stage progress is logged (visible in `docker logs -f`) since a
  multi-minute `/index-all` call otherwise gives no feedback until the end.
- `/index-doc` is still a stub returning a `TODO:` detail.

`/search` generation notes:
- `retrieval/generation.py`'s `generate_answer()` calls Groq with a
  context-only system prompt (answer only from retrieved chunks) plus a
  user prompt built from the chunk texts + question. Returns a
  `GenerationResult` dataclass bundling the answer with the exact
  model/temperature/max_tokens/reasoning_format/system_prompt/user_prompt
  used, which the `/search` router serializes into a `generation` field on
  the response (and logs as one JSON blob).
- **Answer-quality knobs (`model`, `temperature`, `max_tokens`,
  `reasoning_format`, `system_prompt`, `user_prompt_template`) live in
  `generation_config.json`, not code or env vars**, re-read from disk on
  *every* call (no caching) — see impl/README.MD for the Docker bind-mount
  workflow this enables. `user_prompt_template` uses `string.Template`
  (`$context`/`$query`), not f-string/`.format()`, so literal `{`/`}` in
  scraped chunk text is never misread. Falls back to a built-in default
  config (logged as a warning) if the file is missing/invalid.
- Only ops/infra knobs stay as env vars: `GROQ_API_KEY`,
  `GENERATION_TIMEOUT_SECONDS` (default 30s), `MAX_CONCURRENT_GENERATIONS`
  (default 4, via `threading.BoundedSemaphore`).
- Never raises for LLM-side failures (missing key, timeout, API error,
  empty completion) — falls back to a short message plus top raw chunk
  excerpts, so `/search` still returns something usable.

The deliberate order of operations for this project is **skeleton → deploy →
implement**: get the server deployed and reachable first, then fill in real
logic endpoint by endpoint, redeploying incrementally.

## Coding standards / preferred patterns

- **Python 3.10+ required** (hard floor) — `X | None` PEP 604 union syntax
  is used throughout, including inside Pydantic models that evaluate
  annotations at runtime; `from __future__ import annotations` defers
  *parsing* but not eventual *evaluation*, so this won't import on 3.9 or
  older. The Docker image pins Python 3.12 regardless.
- **Never hardcode secrets, ever** — not even as an
  `os.getenv("X", "") or "literal-fallback"` pattern. Load everything from
  env vars with no literal fallback; raise a clear error if unset. (We had
  to scrub a real Zilliz password/URI hardcoded as a notebook fallback —
  treat this as a hard rule.)
- Config via `@dataclass(frozen=True)` settings objects populated from
  `os.getenv(...)` in a single `load_settings()`-style function, not
  scattered `os.getenv` calls through business logic.
- Type hints on all function signatures; `from __future__ import annotations`
  at the top of new modules.
- FastAPI request/response models: use pydantic `Field(..., description=...)`
  on every field — populates the auto-generated `/docs` schema.
- Prefer plain `pymilvus.MilvusClient` over LlamaIndex abstractions for new
  code (matches `playground/schema_design`, not `sample/`).
- Minimal dependencies — `impl/requirements.txt` should only contain what's
  actually imported; don't add a framework "just in case."

## CI/CD

- `.github/workflows/security-scan.yml` — Trivy `fs` scan (vulnerabilities +
  secrets + misconfig) on every push/PR to `main` and on manual dispatch.
- `.github/workflows/docker-publish.yml` — builds `impl/Dockerfile` (context
  = `impl/`) and pushes to **GHCR** (`ghcr.io/<owner>/startup-funding-rag`,
  tagged with short commit SHA + `latest`). Gated on Trivy: triggered via
  `workflow_run` off the Security Scan workflow's `completed` event, only
  builds/pushes on `conclusion == 'success'` and `head_branch == 'main'` (or
  manual `workflow_dispatch`). Separate from the Cloud Run deploy path
  (`deploy_cloud_run.sh`, via Google Cloud Build) — both build the same
  source independently, not the same image bytes.
- `.github/dependabot.yml` covers `pip` deps in `impl` and `github-actions`
  deps, weekly.

## Security practices in this repo

- `.gitignore` blocks `*password*`, `.env`/`**/.env`, `*.pem`,
  `__pycache__/`, `.venv/`/`venv/`, `.DS_Store`, `.ipynb_checkpoints/`.
- Trivy (see CI/CD above) is the primary automated safety net — this repo is
  **private** on a **personal** GitHub account, so repo-level secret
  scanning / push protection / CodeQL default setup (GitHub Advanced
  Security) aren't available. Revisit if it ever moves to a public repo or
  an org account.
- Before committing notebooks: check both the **source cells** and the
  **stored outputs** for secrets — real credentials have leaked into printed
  cell outputs from a previous run before, not just source code.

## Git / repo setup notes

- This folder (`rag/`) is its own git repo, separate from the
  home-directory-rooted git repo it happens to live inside on disk. Don't
  assume `git` commands here interact with any outer repo.
- Pushes to `github.com` from inside Cursor on this machine are blocked by
  an org-managed guardrail hook (`intuit-git-push-guard`). GitHub-side
  operations (repo creation, pushes) are done manually by the human in a
  plain terminal, not by an agent.
