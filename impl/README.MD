## Python Version
Python 3.10+

## Docker Build
```bash
docker build -t startup-funding-rag .
```

## Docker Build (tagged with the current git commit, matching CI/deploy)
```bash
docker build -t startup-funding-rag:$(git rev-parse --short=7 HEAD) -t startup-funding-rag:latest .
```

## Docker Run
```bash
docker run --rm -p 8080:8080 -e ZILLIZ_URI="" -e ZILLIZ_USER="" -e ZILLIZ_PASSWORD="" -e HF_TOKEN="" -e GROQ_API_KEY="" startup-funding-rag
```

```bash
docker run --rm -p 8080:8080 \
  -v "/Users/hkumar14/git@harsh/rag/creds.config:/run/secrets/creds.config:ro" \
  -e CREDS_CONFIG_PATH=/run/secrets/creds.config \
  -v "/Users/hkumar14/git@harsh/rag/impl/generation_config.json:/app/generation_config.json:ro" \
  startup-funding-rag
```

## Starting Python Server
```bash
python server.py
```

## Trigger indexing:
```bash
curl -X POST http://127.0.0.1:8080/index-all
```
Note: This is destructive. It drops the rag_chunks collection on whatever Zilliz cluster
is configured.

## Search (Get Chunks + LLM):
```bash
curl -X POST http://127.0.0.1:8080/search -H "Content-Type: application/json" -d '{"query": "What are the various funds and its investment size for each?", "top_k": 5}' | jq .
```

```bash
curl -X POST http://127.0.0.1:8080/search -H "Content-Type: application/json" -d '{"query": "What is the initial target for master fund?", "top_k": 5, "generate_answer": false}' | jq .
```

```bash
curl -X POST http://127.0.0.1:8080/search -H "Content-Type: application/json" -d '{"query": "Tell me about T-fund?", "top_k": 5}' | jq .
```

```bash
curl -X POST http://127.0.0.1:8080/search -H "Content-Type: application/json" -d '{"query": "What is JHUB Hackathon league?", "top_k": 5}' | jq .
```

## Tuning generation (system/user prompt, model, temperature, ...) live
```json
{
  "model": "llama-3.3-70b-versatile",
  "temperature": 0.2,
  "max_tokens": 1024,
  "reasoning_format": "",
  "system_prompt": "...",
  "user_prompt_template": "Context:\n$context\n\nQuestion:\n$query"
}
```

Locally: edit `impl/generation_config.json` and send another request.

In GCP: <TBD>

## ------ Cursor Not Reviewed -------
### Not Yet Implemented

These are gaps identified by comparing against `sample/full_impl/` (an
earlier LlamaIndex + Ollama + self-hosted Milvus prototype — see AGENTS.md
for why it's a reference, not the current direction). Not planned work items
with a committed timeline, just a running list of patterns proven out there
that haven't been ported over here yet.

### Indexing
- **Per-record embedding validation before insert** — `sample/full_impl/custom_vector_store.py`'s
  `add()` checks each record has a correctly-dimensioned embedding and skips
  (logging why) just that record rather than trusting the whole batch.
  `rag_core/indexing/indexer.py` embeds and inserts every record as one
  uniform batch with no per-record check.
- **Debug/introspection utilities** — `sample/milvus_client.py`'s `get_stats()`
  (row count / index status) and `peek_chunks()` (fetch the first N rows with
  no query) make it possible to sanity-check that ingestion actually landed
  data correctly. No equivalent exists today — the only way to inspect the
  collection is indirectly, via `/search`.
- **Abstract vector-store interface** (`sample/base_client.py`'s
  `BaseVectorClient` ABC: `insert_nodes` / `search` / `create_collection` /
  `drop_collection`) so the Zilliz/Milvus implementation could be swapped.
  Deliberately not adopted per AGENTS.md's "prefer plain `pymilvus.MilvusClient`"
  standard — noted here as an available pattern, not a bug, in case that
  standard changes.

### Retrieval
- **Reranking** — `sample/full_impl/index_loader.py` wires in a
  `SentenceTransformerRerank` postprocessor: fetch a wider top-k of vector
  hits, then rerank down to a smaller top-n before use. `retrieval/search.py`
  returns raw vector-similarity order with no reranking step.
- **Multi-turn conversation support** — `sample/full_impl/user_prompt.py`
  threads a `history: list[(role, msg)]` into the prompt template. `/search`
  is single-turn/stateless; each request is answered with no memory of
  previous questions in the same conversation.
- **`user_id` / `session_id` tracking** — `sample/full_impl/retrieval_service.py`'s
  `answer_query()` accepts and echoes back a `user_id`/`session_id` per
  request. `/search` has no such concept.
- **Metadata-filtered search exposed via the API** — `search_chunks()` only
  ever applies the hardcoded `is_active == true` filter; there's no way for
  an API caller to additionally filter by e.g. `scheme_type`, `language`, or
  `content_type`, the way `sample/full_impl/milvus_client.py`'s `search()`
  contract allows via its `filters` param (unused in the sample's own
  implementation, but the intent is there).