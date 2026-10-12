# RAG Tunable Parameters & Eval Metrics

## Evaluation Metrics (DeepEval)

Defined in [rag_evaluation_deepeval.ipynb](rag_evaluation_deepeval.ipynb); only the first three are enabled in `all_metrics`.

| Metric | Stage | What it measures |
|---|---|---|
| Contextual Precision | Retrieval | Are retrieved chunks relevant and ranked higher when more relevant? |
| Contextual Recall | Retrieval | Do retrieved chunks cover everything needed for the expected answer? |
| Faithfulness | Answer | Is the generated answer factually consistent with the retrieved chunks (no hallucination)? |
| Contextual Relevancy *(defined, not enabled)* | Retrieval | How relevant, on average, are retrieved chunks to the question? |
| Answer Relevancy *(defined, not enabled)* | Answer | Does the answer actually address the question? |
| Correctness / GEval *(defined, not enabled)* | Answer | How close is the answer to the expected ground-truth answer? |

## Storage / Indexing-Time Parameters

Source: [milvus_client.py](milvus_client.py), [chunking_experiments.ipynb](chunking_experiments.ipynb)

| Parameter | Example value | Purpose |
|---|---|---|
| `chunk_size` | 512 (tokens) / 1024 (chars) | Size of each chunk stored in the index |
| `chunk_overlap` | 64 / 150 | Overlap between consecutive chunks to preserve context |
| Splitter type | `TokenTextSplitter`, `SentenceSplitter`, semantic/percentile-based | How documents are split into chunks |
| `distance_threshold` (semantic chunking) | 0.92–0.95 | Percentile cosine-distance cutoff for a new chunk breakpoint |
| `embed_model` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Embedding model; determines vector space & dimensionality |
| Index type | `HNSW` vs `IVF_FLAT` | Vector index algorithm (recall vs memory tradeoff) |
| `M` | 16 | HNSW graph connectivity (build-time) |
| `efConstruction` | 200 | HNSW build-time accuracy/speed tradeoff |
| `metric_type` | `COSINE` | Distance/similarity metric (vs L2/IP) |

## Retrieval-Time Parameters

Source: [retrieval_config.py](retrieval_config.py), [retrieval_service.py:121](retrieval_service.py)

| Parameter | Example value | Purpose |
|---|---|---|
| `top_k` (`similarity_top_k`) | 8 | Number of chunks returned by initial vector search |
| `rerank` | `True` | Whether to apply a cross-encoder rerank pass |
| `rerank_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used to rerank candidates |
| `rerank_top_n` | 2 | Number of chunks kept after reranking, passed to the LLM |
| `ef` (HNSW) / `nprobe` (IVF) | — | Search-time recall/latency tradeoff; mentioned in eval notebook, not currently wired into config |

## Generation-Time Parameters

Source: [retrieval_config.py](retrieval_config.py)

| Parameter | Example value | Purpose |
|---|---|---|
| `temperature` | 0.1 (Ollama) / 0.4 (Groq) / 0.2 (HF Hub) | Randomness/creativity of generated answer |
| `max_tokens` | 4096 | Max length of generated answer |
| `context_window` | 4096 (Ollama) / 128000 (Groq, HF Hub) | Max input context size for the LLM |
| `ollama_model` / `groq_model` / `hhub_model` | e.g. `llama2`, `qwen/qwen3.6-27b`, `Llama-3.1-8B-Instruct` | Which LLM backend/model generates the answer |
