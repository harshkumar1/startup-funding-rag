import logging
from dataclasses import dataclass


# ---------------------------
# Config / Logger
# ---------------------------
@dataclass
class RetrievalConfig:
    # retrieval
    top_k: int = 8
    rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 2

    # embedding
    embed_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # Ollama
    ollama_model: str = "llama2"  # "gemma3"
    ollama_timeout: int = 3600
    ollama_context_window: int = 4096
    keep_alive: int = 0  # Avoid reloading model
    temperature = 0.1
    stream = False

    # groq
    # decommisioned llama-3.1-70b-versatile other alternate openai/gpt-oss-120b or openai/gpt-oss-20b
    groq_model: str = "qwen/qwen3.6-27b"

    groq_timeout: int = 3600
    groq_context_window: int = 128000
    temperature = 0.4
    stream = False

    # hugging_face_hub
    hhub_model: str = "Llama-3.1-8B-Instruct"
    hhub_timeout: int = 3600
    hhub_context_window: int = 128000
    temperature = 0.2
    max_tokens: int = 4096
    stream = False

    # Redis (sessions / memory)
    redis_host: str = "localhost"
    redis_port: int = 6380
    redis_db: int = 0
    session_ttl_seconds: int = 60 * 60 * 24 * 30  # 30 days

    # concurrency
    max_workers: int = 4
    max_concurrent_llm: int = 2

    # logging
    log_level: int = logging.INFO  # todo: configure logger separately
